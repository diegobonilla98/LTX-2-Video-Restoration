import json
import random
from pathlib import Path

from torch.utils.data import Sampler


class CurriculumSchedule:
    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.seed = int(payload["seed"])
        self.total_steps = int(payload["total_steps"])
        self.anchors = payload["anchors"]
        self.tiers = {name: tuple(values) for name, values in payload["tiers"].items()}
        self.tier_names = tuple(self.tiers)
        self._validate()

    def _validate(self) -> None:
        steps = [int(anchor["step"]) for anchor in self.anchors]
        if steps != sorted(steps) or len(set(steps)) != len(steps):
            raise RuntimeError("Curriculum anchors must have unique increasing steps")
        if steps[0] != 1 or steps[-1] != self.total_steps:
            raise RuntimeError("Curriculum anchors must cover step 1 through total_steps")
        if any(not members for members in self.tiers.values()):
            raise RuntimeError("Every curriculum tier must contain samples")
        for anchor in self.anchors:
            weights = anchor["weights"]
            if set(weights) != set(self.tier_names):
                raise RuntimeError("Every curriculum anchor must contain every tier")
            if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6:
                raise RuntimeError("Curriculum weights must sum to one")

    def weights_for_step(self, step: int) -> dict[str, float]:
        bounded = max(1, min(step, self.total_steps))
        if bounded <= self.anchors[0]["step"]:
            return {key: float(value) for key, value in self.anchors[0]["weights"].items()}
        for left, right in zip(self.anchors, self.anchors[1:]):
            if bounded <= right["step"]:
                width = right["step"] - left["step"]
                fraction = (bounded - left["step"]) / width
                return {
                    tier: float(left["weights"][tier])
                    + fraction * (float(right["weights"][tier]) - float(left["weights"][tier]))
                    for tier in self.tier_names
                }
        return {key: float(value) for key, value in self.anchors[-1]["weights"].items()}


class CurriculumSampler(Sampler[int]):
    def __init__(
        self,
        schedule: CurriculumSchedule,
        sample_ids: list[str],
        total_steps: int,
        start_step: int,
        gradient_accumulation_steps: int,
    ) -> None:
        self.schedule = schedule
        self.total_steps = total_steps
        self.start_step = start_step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        self.tier_indices = {
            tier: tuple(id_to_index[sample_id] for sample_id in members if sample_id in id_to_index)
            for tier, members in schedule.tiers.items()
        }
        missing = {
            tier: len(members) - len(self.tier_indices[tier])
            for tier, members in schedule.tiers.items()
            if len(members) != len(self.tier_indices[tier])
        }
        if missing:
            raise RuntimeError(f"Precomputed curriculum samples are missing: {missing}")
        self.sample_ids = sample_ids
        self._tier_by_index = {
            index: tier for tier, indices in self.tier_indices.items() for index in indices
        }

    def build_plan(self) -> list[dict]:
        rng = random.Random(self.schedule.seed)
        total_batches = self.total_steps * self.gradient_accumulation_steps
        start_batch = self.start_step * self.gradient_accumulation_steps
        plan = []
        for batch_index in range(total_batches):
            step = batch_index // self.gradient_accumulation_steps + 1
            weights = self.schedule.weights_for_step(step)
            draw = rng.random()
            cumulative = 0.0
            selected_tier = self.schedule.tier_names[-1]
            for tier in self.schedule.tier_names:
                cumulative += weights[tier]
                if draw <= cumulative:
                    selected_tier = tier
                    break
            index = self.tier_indices[selected_tier][rng.randrange(len(self.tier_indices[selected_tier]))]
            if batch_index >= start_batch:
                plan.append(
                    {
                        "batch_index": batch_index,
                        "step": step,
                        "tier": selected_tier,
                        "index": index,
                        "sample_id": self.sample_ids[index],
                    }
                )
        return plan

    def tier_for_index(self, index: int) -> str:
        return self._tier_by_index[index]

    def __iter__(self):
        return iter(item["index"] for item in self.build_plan())

    def __len__(self) -> int:
        return (self.total_steps - self.start_step) * self.gradient_accumulation_steps
