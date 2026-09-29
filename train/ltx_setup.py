import json
from pathlib import Path

import torch
import yaml
import ltx_trainer.validation_runner as validation_runner_module
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.validation_runner import ValidationRunner
from torch.utils.data import DataLoader

from project_config import DISTILLED_SIGMAS, VIDEO_LORA_TARGETS
from restoration.io import append_jsonl, atomic_json
from restoration.lora import save_lora_metadata
from train.curriculum import CurriculumSampler, CurriculumSchedule

FORBIDDEN_TRAINABLE_FRAGMENTS = (
    "audio",
    "video_to_audio",
    "audio_to_video",
    "connector",
    "text_projection",
)


class FixedDistilledScheduler:
    def execute(self, steps: int) -> torch.Tensor:
        expected = len(DISTILLED_SIGMAS) - 1
        if steps != expected:
            raise RuntimeError(f"Distilled LTX validation requires {expected} steps, received {steps}")
        return torch.tensor(DISTILLED_SIGMAS, dtype=torch.float32)


class ExactDistilledValidationRunner(ValidationRunner):
    def _run_denoising(self, *args, **kwargs):
        original = validation_runner_module.LTX2Scheduler
        validation_runner_module.LTX2Scheduler = FixedDistilledScheduler
        try:
            return super()._run_denoising(*args, **kwargs)
        finally:
            validation_runner_module.LTX2Scheduler = original


def load_training_config(config_path: Path, step_override: int | None = None) -> LtxTrainerConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if step_override is not None:
        payload["optimization"]["steps"] = step_override
        payload["checkpoints"]["interval"] = max(1, step_override)
        payload["validation"]["interval"] = None
        payload["validation"]["samples"] = []
    return LtxTrainerConfig(**payload)


class VerifiedLtxTrainer(LtxvTrainer):
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        super().__init__(trainer_config)
        if "distilled" in Path(trainer_config.model.model_path).name.lower():
            self._validation_runner = ExactDistilledValidationRunner(
                config=trainer_config.validation,
                model_path=trainer_config.model.model_path,
                text_encoder_path=trainer_config.model.text_encoder_path,
                video_vae_path=trainer_config.model.video_vae_path,
                audio_vae_path=trainer_config.model.audio_vae_path,
                load_text_encoder_in_8bit=trainer_config.acceleration.load_text_encoder_in_8bit,
            )

    def _setup_lora(self) -> None:
        super()._setup_lora()
        trainable_names = [name for name, parameter in self._transformer.named_parameters() if parameter.requires_grad]
        if not trainable_names:
            raise RuntimeError("LoRA setup produced no trainable parameters")
        forbidden = [
            name
            for name in trainable_names
            if any(fragment in name.lower() for fragment in FORBIDDEN_TRAINABLE_FRAGMENTS)
        ]
        unmatched = [
            name
            for name in trainable_names
            if not any(f".{target}." in name for target in VIDEO_LORA_TARGETS)
        ]
        if forbidden or unmatched:
            raise RuntimeError(
                json.dumps(
                    {"forbidden_trainables": forbidden[:20], "unmatched_trainables": unmatched[:20]},
                    indent=2,
                )
            )
        counts = {
            "parameter_tensors": len(trainable_names),
            "parameters": sum(
                parameter.numel() for _, parameter in self._transformer.named_parameters() if parameter.requires_grad
            ),
            "names": trainable_names,
        }
        atomic_json(Path(self._config.output_dir) / "lora_targets.json", counts)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        serializable = {key: float(value) for key, value in metrics.items()}
        serializable["global_step"] = self._global_step
        append_jsonl(Path(self._config.output_dir) / "metrics.jsonl", serializable)
        super()._log_metrics(metrics)

    def _save_checkpoint(self) -> Path | None:
        existing = self._checkpoint_paths[-1] if self._checkpoint_paths else None
        expected_suffix = f"step_{self._global_step:05d}.safetensors"
        if existing is not None and existing.name.endswith(expected_suffix) and existing.is_file():
            saved_path = existing
        else:
            saved_path = super()._save_checkpoint()
        if saved_path is not None:
            save_lora_metadata(
                checkpoint_path=saved_path,
                rank=self._config.lora.rank,
                alpha=self._config.lora.alpha,
                target_modules=list(self._config.lora.target_modules),
                base_model=self._config.model.model_path,
            )
            link = Path(self._config.output_dir) / "last.safetensors"
            temporary = link.with_name("last.safetensors.tmp")
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(saved_path.relative_to(link.parent))
            temporary.replace(link)
        return saved_path


class CurriculumLtxTrainer(VerifiedLtxTrainer):
    def _curriculum_path(self) -> Path:
        root = Path(self._config.data.preprocessed_data_root).resolve()
        data_root = root.parent if root.name == ".precomputed" else root
        return data_root / "curriculum.json"

    def _init_dataloader(self) -> None:
        curriculum_path = self._curriculum_path()
        if not curriculum_path.is_file():
            super()._init_dataloader()
            return
        data_sources = self._config.training_strategy.get_data_sources()
        self._dataset = PrecomputedDataset(
            self._config.data.preprocessed_data_root,
            data_sources=data_sources,
        )
        primary_key = next(iter(self._dataset.sample_files))
        sample_ids = [path.stem for path in self._dataset.sample_files[primary_key]]
        self._curriculum_schedule = CurriculumSchedule(curriculum_path)
        initial_step = int(self._resume_state[0])
        self._curriculum_sampler = CurriculumSampler(
            schedule=self._curriculum_schedule,
            sample_ids=sample_ids,
            total_steps=self._config.optimization.steps,
            start_step=initial_step,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )
        plan = self._curriculum_sampler.build_plan()
        atomic_json(
            Path(self._config.output_dir) / "curriculum_plan.json",
            {
                "start_step": initial_step,
                "total_steps": self._config.optimization.steps,
                "samples": plan,
            },
        )
        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            sampler=self._curriculum_sampler,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )
        self._dataloader = self._accelerator.prepare(dataloader)
        logger.info(
            f"Loaded curriculum dataset with {len(self._dataset):,} samples and {len(plan):,} scheduled batches"
        )

    def _record_curriculum_batch(self, batch) -> None:
        if hasattr(self, "_curriculum_sampler"):
            index_value = batch["idx"]
            index = int(index_value.flatten()[0].item()) if hasattr(index_value, "flatten") else int(index_value)
            self._last_curriculum_index = index
            self._last_curriculum_tier = self._curriculum_sampler.tier_for_index(index)

    def _training_step(self, batch):
        self._record_curriculum_batch(batch)
        return super()._training_step(batch)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if hasattr(self, "_curriculum_schedule"):
            weights = self._curriculum_schedule.weights_for_step(self._global_step)
            for tier, weight in weights.items():
                metrics[f"curriculum/weight_{tier}"] = weight
                metrics[f"curriculum/sampled_{tier}"] = float(self._last_curriculum_tier == tier)
            append_jsonl(
                Path(self._config.output_dir) / "curriculum_trace.jsonl",
                {
                    "global_step": self._global_step,
                    "tier": self._last_curriculum_tier,
                    "dataset_index": self._last_curriculum_index,
                    "weights": weights,
                },
            )
        super()._log_metrics(metrics)
