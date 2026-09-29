import json

from train.curriculum import CurriculumSampler, CurriculumSchedule


def write_schedule(path) -> CurriculumSchedule:
    path.write_text(
        json.dumps(
            {
                "seed": 17,
                "total_steps": 10,
                "anchors": [
                    {"step": 1, "weights": {"easy": 1.0, "medium": 0.0, "hard": 0.0}},
                    {"step": 5, "weights": {"easy": 0.5, "medium": 0.4, "hard": 0.1}},
                    {"step": 10, "weights": {"easy": 0.2, "medium": 0.3, "hard": 0.5}},
                ],
                "tiers": {"easy": ["e"], "medium": ["m"], "hard": ["h"]},
            }
        ),
        encoding="utf-8",
    )
    return CurriculumSchedule(path)


def test_curriculum_interpolates_and_retains_easy_examples(tmp_path) -> None:
    schedule = write_schedule(tmp_path / "curriculum.json")
    assert schedule.weights_for_step(1) == {"easy": 1.0, "medium": 0.0, "hard": 0.0}
    assert schedule.weights_for_step(10) == {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    middle = schedule.weights_for_step(7)
    assert 0.2 < middle["easy"] < 0.5
    assert 0.1 < middle["hard"] < 0.5
    assert abs(sum(middle.values()) - 1.0) < 1e-6


def test_curriculum_plan_is_deterministic_and_resume_exact(tmp_path) -> None:
    schedule = write_schedule(tmp_path / "curriculum.json")
    full = CurriculumSampler(schedule, ["e", "m", "h"], 10, 0, 1).build_plan()
    repeated = CurriculumSampler(schedule, ["e", "m", "h"], 10, 0, 1).build_plan()
    resumed = CurriculumSampler(schedule, ["e", "m", "h"], 10, 4, 1).build_plan()
    assert full == repeated
    assert resumed == full[4:]
    assert full[0]["tier"] == "easy"
