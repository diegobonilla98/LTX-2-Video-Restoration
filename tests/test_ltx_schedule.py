import pytest

from project_config import DISTILLED_SIGMAS
from train.ltx_setup import FixedDistilledScheduler


def test_distilled_validation_uses_exact_checkpoint_schedule() -> None:
    values = FixedDistilledScheduler().execute(8).tolist()
    assert values == pytest.approx(DISTILLED_SIGMAS, abs=5e-8)


def test_distilled_validation_rejects_generic_step_counts() -> None:
    with pytest.raises(RuntimeError, match="requires 8 steps"):
        FixedDistilledScheduler().execute(30)
