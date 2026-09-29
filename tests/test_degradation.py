import numpy as np
from PIL import Image

from data_gen.degradation import apply_degradation, pixelate, sample_degradation
from restoration.schema import DegradationSpec


def test_zero_severity_preserves_pixels() -> None:
    array = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    image = Image.fromarray(array, mode="RGB")
    for kind, sigma, factor in (
        ("gaussian", 4.0, None),
        ("pixelation", None, 8),
        ("combined", 4.0, 8),
    ):
        spec = DegradationSpec(kind, sigma, factor, 1)
        assert np.array_equal(np.asarray(apply_degradation(image, spec, 0.0)), array)


def test_pixelation_uses_nearest_neighbor_blocks() -> None:
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8), mode="RGB")
    degraded = np.asarray(pixelate(image, 8.0))
    assert np.unique(degraded.reshape(-1, 3), axis=0).shape[0] <= 64


def test_degradation_mixture_is_seed_deterministic() -> None:
    first = [sample_degradation(seed) for seed in range(10_000)]
    second = [sample_degradation(seed) for seed in range(10_000)]
    assert first == second
    proportions = {
        kind: sum(spec.kind == kind for spec in first) / len(first)
        for kind in ("gaussian", "pixelation", "combined")
    }
    assert abs(proportions["gaussian"] - 0.45) < 0.02
    assert abs(proportions["pixelation"] - 0.35) < 0.02
    assert abs(proportions["combined"] - 0.20) < 0.02

