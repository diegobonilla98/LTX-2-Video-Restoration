import numpy as np
from PIL import Image

from data_gen.build_ffhq_full_restoration_dataset import (
    DISTANCE_UNITS,
    FAMILIES,
    FRAME_COUNTS,
    TIERS,
    cell_for_index,
    spec_for,
)
from data_gen.degradation import apply_degradation
from data_gen.restoration_trajectory import build_restoration_frames
from data_gen.trajectory import cosine_severity


def test_joint_cells_cover_every_family_and_tier() -> None:
    cells = {cell_for_index(index) for index in range(len(FAMILIES) * len(TIERS))}
    assert cells == {(family, tier) for family in FAMILIES for tier in TIERS}


def test_adaptive_frame_counts_match_latent_distance() -> None:
    for tier in TIERS:
        frames = FRAME_COUNTS[tier]
        assert (frames - 1) % 8 == 0
        assert (frames - 1) // 8 == DISTANCE_UNITS[tier]
        assert cosine_severity(0, frames) == 1.0
        assert cosine_severity(frames - 1, frames) == 0.0


def test_every_degradation_has_normalized_clean_endpoint() -> None:
    grid = np.arange(512 * 512 * 3, dtype=np.uint32).reshape(512, 512, 3)
    clean = Image.fromarray((grid % 256).astype(np.uint8), mode="RGB")
    for family in FAMILIES:
        spec = spec_for(family, "hard", 1)
        degraded = apply_degradation(clean, spec, 1.0)
        restored = apply_degradation(clean, spec, 0.0)
        assert degraded.size == clean.size
        assert not np.array_equal(np.asarray(degraded), np.asarray(clean))
        assert np.array_equal(np.asarray(restored), np.asarray(clean))


def test_progressive_trajectory_ends_at_exact_clean_image() -> None:
    rng = np.random.default_rng(7)
    clean = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), mode="RGB")
    for family in FAMILIES:
        frames = build_restoration_frames(clean, spec_for(family, "easy", 2), 17)
        assert len(frames) == 17
        assert np.array_equal(np.asarray(frames[-1]), np.asarray(clean))


def test_restoration_error_is_monotonic() -> None:
    rng = np.random.default_rng(11)
    clean = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), mode="RGB")
    clean_array = np.asarray(clean, dtype=np.float32)
    for family in FAMILIES:
        frames = build_restoration_frames(clean, spec_for(family, "extreme", 3), 49)
        errors = [float(np.mean((np.asarray(frame, dtype=np.float32) - clean_array) ** 2)) for frame in frames]
        assert all(right <= left for left, right in zip(errors, errors[1:]))
