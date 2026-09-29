import numpy as np
from PIL import Image

from data_gen.degradation import apply_degradation
from data_gen.trajectory import center_crop_512, cosine_severity, frames_to_tensor
from restoration.schema import DegradationSpec


def interpolate(clean: Image.Image, degraded: Image.Image, severity: float) -> Image.Image:
    if severity <= 0:
        return clean.copy()
    if severity >= 1:
        return degraded.copy()
    clean_array = np.asarray(clean, dtype=np.float32)
    degraded_array = np.asarray(degraded, dtype=np.float32)
    mixed = np.rint(clean_array + severity * (degraded_array - clean_array))
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def build_restoration_frames(
    clean: Image.Image,
    spec: DegradationSpec,
    frame_count: int,
) -> list[Image.Image]:
    clean = center_crop_512(clean)
    degraded = apply_degradation(clean, spec, 1.0)
    frames = [
        interpolate(clean, degraded, cosine_severity(index, frame_count))
        for index in range(frame_count)
    ]
    frames[0] = degraded
    frames[-1] = clean.copy()
    return frames


def build_restoration_trajectory(
    clean: Image.Image,
    spec: DegradationSpec,
    frame_count: int,
):
    return frames_to_tensor(build_restoration_frames(clean, spec, frame_count))
