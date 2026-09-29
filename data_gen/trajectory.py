import math
import random

import numpy as np
import torch
from PIL import Image

from data_gen.degradation import apply_degradation
from project_config import FRAME_COUNT, IMAGE_SIZE
from restoration.schema import DegradationSpec, TrajectoryKind


def cosine_severity(frame_index: int, frame_count: int = FRAME_COUNT) -> float:
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"frame_index must be in [0, {frame_count - 1}]")
    if frame_index == frame_count - 1:
        return 0.0
    progress = frame_index / (frame_count - 1)
    return math.cos(math.pi * progress / 2.0) ** 2


def center_crop_512(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = max(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - IMAGE_SIZE) // 2
    top = (resized.height - IMAGE_SIZE) // 2
    return resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))


def build_frames(
    clean: Image.Image,
    spec: DegradationSpec,
    trajectory: TrajectoryKind,
    frame_count: int = FRAME_COUNT,
) -> list[Image.Image]:
    clean = center_crop_512(clean)
    progressive = [apply_degradation(clean, spec, cosine_severity(index, frame_count)) for index in range(frame_count)]
    progressive[-1] = clean.copy()
    if trajectory == "progressive":
        return progressive
    if trajectory == "oneshot":
        return [progressive[0], *[clean.copy() for _ in range(frame_count - 1)]]
    if trajectory == "shuffled":
        middle = progressive[1:-1]
        random.Random(spec.seed).shuffle(middle)
        return [progressive[0], *middle, clean.copy()]
    raise ValueError(f"Unknown trajectory {trajectory}")


def frames_to_tensor(frames: list[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(frame.convert("RGB"), dtype=np.float32) for frame in frames]
    video = np.stack(arrays, axis=0) / 127.5 - 1.0
    return torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()


def build_trajectory(clean: Image.Image, spec: DegradationSpec, trajectory: TrajectoryKind) -> torch.Tensor:
    return frames_to_tensor(build_frames(clean, spec, trajectory))
