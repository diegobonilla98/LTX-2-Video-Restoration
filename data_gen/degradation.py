import math
import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from restoration.schema import DegradationSpec


def sample_degradation(seed: int) -> DegradationSpec:
    rng = random.Random(seed)
    draw = rng.random()
    if draw < 0.45:
        kind = "gaussian"
    elif draw < 0.80:
        kind = "pixelation"
    else:
        kind = "combined"
    sigma0 = rng.uniform(1.5, 8.0) if kind in {"gaussian", "combined"} else None
    pixel_factor0 = rng.choice((2, 4, 8, 16)) if kind in {"pixelation", "combined"} else None
    return DegradationSpec(kind=kind, sigma0=sigma0, pixel_factor0=pixel_factor0, seed=seed)


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return image.copy()
    array = np.asarray(image.convert("RGB"))
    blurred = cv2.GaussianBlur(array, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    return Image.fromarray(blurred, mode="RGB")


def pixelate(image: Image.Image, factor: float) -> Image.Image:
    if factor <= 1:
        return image.copy()
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    reduced_width = max(1, math.floor(width / factor))
    reduced_height = max(1, math.floor(height / factor))
    reduced = cv2.resize(array, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(restored, mode="RGB")


def low_resolution(image: Image.Image, factor: float) -> Image.Image:
    if factor <= 1:
        return image.copy()
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    reduced_width = max(1, round(width / factor))
    reduced_height = max(1, round(height / factor))
    reduced = cv2.resize(array, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(restored, mode="RGB")


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    if quality >= 100:
        return image.copy()
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=max(1, min(quality, 100)),
        subsampling=2,
        optimize=False,
    )
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degradation_operators(spec: DegradationSpec) -> tuple[str, ...]:
    if spec.operator_order:
        return spec.operator_order
    operators = []
    if spec.pixel_factor0 is not None:
        operators.append("pixelation")
    if spec.lowres_factor0 is not None:
        operators.append("low_resolution")
    if spec.sigma0 is not None:
        operators.append("gaussian")
    if spec.jpeg_quality0 is not None:
        operators.append("jpeg")
    return tuple(operators)


def apply_degradation(image: Image.Image, spec: DegradationSpec, severity: float) -> Image.Image:
    if severity <= 0:
        return image.copy()
    result = image.convert("RGB")
    for operator in degradation_operators(spec):
        if operator == "pixelation":
            factor = 1.0 + severity * (spec.pixel_factor0 - 1.0)
            result = pixelate(result, factor)
        elif operator == "low_resolution":
            factor = 1.0 + severity * (spec.lowres_factor0 - 1.0)
            result = low_resolution(result, factor)
        elif operator == "gaussian":
            result = gaussian_blur(result, severity * spec.sigma0)
        elif operator == "jpeg":
            quality = round(100.0 - severity * (100.0 - spec.jpeg_quality0))
            result = jpeg_compress(result, quality)
        else:
            raise ValueError(f"Unknown degradation operator {operator}")
    return result
