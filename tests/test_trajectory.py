import numpy as np
from PIL import Image

from data_gen.trajectory import build_frames, build_trajectory, cosine_severity
from restoration.schema import DegradationSpec


def clean_image() -> Image.Image:
    yy, xx = np.mgrid[:512, :512]
    array = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_cosine_schedule_endpoints_and_monotonicity() -> None:
    values = [cosine_severity(index) for index in range(25)]
    assert values[0] == 1.0
    assert abs(values[-1]) < 1e-12
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_progressive_and_oneshot_share_exact_endpoints() -> None:
    image = clean_image()
    spec = DegradationSpec("combined", 4.0, 8, 7)
    progressive = build_frames(image, spec, "progressive")
    oneshot = build_frames(image, spec, "oneshot")
    assert np.array_equal(np.asarray(progressive[0]), np.asarray(oneshot[0]))
    assert np.array_equal(np.asarray(progressive[-1]), np.asarray(image))
    assert np.array_equal(np.asarray(oneshot[-1]), np.asarray(image))
    assert all(np.array_equal(np.asarray(frame), np.asarray(image)) for frame in oneshot[1:])


def test_video_tensor_shape_and_range() -> None:
    tensor = build_trajectory(clean_image(), DegradationSpec("gaussian", 3.0, None, 2), "progressive")
    assert tensor.shape == (3, 25, 512, 512)
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0

