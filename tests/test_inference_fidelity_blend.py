import numpy as np
import pytest
from PIL import Image

from inference.restoration_pipeline import conservative_final_image


def test_conservative_final_image_limits_generated_correction() -> None:
    conditioning = Image.fromarray(np.full((32, 32, 3), 50, dtype=np.uint8))
    generated = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
    blended = np.asarray(conservative_final_image(conditioning, generated, 0.2))
    assert np.all(blended == 80)


def test_conservative_final_image_rejects_unsafe_strength() -> None:
    image = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="between 0 and 1"):
        conservative_final_image(image, image, 1.1)
