import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_gen.trajectory import center_crop_512
from project_config import COCO_ROOT, FONT_MANIFEST_PATH, IMAGE_SIZE

FONT_INDEX: dict[str, dict] | None = None


def font_index() -> dict[str, dict]:
    global FONT_INDEX
    if FONT_INDEX is None:
        payload = json.loads(FONT_MANIFEST_PATH.read_text(encoding="utf-8"))
        FONT_INDEX = {record["font_id"]: record for record in payload["fonts"]}
    return FONT_INDEX


def split_lines(text: str, line_count: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    line_count = min(line_count, len(words))
    lines = []
    start = 0
    for line_index in range(line_count):
        remaining_words = len(words) - start
        remaining_lines = line_count - line_index
        take = max(1, round(remaining_words / remaining_lines))
        lines.append(" ".join(words[start : start + take]))
        start += take
    if start < len(words):
        lines[-1] = f"{lines[-1]} {' '.join(words[start:])}"
    return lines


def plain_background(rng: random.Random) -> Image.Image:
    color_a = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    color_b = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    axis = np.linspace(0.0, 1.0, IMAGE_SIZE, dtype=np.float32)
    gradient = color_a[None, :] * (1.0 - axis[:, None]) + color_b[None, :] * axis[:, None]
    array = np.repeat(gradient[:, None, :], IMAGE_SIZE, axis=1)
    if rng.random() < 0.5:
        array = np.transpose(array, (1, 0, 2))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def choose_foreground(background: Image.Image, rng: random.Random) -> tuple[int, int, int]:
    luminance = float(np.asarray(background, dtype=np.float32).mean())
    base = 16 if luminance >= 128 else 239
    jitter = rng.randint(-12, 12)
    value = max(0, min(255, base + jitter))
    return value, value, value


def render_text_record(record: dict) -> Image.Image:
    rng = random.Random(record["degradation"]["seed"])
    background_relpath = record.get("background_relpath")
    if background_relpath:
        background = center_crop_512(Image.open(COCO_ROOT / background_relpath))
    else:
        background = plain_background(rng)
    render = record["render"]
    lines = split_lines(record["text"], render["line_count"])
    text = "\n".join(lines)
    font_path = Path(font_index()[record["font_id"]]["path"])
    font_size = render["font_size"]
    spacing = render["line_spacing"]
    while font_size >= 28:
        font = ImageFont.truetype(str(font_path), size=font_size)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = probe.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= 440 and height <= 400 and height / len(lines) >= 28:
            break
        font_size -= 2
    if font_size < 28:
        raise ValueError(f"Text does not fit with minimum glyph size for sample {record['sample_id']}")
    padding = 18
    layer = Image.new("RGBA", (width + padding * 2, height + padding * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    foreground = choose_foreground(background, rng)
    draw.multiline_text(
        (layer.width / 2, padding - bbox[1]),
        text,
        font=font,
        fill=(*foreground, 255),
        spacing=spacing,
        align="center",
        anchor="ma",
        stroke_width=render["stroke_width"],
        stroke_fill=(*foreground, 255),
    )
    rotated = layer.rotate(render["rotation_degrees"], resample=Image.Resampling.BICUBIC, expand=True)
    max_x = IMAGE_SIZE - rotated.width
    max_y = IMAGE_SIZE - rotated.height
    if max_x < 0 or max_y < 0:
        raise ValueError(f"Rotated text does not fit for sample {record['sample_id']}")
    x = rng.randint(0, max_x)
    y = rng.randint(0, max_y)
    background.paste(rotated, (x, y), rotated)
    return background.convert("RGB")

