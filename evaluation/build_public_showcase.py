import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path('/home/boni/projects/video_training/degradation_undoing')
GALLERY_ROOT = PROJECT_ROOT / 'outputs/coco_pixel_aggressive_49/heldout_gallery'
DEPTH_ROOT = PROJECT_ROOT / 'docs/media/depth-only'
OUTPUT_ROOT = PROJECT_ROOT / 'docs/media'
OUTPUT_VIDEO = OUTPUT_ROOT / 'restoration-depth-grid.mp4'
OUTPUT_GIF = OUTPUT_ROOT / 'restoration-depth-grid.gif'
OUTPUT_POSTER = OUTPUT_ROOT / 'restoration-depth-grid.png'
OUTPUT_SIZE = (1080, 1600)
FONT_REGULAR = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
RESTORATION_ROWS = [
    ('EASY', 'pixel factor 12', '04_easy_ae779fc68e9d54c794bf'),
    ('MEDIUM', 'pixel factor 24', '05_medium_27a26fd2d3823b98fe45'),
    ('HARD', 'pixel factor 24', '09_hard_a81391182fa6795d607d'),
]
RESTORATION_COLUMNS = [('DEGRADED INPUT', 'degraded.png'), ('SAVED MODEL OUTPUT', 'restored.png'), ('CLEAN TARGET', 'clean.png')]
DEPTH_IMAGES = [DEPTH_ROOT / f'depth-output-{index}.png' for index in range(1, 4)]
DEPTH_COLUMNS = ['PREDICTION 01', 'PREDICTION 02', 'PREDICTION 03']
COLORS = {'background': '#09121f', 'card': '#111d2d', 'line': '#26364b', 'teal': '#6fe0d2', 'white': '#f5f8fc', 'muted': '#9bacc1', 'soft': '#d9e3ef'}


def image_crop(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert('RGB')
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def draw_image(canvas: Image.Image, draw: ImageDraw.ImageDraw, path: Path, x: int, y: int, size: tuple[int, int]) -> None:
    image = image_crop(path, size)
    canvas.paste(image, (x, y))
    draw.rounded_rectangle((x, y, x + size[0] - 1, y + size[1] - 1), radius=9, outline='#34465c', width=1)


def render_frame(active_row: int | None, index_text: str, fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    canvas = Image.new('RGB', OUTPUT_SIZE, COLORS['background'])
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((38, 30, 365, 61), radius=15, fill='#14283a')
    draw.text((54, 37), 'LTX-2.5  /  VIDEO + DEPTH', font=fonts['small_bold'], fill=COLORS['teal'])
    draw.text((38, 76), 'Restoration, plus a look at depth', font=fonts['title'], fill=COLORS['white'])
    draw.text((40, 142), 'Saved examples from two exploratory LTX-2.5 experiments', font=fonts['body'], fill=COLORS['muted'])
    if index_text:
        draw.rounded_rectangle((882, 40, 1040, 72), radius=15, fill='#172536', outline=COLORS['line'], width=1)
        draw.text((899, 48), index_text, font=fonts['small_bold'], fill=COLORS['soft'])

    left, gap = 36, 14
    column_width = 326
    image_size = (244, 184)
    image_left = (column_width - image_size[0]) // 2
    row_top, row_height, row_gap = 218, 249, 11
    for col_idx, (label, _) in enumerate(RESTORATION_COLUMNS):
        x = left + col_idx * (column_width + gap)
        draw.text((x + 18, 191), label, font=fonts['column'], fill=COLORS['muted'])

    row_specs = [(tier, factor, index) for index, (tier, factor, _) in enumerate(RESTORATION_ROWS)]
    row_specs.append(('DEPTH OUTPUTS', 'held-out validation · step 3,500', 3))
    for row_idx, (title, detail, source_index) in enumerate(row_specs):
        x0, y0 = 27, row_top + row_idx * (row_height + row_gap)
        x1, y1 = 1053, y0 + row_height
        is_active = active_row == row_idx
        accent = COLORS['teal'] if is_active else COLORS['line']
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=COLORS['card'], outline=accent, width=3 if is_active else 1)
        title_color = COLORS['teal'] if is_active else COLORS['white']
        draw.text((x0 + 18, y0 + 10), title, font=fonts['row'], fill=title_color)
        pill_left = x0 + 18 + draw.textlength(title, font=fonts['row']) + 14
        pill_width = max(150, int(draw.textlength(detail, font=fonts['small']) + 23))
        pill_width = min(pill_width, x1 - pill_left - 25)
        draw.rounded_rectangle((pill_left, y0 + 10, pill_left + pill_width, y0 + 37), radius=13, fill='#1a2c3d')
        draw.text((pill_left + 11, y0 + 16), detail, font=fonts['small'], fill=COLORS['soft'])
        if is_active:
            draw.text((x1 - 133, y0 + 15), f'ROW {row_idx + 1} / 4', font=fonts['tiny_bold'], fill=COLORS['teal'])
        if row_idx < 3:
            _, _, directory = RESTORATION_ROWS[row_idx]
            for col_idx, (_, filename) in enumerate(RESTORATION_COLUMNS):
                x = left + col_idx * (column_width + gap) + image_left
                draw_image(canvas, draw, GALLERY_ROOT / directory / filename, x, y0 + 51, image_size)
        else:
            for col_idx, path in enumerate(DEPTH_IMAGES):
                x = left + col_idx * (column_width + gap) + image_left
                draw_image(canvas, draw, path, x, y0 + 51, image_size)
                draw.rounded_rectangle((x + 5, y0 + 55, x + 130, y0 + 81), radius=12, fill='#0b1522')
                draw.text((x + 13, y0 + 61), DEPTH_COLUMNS[col_idx], font=fonts['tiny_bold'], fill=COLORS['soft'])

    footer_y = 1280
    draw.line((38, footer_y, 1042, footer_y), fill=COLORS['line'], width=2)
    draw.text((38, footer_y + 18), 'Restoration: COCO held-out endpoints · checkpoint step 8,000', font=fonts['small_bold'], fill=COLORS['soft'])
    draw.text((38, footer_y + 46), 'Depth: terminal predictions from saved held-out validation videos · checkpoint step 3,500', font=fonts['small_bold'], fill=COLORS['soft'])
    draw.text((38, footer_y + 77), 'The restoration rows compare input → output → target; they are not generated intermediate frames.', font=fonts['small'], fill=COLORS['muted'])
    draw.text((38, footer_y + 103), 'Depth tiles show prediction only. SUN RGB-D source frames and ground truth are not included.', font=fonts['small'], fill=COLORS['muted'])
    draw.text((38, footer_y + 139), 'Credits, SUN RGB-D citation, and media license: README', font=fonts['tiny_bold'], fill=COLORS['teal'])
    return canvas


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    required = [GALLERY_ROOT / directory / filename for _, _, directory in RESTORATION_ROWS for _, filename in RESTORATION_COLUMNS] + DEPTH_IMAGES
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(map(str, missing)))
    fonts = {
        'title': ImageFont.truetype(FONT_BOLD, 42),
        'body': ImageFont.truetype(FONT_REGULAR, 22),
        'column': ImageFont.truetype(FONT_BOLD, 16),
        'row': ImageFont.truetype(FONT_BOLD, 19),
        'small_bold': ImageFont.truetype(FONT_BOLD, 15),
        'small': ImageFont.truetype(FONT_REGULAR, 14),
        'tiny_bold': ImageFont.truetype(FONT_BOLD, 13),
    }
    render_frame(None, '', fonts).save(OUTPUT_POSTER, optimize=True)
    temporary_root = Path(tempfile.mkdtemp(prefix='restoration-depth-grid-'))
    raw_video = temporary_root / 'restoration-depth-grid-raw.mp4'
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*'mp4v'), 12, OUTPUT_SIZE)
    if not writer.isOpened():
        raise RuntimeError('Could not open a temporary MP4 writer')
    gif_frames = []
    for row_idx in range(4):
        frame = render_frame(row_idx, f'ROW {row_idx + 1} / 4', fonts)
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        for _ in range(18):
            writer.write(bgr)
        gif_frames.append(frame.resize((540, 800), Image.Resampling.LANCZOS))
    writer.release()
    subprocess.run(['ffmpeg', '-y', '-i', str(raw_video), '-c:v', 'libx264', '-preset', 'slow', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUTPUT_VIDEO)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gif_frames[0].save(OUTPUT_GIF, save_all=True, append_images=gif_frames[1:], duration=1400, loop=0, optimize=True)
    for path in (OUTPUT_POSTER, OUTPUT_GIF, OUTPUT_VIDEO):
        print(f'{path} {path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
