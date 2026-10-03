import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path('/home/boni/projects/video_training/degradation_undoing')
SHOWCASE_OUTPUTS = PROJECT_ROOT / 'outputs/public_showcase_trajectories_20261003'
MEDIA_ROOT = PROJECT_ROOT / 'docs/media'
TRAJECTORY_ROOT = MEDIA_ROOT / 'trajectories'
POSTER_PATH = MEDIA_ROOT / 'restoration-trajectory-grid.png'
GIF_PATH = MEDIA_ROOT / 'restoration-trajectory-grid.gif'
VIDEO_PATH = MEDIA_ROOT / 'restoration-trajectory-grid.mp4'
DEPTH_IMAGES = [MEDIA_ROOT / 'depth-only' / f'depth-output-{index}.png' for index in range(1, 4)]
SAMPLES = [
    ('EASY', 'pixel factor 12 · seed 20261003', 'easy'),
    ('MEDIUM', 'pixel factor 24 · seed 20261004', 'medium'),
    ('HARD', 'pixel factor 24 · seed 20261005', 'hard'),
]
SAMPLE_VIDEO_NAMES = [('easy', 'restoration-trajectory-easy.mp4'), ('medium', 'restoration-trajectory-medium.mp4'), ('hard', 'restoration-trajectory-hard.mp4')]
PERCENT_INDICES = [(0, '0%', 'f00'), (12, '25%', 'f12'), (24, '50%', 'f24'), (36, '75%', 'f36'), (48, '100%', 'f48')]
POSTER_SIZE = (1350, 1360)
VIDEO_SIZE = (1350, 1200)
FONT_REGULAR = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
COLORS = {'background': '#09121f', 'card': '#111d2d', 'line': '#26364b', 'teal': '#6fe0d2', 'white': '#f5f8fc', 'muted': '#9bacc1', 'soft': '#d9e3ef'}


def sample_frame(sample_name: str, index: int) -> Image.Image:
    path = SHOWCASE_OUTPUTS / sample_name / 'frames' / f'{index:03d}.png'
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert('RGB')


def fit_frame(path: Path, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(Image.open(path).convert('RGB'), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def draw_tile(canvas: Image.Image, draw: ImageDraw.ImageDraw, image: Image.Image, x: int, y: int, size: tuple[int, int]) -> None:
    tile = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    canvas.paste(tile, (x, y))
    draw.rounded_rectangle((x, y, x + size[0] - 1, y + size[1] - 1), radius=9, outline='#34465c', width=1)


def fonts() -> dict[str, ImageFont.FreeTypeFont]:
    return {
        'title': ImageFont.truetype(FONT_BOLD, 46),
        'subtitle': ImageFont.truetype(FONT_REGULAR, 21),
        'row': ImageFont.truetype(FONT_BOLD, 18),
        'head': ImageFont.truetype(FONT_BOLD, 16),
        'small': ImageFont.truetype(FONT_REGULAR, 14),
        'small_bold': ImageFont.truetype(FONT_BOLD, 14),
        'tiny': ImageFont.truetype(FONT_REGULAR, 13),
        'tiny_bold': ImageFont.truetype(FONT_BOLD, 13),
    }


def render_poster(font: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    canvas = Image.new('RGB', POSTER_SIZE, COLORS['background'])
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((38, 30, 430, 61), radius=15, fill='#14283a')
    draw.text((54, 37), 'LTX-2.5  /  49-FRAME INFERENCE', font=font['small_bold'], fill=COLORS['teal'])
    draw.text((38, 76), 'Restoration trajectories', font=font['title'], fill=COLORS['white'])
    draw.text((40, 139), 'Frame positions within three generated clips · 24 fps', font=font['subtitle'], fill=COLORS['muted'])
    left, cell_width, gap = 137, 204, 13
    image_size = (178, 178)
    image_left = (cell_width - image_size[0]) // 2
    for column, (_, percent, frame_name) in enumerate(PERCENT_INDICES):
        x = left + column * (cell_width + gap)
        draw.text((x + 37, 192), f'{percent}  ·  {frame_name}', font=font['head'], fill=COLORS['teal'])

    row_top, row_height, row_gap = 244, 220, 12
    for row_index, (tier, detail, sample_name) in enumerate(SAMPLES):
        x0, y0, x1 = 27, row_top + row_index * (row_height + row_gap), 1323
        y1 = y0 + row_height
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=COLORS['card'], outline=COLORS['line'], width=1)
        draw.text((x0 + 18, y0 + 9), tier, font=font['row'], fill=COLORS['white'])
        pill_x = x0 + 18 + draw.textlength(tier, font=font['row']) + 14
        pill_width = int(draw.textlength(detail, font=font['small']) + 22)
        draw.rounded_rectangle((pill_x, y0 + 8, pill_x + pill_width, y0 + 32), radius=12, fill='#1a2c3d')
        draw.text((pill_x + 11, y0 + 13), detail, font=font['small'], fill=COLORS['soft'])
        for column, (frame_index, _, _) in enumerate(PERCENT_INDICES):
            image = sample_frame(sample_name, frame_index)
            x = left + column * (cell_width + gap) + image_left
            draw_tile(canvas, draw, image, x, y0 + 37, image_size)

    depth_y = row_top + 3 * (row_height + row_gap)
    draw.rounded_rectangle((27, depth_y, 1323, depth_y + 220), radius=18, fill=COLORS['card'], outline=COLORS['line'], width=1)
    draw.text((45, depth_y + 9), 'DEPTH ONLY', font=font['row'], fill=COLORS['white'])
    detail = 'held-out validation · step 3,500'
    pill_x = 181
    draw.rounded_rectangle((pill_x, depth_y + 8, pill_x + 254, depth_y + 32), radius=12, fill='#1a2c3d')
    draw.text((pill_x + 11, depth_y + 13), detail, font=font['small'], fill=COLORS['soft'])
    depth_size = (158, 158)
    depth_positions = [left + (cell_width + gap) * index + (cell_width - depth_size[0]) // 2 for index in (0, 1, 2)]
    for index, path in enumerate(DEPTH_IMAGES):
        image = fit_frame(path, depth_size)
        x = depth_positions[index]
        draw_tile(canvas, draw, image, x, depth_y + 43, depth_size)
        draw.rounded_rectangle((x + 4, depth_y + 47, x + 107, depth_y + 70), radius=11, fill='#0b1522')
        draw.text((x + 11, depth_y + 52), f'SAMPLE {index + 1}', font=font['tiny_bold'], fill=COLORS['soft'])
    text_x = depth_positions[2] + depth_size[0] + 35
    draw.text((text_x, depth_y + 74), 'Prediction frames only', font=font['small_bold'], fill=COLORS['teal'])
    draw.text((text_x, depth_y + 99), 'SUN RGB-D inputs and ground', font=font['small'], fill=COLORS['muted'])
    draw.text((text_x, depth_y + 119), 'truth maps are not included.', font=font['small'], fill=COLORS['muted'])

    footer_y = depth_y + 238
    draw.line((38, footer_y, 1312, footer_y), fill=COLORS['line'], width=2)
    draw.text((38, footer_y + 16), 'Three 49-frame outputs · step-8,000 COCO adapter · seeds 20261003–20261005', font=font['small_bold'], fill=COLORS['soft'])
    draw.text((38, footer_y + 43), 'Time labels are positions in the generated video, not diffusion denoising steps.', font=font['small'], fill=COLORS['muted'])
    draw.text((38, footer_y + 68), 'Depth: terminal held-out predictions from step 3,500 · citation and image credits in README.', font=font['small'], fill=COLORS['muted'])
    draw.text((38, footer_y + 103), 'CC BY-SA 2.0 restoration composite · attribution and reuse details in README', font=font['tiny_bold'], fill=COLORS['teal'])
    return canvas


def render_video_frame(frame_index: int, font: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    canvas = Image.new('RGB', VIDEO_SIZE, COLORS['background'])
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((38, 24, 430, 55), radius=15, fill='#14283a')
    draw.text((54, 31), 'LTX-2.5  /  SYNCHRONIZED CLIPS', font=font['small_bold'], fill=COLORS['teal'])
    percent = frame_index / 48
    draw.text((38, 68), 'Restoration trajectories', font=font['title'], fill=COLORS['white'])
    draw.text((970, 82), f't={percent:.0%} · f{frame_index:02d}/48', font=font['head'], fill=COLORS['teal'])
    draw.text((40, 128), 'Three licensed inputs · same time point in each generated sequence', font=font['subtitle'], fill=COLORS['muted'])
    card_y, card_h, card_w, gap = 174, 420, 424, 15
    image_size = (350, 350)
    for index, (tier, detail, sample_name) in enumerate(SAMPLES):
        x = 24 + index * (card_w + gap)
        draw.rounded_rectangle((x, card_y, x + card_w, card_y + card_h), radius=18, fill=COLORS['card'], outline=COLORS['line'], width=1)
        draw.text((x + 18, card_y + 12), tier, font=font['row'], fill=COLORS['white'])
        draw.text((x + 18, card_y + 39), detail, font=font['tiny'], fill=COLORS['muted'])
        frame = sample_frame(sample_name, frame_index)
        draw_tile(canvas, draw, frame, x + (card_w - image_size[0]) // 2, card_y + 66, image_size)

    depth_y = 620
    draw.rounded_rectangle((24, depth_y, 1326, depth_y + 380), radius=18, fill=COLORS['card'], outline=COLORS['line'], width=1)
    draw.text((44, depth_y + 12), 'DEPTH PREDICTIONS', font=font['row'], fill=COLORS['white'])
    draw.text((290, depth_y + 15), 'saved held-out validation · step 3,500 · terminal frames only', font=font['small'], fill=COLORS['muted'])
    depth_size = (236, 236)
    depth_start, depth_gap = 112, 26
    for index, path in enumerate(DEPTH_IMAGES):
        x = depth_start + index * (depth_size[0] + depth_gap)
        image = fit_frame(path, depth_size)
        draw_tile(canvas, draw, image, x, depth_y + 61, depth_size)
        draw.text((x, depth_y + 310), f'SAMPLE {index + 1}', font=font['small_bold'], fill=COLORS['soft'])
    note_x = 948
    draw.text((note_x, depth_y + 123), 'Predictions only.', font=font['small_bold'], fill=COLORS['teal'])
    draw.text((note_x, depth_y + 151), 'SUN RGB-D source', font=font['small'], fill=COLORS['muted'])
    draw.text((note_x, depth_y + 174), 'RGB and ground-truth', font=font['small'], fill=COLORS['muted'])
    draw.text((note_x, depth_y + 197), 'maps are omitted.', font=font['small'], fill=COLORS['muted'])
    draw.line((38, 1038, 1312, 1038), fill=COLORS['line'], width=2)
    draw.text((38, 1057), 'Time is the video frame position, not a diffusion step. Frame 48 is the final generated frame.', font=font['small'], fill=COLORS['muted'])
    draw.text((38, 1084), 'COCO/Flickr attributions, CC BY-SA 2.0 media terms, and SUN RGB-D citation: repository README.', font=font['tiny_bold'], fill=COLORS['teal'])
    return canvas


def create_source_videos() -> None:
    TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)
    for sample_name, filename in SAMPLE_VIDEO_NAMES:
        frame_pattern = SHOWCASE_OUTPUTS / sample_name / 'frames' / '%03d.png'
        output_path = TRAJECTORY_ROOT / filename
        subprocess.run(['ffmpeg', '-y', '-framerate', '24', '-start_number', '0', '-i', str(frame_pattern), '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    for _, _, sample_name in SAMPLES:
        frames = sorted((SHOWCASE_OUTPUTS / sample_name / 'frames').glob('*.png'))
        metadata = SHOWCASE_OUTPUTS / sample_name / 'metadata.json'
        if len(frames) != 49 or not metadata.is_file():
            raise RuntimeError(f'Expected 49 saved frames and metadata for {sample_name}')
    if any(not path.is_file() for path in DEPTH_IMAGES):
        raise FileNotFoundError('Depth prediction tile missing')
    create_source_videos()
    font = fonts()
    poster = render_poster(font)
    poster.save(POSTER_PATH, optimize=True)
    temporary_root = Path(tempfile.mkdtemp(prefix='portfolio-trajectory-grid-'))
    raw_video = temporary_root / 'trajectory-grid-raw.mp4'
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*'mp4v'), 24, VIDEO_SIZE)
    if not writer.isOpened():
        raise RuntimeError('Could not open a temporary MP4 writer')
    gif_frames = []
    for frame_index in range(49):
        frame = render_video_frame(frame_index, font)
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        if frame_index % 2 == 0 or frame_index == 48:
            gif_frames.append(frame.resize((675, 600), Image.Resampling.LANCZOS))
    for _ in range(24):
        writer.write(cv2.cvtColor(np.asarray(render_video_frame(48, font)), cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run(['ffmpeg', '-y', '-i', str(raw_video), '-c:v', 'libx264', '-preset', 'slow', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(VIDEO_PATH)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gif_frames.extend([gif_frames[-1]] * 12)
    gif_frames[0].save(GIF_PATH, save_all=True, append_images=gif_frames[1:], duration=83, loop=0, optimize=True)
    for path in [POSTER_PATH, GIF_PATH, VIDEO_PATH] + [TRAJECTORY_ROOT / filename for _, filename in SAMPLE_VIDEO_NAMES]:
        print(f'{path} {path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
