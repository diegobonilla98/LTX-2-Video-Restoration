from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_ROOT = DATA_ROOT / "manifests"
PRECOMPUTED_ROOT = DATA_ROOT / "precomputed"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
COCO_ROOT = Path("/mnt/hdd/coco2017")
FFHQ_ROOT = Path("/mnt/hdd/ffhq")
MINIPILE_ROOT = Path(
    "/mnt/hdd/hf/JeanKaddour___minipile/default/0.0.0/18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0"
)
LTX_SNAPSHOT = Path(
    "/home/boni/.cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/"
    "e8dc69fd26150afbfa20351f6bc9ac384257f9fd"
)
TRANSFORMER_PATH = LTX_SNAPSHOT / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
DEV_TRANSFORMER_PATH = Path(
    "/home/boni/.cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/"
    "5e6e71018ee1756ed329b697a7b4aedc934dfce9/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
)
TEXT_ENCODER_PATH = LTX_SNAPSHOT / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE_PATH = LTX_SNAPSHOT / "vae/ltx-2.5-video-vae-bf16.safetensors"
LTX_SOURCE_ROOT = Path("/home/boni/projects/video_training/LTX-2-v1.3.0")
FONT_MANIFEST_PATH = DATA_ROOT / "font_manifest.json"
FRAME_COUNT = 25
IMAGE_SIZE = 512
FRAME_RATE = 24.0
NATURAL_PROMPT = "Progressively restore the degraded image to the original clean image."
FFHQ_PROMPT = (
    "Progressively depixelate the face while preserving identity, facial geometry, expression, lighting, and background."
)
TEXT_PROMPT = "Progressively restore the degraded text image to the original clean text."
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
VIDEO_LORA_TARGETS = (
    "attn1.to_k",
    "attn1.to_q",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_k",
    "attn2.to_q",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


@dataclass(frozen=True)
class DatasetCounts:
    train: int
    validation: int
    test: int


NATURAL_COUNTS = DatasetCounts(train=90_000, validation=5_000, test=5_000)
TEXT_COUNTS = DatasetCounts(train=100_000, validation=5_000, test=20_000)
RL_TEXT_COUNT = 20_000
