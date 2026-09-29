import gc
import json
import os
from pathlib import Path

import torch
from PIL import Image

from data_gen.render_text import render_text_record
from data_gen.restoration_trajectory import build_restoration_frames
from data_gen.trajectory import build_frames, center_crop_512, frames_to_tensor
from ltx_trainer.model_loader import (
    embedding_weight_paths,
    load_embeddings_processor,
    load_text_encoder,
    load_video_vae_encoder,
    read_video_scale_factors,
)
from project_config import (
    COCO_ROOT,
    FFHQ_ROOT,
    FRAME_COUNT,
    FRAME_RATE,
    MANIFEST_ROOT,
    PRECOMPUTED_ROOT,
    TEXT_ENCODER_PATH,
    TRANSFORMER_PATH,
    VIDEO_VAE_PATH,
)
from restoration.io import atomic_json, atomic_torch_save, read_jsonl
from restoration.schema import DegradationSpec

DATASET_NAME = "ffhq_pixel_curriculum"
SPLIT = "train"
PHASE = "both"
LIMIT = None
OVERWRITE = False
RETAIN_RGB = SPLIT in {"validation", "test"}
DEVICE = "cuda"
PROGRESS_CALLBACK = None
FRAME_COUNT_OVERRIDE = None


def load_clean(record: dict) -> Image.Image:
    if record["clean_kind"] == "natural":
        roots = {"coco2017": COCO_ROOT, "ffhq": FFHQ_ROOT}
        return center_crop_512(Image.open(roots[record["source_root"]] / record["source_relpath"]))
    return render_text_record(record)


def record_spec(record: dict) -> DegradationSpec:
    payload = record["degradation"]
    return DegradationSpec(
        kind=payload["kind"],
        sigma0=payload["sigma0"],
        pixel_factor0=payload["pixel_factor0"],
        seed=payload["seed"],
        jpeg_quality0=payload.get("jpeg_quality0"),
        lowres_factor0=payload.get("lowres_factor0"),
        operator_order=tuple(payload.get("operator_order", ())),
    )


def record_frame_count(record: dict) -> int:
    if FRAME_COUNT_OVERRIDE is not None:
        return FRAME_COUNT_OVERRIDE
    return int(record.get("render", {}).get("frame_count", FRAME_COUNT))


def frames_for_record(record: dict, clean: Image.Image, frame_count: int) -> list[Image.Image]:
    if record.get("render", {}).get("trajectory_builder") == "fixed_endpoint_homotopy_v1":
        return build_restoration_frames(clean, record_spec(record), frame_count)
    return build_frames(clean, record_spec(record), record["trajectory"], frame_count)


def atomic_save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def atomic_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def load_records() -> list[dict]:
    path = MANIFEST_ROOT / DATASET_NAME / f"{SPLIT}.jsonl"
    records = list(read_jsonl(path))
    return records if LIMIT is None else records[:LIMIT]


def precompute_conditions(records: list[dict], destination: Path) -> None:
    text_encoder = load_text_encoder(
        TEXT_ENCODER_PATH,
        device=DEVICE,
        dtype=torch.bfloat16,
        load_in_8bit=False,
    )
    processor = load_embeddings_processor(
        embedding_weight_paths(TRANSFORMER_PATH, TEXT_ENCODER_PATH),
        gemma_model_path=TEXT_ENCODER_PATH,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    prompt_outputs = {}
    for index, record in enumerate(records):
        output = destination / "conditions" / f"{record['sample_id']}.pt"
        if output.is_file() and not OVERWRITE:
            prompt_outputs.setdefault(record["prompt"], output)
            continue
        existing = prompt_outputs.get(record["prompt"])
        if existing is not None and existing.is_file():
            atomic_hardlink(existing, output)
            continue
        with torch.inference_mode():
            hidden_states, attention_mask = text_encoder.encode([record["prompt"]])[0]
            video_features, audio_features = processor.feature_extractor(hidden_states, attention_mask, "left")
        payload = {
            "video_prompt_embeds": video_features[0].cpu().contiguous(),
            "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
        }
        if audio_features is not None:
            payload["audio_prompt_embeds"] = audio_features[0].cpu().contiguous()
        atomic_torch_save(output, payload)
        prompt_outputs[record["prompt"]] = output
        if (index + 1) % 25 == 0:
            print(json.dumps({"phase": "conditions", "complete": index + 1, "total": len(records)}))
            if PROGRESS_CALLBACK is not None:
                PROGRESS_CALLBACK("conditions", index + 1, len(records))
    del processor
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()


def precompute_latents(records: list[dict], destination: Path) -> None:
    vae = load_video_vae_encoder(VIDEO_VAE_PATH, device=DEVICE, dtype=torch.bfloat16)
    scale_factors = read_video_scale_factors(VIDEO_VAE_PATH)
    for index, record in enumerate(records):
        output = destination / "latents" / f"{record['sample_id']}.pt"
        if output.is_file() and not OVERWRITE:
            continue
        frame_count = record_frame_count(record)
        if frame_count < 1 or (frame_count - 1) % scale_factors.time:
            raise RuntimeError(
                f"Frame count {frame_count} must satisfy (frame_count - 1) % {scale_factors.time} == 0"
            )
        clean = load_clean(record)
        frames = frames_for_record(record, clean, frame_count)
        video = frames_to_tensor(frames).unsqueeze(0).to(device=DEVICE, dtype=torch.bfloat16)
        with torch.inference_mode():
            latents = vae(video)
        if latents.shape[2] != 1 + (len(frames) - 1) // scale_factors.time:
            raise RuntimeError(f"Unexpected latent temporal shape {tuple(latents.shape)} for {record['sample_id']}")
        payload = {
            "latents": latents[0].cpu().contiguous(),
            "num_frames": latents.shape[2],
            "height": latents.shape[3],
            "width": latents.shape[4],
            "fps": FRAME_RATE,
        }
        atomic_torch_save(output, payload)
        if RETAIN_RGB:
            for frame_index, frame in enumerate(frames):
                atomic_save_image(destination.parent / "rgb" / record["sample_id"] / f"{frame_index:03d}.png", frame)
        if (index + 1) % 10 == 0:
            print(json.dumps({"phase": "latents", "complete": index + 1, "total": len(records)}))
        if PROGRESS_CALLBACK is not None and ((index + 1) % 100 == 0 or index + 1 == len(records)):
            PROGRESS_CALLBACK("latents", index + 1, len(records))
    del vae
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    records = load_records()
    destination = PRECOMPUTED_ROOT / DATASET_NAME / SPLIT / ".precomputed"
    destination.mkdir(parents=True, exist_ok=True)
    if PHASE in {"conditions", "both"}:
        precompute_conditions(records, destination)
    if PHASE in {"latents", "both"}:
        precompute_latents(records, destination)
    metadata = {
        "dataset_name": DATASET_NAME,
        "split": SPLIT,
        "records": len(records),
        "phase": PHASE,
        "transformer": str(TRANSFORMER_PATH),
        "text_encoder": str(TEXT_ENCODER_PATH),
        "video_vae": str(VIDEO_VAE_PATH),
        "pixel_frame_count": FRAME_COUNT if FRAME_COUNT_OVERRIDE is None else FRAME_COUNT_OVERRIDE,
        "adaptive_frame_counts": sorted({record_frame_count(record) for record in records}),
    }
    atomic_json(destination.parent / "precompute_metadata.json", metadata)


if __name__ == "__main__":
    main()
