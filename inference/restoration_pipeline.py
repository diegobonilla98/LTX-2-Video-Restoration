import json
import os
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import AUTO_TILING
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    combined_image_conditionings,
    ensure_tiling_config,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ModalitySpec
from ltx_core.types import VideoPixelShape
from data_gen.trajectory import center_crop_512
from project_config import (
    DISTILLED_SIGMAS,
    FRAME_COUNT,
    FRAME_RATE,
    IMAGE_SIZE,
    TEXT_ENCODER_PATH,
    TRANSFORMER_PATH,
    VIDEO_VAE_PATH,
)
from restoration.io import atomic_json
from restoration.lora import effective_lora_strength, validate_lora_base


@dataclass(frozen=True)
class RestorationResult:
    frames: tuple[Image.Image, ...]
    final_image: Image.Image
    latent: torch.Tensor
    metadata: dict


def conservative_final_image(conditioning: Image.Image, generated: Image.Image, strength: float) -> Image.Image:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Output blend strength must be between 0 and 1")
    return Image.blend(conditioning.convert("RGB"), generated.convert("RGB"), strength)


def normalize_sigma_schedule(sigma_schedule) -> tuple[float, ...]:
    values = tuple(float(value) for value in sigma_schedule)
    if len(values) < 2 or abs(values[0] - 1.0) > 1e-5 or abs(values[-1]) > 1e-7:
        raise ValueError("Sigma schedule must begin at 1.0 and end at 0.0")
    values = (1.0, *values[1:-1], 0.0)
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError("Sigma schedule must be strictly decreasing")
    return values


def atomic_save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


class RestorationPipeline:
    def __init__(
        self,
        lora_path: Path | None = None,
        device: str = "cuda",
        frame_count: int = FRAME_COUNT,
        lora_strength: float = 1.0,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        output_blend_strength: float = 1.0,
        transformer_path: Path = TRANSFORMER_PATH,
        sigma_schedule: tuple[float, ...] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.frame_count = frame_count
        self.lora_strength = lora_strength
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.applied_lora_strength = effective_lora_strength(lora_strength, lora_rank, lora_alpha)
        self.output_blend_strength = output_blend_strength
        self.transformer_path = Path(transformer_path)
        if sigma_schedule is None:
            if self.transformer_path.resolve() != TRANSFORMER_PATH.resolve():
                raise RuntimeError("A non-distilled transformer requires an explicit inference sigma schedule")
            sigma_schedule = DISTILLED_SIGMAS
        self.sigma_schedule = normalize_sigma_schedule(sigma_schedule)
        self.lora_metadata = None
        if lora_path is not None:
            self.lora_metadata = validate_lora_base(Path(lora_path), self.transformer_path)
            if self.lora_metadata is not None:
                if int(self.lora_metadata["rank"]) != self.lora_rank:
                    raise RuntimeError("Requested LoRA rank does not match checkpoint metadata")
                if int(self.lora_metadata["alpha"]) != self.lora_alpha:
                    raise RuntimeError("Requested LoRA alpha does not match checkpoint metadata")
        self.model_paths = ModelPaths.from_split(
            transformer_path=str(self.transformer_path),
            text_encoder_path=str(TEXT_ENCODER_PATH),
            video_vae_path=str(VIDEO_VAE_PATH),
        )
        loras = ()
        if lora_path is not None:
            loras = (
                LoraPathStrengthAndSDOps(
                    str(lora_path),
                    self.applied_lora_strength,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                ),
            )
        self.prompt_encoder = PromptEncoder(self.model_paths, dtype=self.dtype, device=self.device)
        self.image_conditioner = ImageConditioner(str(VIDEO_VAE_PATH), dtype=self.dtype, device=self.device)
        self.stage = DiffusionStage.from_checkpoint(
            str(self.transformer_path),
            dtype=self.dtype,
            device=self.device,
            loras=loras,
        )
        self.decoder = VideoDecoder(
            str(VIDEO_VAE_PATH),
            dtype=self.dtype,
            device=self.device,
            diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
        )
        self.lora_path = lora_path
        self.pipeline_name = "single_stage_512_distilled_video_only"

    def close(self) -> None:
        return None

    def _create_conditionings(self, image_path: Path):
        image_input = ImageConditioningInput(str(image_path), 0, 1.0, 0)
        return self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=[image_input],
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                video_encoder=encoder,
                dtype=self.dtype,
                device=self.device,
                color_space=None,
            )
        )

    def _create_noiser(self, generator: torch.Generator):
        return GaussianNoiser(generator=generator)

    @torch.inference_mode()
    def restore(self, image_path: Path, prompt: str, seed: int = 42) -> RestorationResult:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        context = self.prompt_encoder([prompt])[0]
        video_context = context.video_encoding
        conditionings = self._create_conditionings(image_path)
        sigmas = torch.tensor(self.sigma_schedule, dtype=torch.float32, device=self.device)
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(v_context=video_context, a_context=None),
            sigmas=sigmas,
            noiser=self._create_noiser(generator),
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            frames=self.frame_count,
            fps=FRAME_RATE,
            video=ModalitySpec(context=video_context, conditionings=conditionings),
            audio=None,
            max_batch_size=1,
        )
        if video_state is None:
            raise RuntimeError("Video-only diffusion returned no video state")
        scale_factors = tiling_scale_factors_for_vae(str(VIDEO_VAE_PATH))
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=scale_factors,
            vae_checkpoint_path=str(VIDEO_VAE_PATH),
            video_shape=VideoPixelShape(
                batch=1,
                frames=self.frame_count,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                fps=FRAME_RATE,
            ),
            diffvae_optimization=self.decoder.diffvae_optimization,
            device=self.device,
        )
        chunks = list(self.decoder(video_state.latent, tiling, generator=generator, dtype=self.dtype))
        decoded = torch.cat(chunks, dim=0).float().cpu().clamp(0.0, 1.0)
        if decoded.ndim != 4 or decoded.shape[-1] != 3:
            raise RuntimeError(f"Expected decoded video with shape [F,H,W,C], received {tuple(decoded.shape)}")
        frames = tuple(
            Image.fromarray(
                (decoded[frame_index] * 255.0).round().to(torch.uint8).numpy(),
                mode="RGB",
            )
            for frame_index in range(decoded.shape[0])
        )
        conditioning = center_crop_512(Image.open(image_path).convert("RGB"))
        final_image = conservative_final_image(conditioning, frames[-1], self.output_blend_strength)
        frames = (*frames[:-1], final_image)
        metadata = {
            "seed": seed,
            "prompt": prompt,
            "sigmas": list(self.sigma_schedule),
            "checkpoint": str(self.transformer_path),
            "lora": str(self.lora_path) if self.lora_path is not None else None,
            "lora_strength": self.lora_strength,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "applied_lora_strength": self.applied_lora_strength,
            "lora_metadata": self.lora_metadata,
            "output_blend_strength": self.output_blend_strength,
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "frames": len(frames),
            "frame_rate": FRAME_RATE,
            "pipeline": self.pipeline_name,
        }
        return RestorationResult(frames, final_image, video_state.latent.float().cpu(), metadata)


class PersistentRestorationPipeline(RestorationPipeline):
    def __init__(
        self,
        lora_path: Path | None = None,
        device: str = "cuda",
        frame_count: int = FRAME_COUNT,
        lora_strength: float = 1.0,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        output_blend_strength: float = 1.0,
        transformer_path: Path = TRANSFORMER_PATH,
        sigma_schedule: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(
            lora_path,
            device,
            frame_count,
            lora_strength,
            lora_rank,
            lora_alpha,
            output_blend_strength,
            transformer_path,
            sigma_schedule,
        )
        self.stack = ExitStack()
        self.transformer = None
        self.prompt_contexts = {}

    def prepare(self, prompt: str) -> None:
        if self.transformer is not None:
            if prompt not in self.prompt_contexts:
                raise RuntimeError("Persistent restoration pipeline supports one prompt per process")
            return
        self.prompt_contexts[prompt] = self.prompt_encoder([prompt])[0].video_encoding
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=self.frame_count,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            fps=FRAME_RATE,
        )
        latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=self.stage.video_scale_factors)
        video_tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1),
            latent_shape,
            FRAME_RATE,
            scale_factors=self.stage.video_scale_factors,
        )
        self.transformer = self.stack.enter_context(self.stage._transformer_ctx(video_tools=video_tools))

    @torch.inference_mode()
    def restore_final(self, image_path: Path, prompt: str, seed: int = 42) -> RestorationResult:
        self.prepare(prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video_context = self.prompt_contexts[prompt]
        image_input = ImageConditioningInput(str(image_path), 0, 1.0, 0)
        conditionings = self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=[image_input],
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                video_encoder=encoder,
                dtype=self.dtype,
                device=self.device,
                color_space=None,
            )
        )
        sigmas = torch.tensor(self.sigma_schedule, dtype=torch.float32, device=self.device)
        transformer_context = self.stage._transformer_ctx
        self.stage._transformer_ctx = lambda **kwargs: nullcontext(self.transformer)
        try:
            video_state, _ = self.stage(
                denoiser=SimpleDenoiser(v_context=video_context, a_context=None),
                sigmas=sigmas,
                noiser=GaussianNoiser(generator=generator),
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                frames=self.frame_count,
                fps=FRAME_RATE,
                video=ModalitySpec(context=video_context, conditionings=conditionings),
                audio=None,
                max_batch_size=1,
            )
        finally:
            self.stage._transformer_ctx = transformer_context
        if video_state is None:
            raise RuntimeError("Video-only diffusion returned no video state")
        scale_factors = tiling_scale_factors_for_vae(str(VIDEO_VAE_PATH))
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=scale_factors,
            vae_checkpoint_path=str(VIDEO_VAE_PATH),
            video_shape=VideoPixelShape(
                batch=1,
                frames=self.frame_count,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                fps=FRAME_RATE,
            ),
            diffvae_optimization=self.decoder.diffvae_optimization,
            device=self.device,
        )
        chunks = list(self.decoder(video_state.latent, tiling, generator=generator, dtype=self.dtype))
        decoded = torch.cat(chunks, dim=0).float().cpu().clamp(0.0, 1.0)
        if decoded.ndim != 4 or decoded.shape[-1] != 3 or decoded.shape[0] != self.frame_count:
            raise RuntimeError(f"Expected decoded video with shape [F,H,W,C], received {tuple(decoded.shape)}")
        generated_final = Image.fromarray(
            (decoded[-1] * 255.0).round().to(torch.uint8).numpy(),
            mode="RGB",
        )
        conditioning = center_crop_512(Image.open(image_path).convert("RGB"))
        final_image = conservative_final_image(conditioning, generated_final, self.output_blend_strength)
        metadata = {
            "seed": seed,
            "prompt": prompt,
            "sigmas": list(self.sigma_schedule),
            "checkpoint": str(self.transformer_path),
            "lora": str(self.lora_path) if self.lora_path is not None else None,
            "lora_strength": self.lora_strength,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "applied_lora_strength": self.applied_lora_strength,
            "lora_metadata": self.lora_metadata,
            "output_blend_strength": self.output_blend_strength,
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "frames": self.frame_count,
            "frame_rate": FRAME_RATE,
            "pipeline": "single_stage_512_distilled_video_only_persistent_exact",
        }
        return RestorationResult((final_image,), final_image, video_state.latent.float().cpu(), metadata)

    def close(self) -> None:
        self.prompt_contexts.clear()
        self.stack.close()


def save_result(result: RestorationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(result.frames):
        atomic_save_png(output_dir / "frames" / f"{index:03d}.png", frame)
    atomic_save_png(output_dir / "final.png", result.final_image)
    atomic_json(output_dir / "metadata.json", result.metadata)
