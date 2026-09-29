import json
import platform
import subprocess
import sys
from pathlib import Path

import diffusers
import easyocr
import fairseq2
import huggingface_hub
import lpips
import ltx_core
import ltx_pipelines
import ltx_trainer
import psutil
import torch
import transformers
import vllm

from project_config import DEV_TRANSFORMER_PATH, OUTPUT_ROOT, TEXT_ENCODER_PATH, TRANSFORMER_PATH, VIDEO_VAE_PATH
from restoration.io import atomic_json, atomic_text

MINIMUM_AVAILABLE_GIB = 12.0
ALLOWED_PIP_CHECK_FRAGMENTS = (
    "fairseq2 0.8.1 has requirement huggingface_hub",
    "fairseq2 0.8.1 has requirement psutil",
    "fairseq2 0.8.1 has requirement transformers",
    "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform",
)


def command_output(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def bf16_smoke() -> dict:
    torch.cuda.reset_peak_memory_stats()
    layer = torch.nn.Linear(1024, 1024, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    inputs = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    loss = layer(inputs).float().square().mean()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
        "finite": bool(torch.isfinite(loss)),
    }


def main() -> None:
    pip_code, pip_output = command_output([sys.executable, "-m", "pip", "check"])
    pip_lines = [line for line in pip_output.splitlines() if line.strip()]
    unexpected_pip = [
        line for line in pip_lines if not any(fragment in line for fragment in ALLOWED_PIP_CHECK_FRAGMENTS)
    ]
    nvidia_code, nvidia_output = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    )
    freeze_code, freeze_output = command_output([sys.executable, "-m", "pip", "freeze"])
    if freeze_code != 0:
        raise RuntimeError("pip freeze failed")
    atomic_text(OUTPUT_ROOT / "readiness" / "pip-freeze-current.txt", freeze_output + "\n")
    scheduler = diffusers.DDIMScheduler(num_train_timesteps=20)
    scheduler.set_timesteps(4)
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    smoke = bf16_smoke()
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "packages": {
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "diffusers": diffusers.__version__,
            "vllm": vllm.__version__,
            "ltx_core": ltx_core.__name__,
            "ltx_pipelines": ltx_pipelines.__name__,
            "ltx_trainer": ltx_trainer.__name__,
            "easyocr": easyocr.__name__,
            "lpips": lpips.__name__,
            "fairseq2": fairseq2.__name__,
        },
        "models": {
            str(TRANSFORMER_PATH): TRANSFORMER_PATH.stat().st_size,
            str(DEV_TRANSFORMER_PATH): DEV_TRANSFORMER_PATH.stat().st_size,
            str(TEXT_ENCODER_PATH): TEXT_ENCODER_PATH.stat().st_size,
            str(VIDEO_VAE_PATH): VIDEO_VAE_PATH.stat().st_size,
        },
        "pip_check": {"returncode": pip_code, "lines": pip_lines, "unexpected": unexpected_pip},
        "nvidia_smi": {"returncode": nvidia_code, "compute_processes": nvidia_output},
        "regressions": {
            "diffusers_scheduler_timesteps": len(scheduler.timesteps),
            "vllm_imported": bool(vllm.__version__),
            "fairseq2_imported": bool(fairseq2.__name__),
        },
        "memory": {
            "available_gib": memory.available / 2**30,
            "swap_used_gib": swap.used / 2**30,
        },
        "bf16_smoke": smoke,
    }
    payload["passed"] = bool(
        not unexpected_pip
        and nvidia_code == 0
        and not nvidia_output
        and memory.available / 2**30 >= MINIMUM_AVAILABLE_GIB
        and smoke["finite"]
        and len(scheduler.timesteps) == 4
        and torch.__version__ == "2.11.0+cu130"
    )
    atomic_json(OUTPUT_ROOT / "readiness" / "environment.json", payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("Environment readiness gate failed")


if __name__ == "__main__":
    main()
