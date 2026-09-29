import datetime
import json
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import torch
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot

from project_config import PROJECT_ROOT
from restoration.io import append_jsonl, atomic_json, seed_everything
from train.ltx_setup import CurriculumLtxTrainer, load_training_config

CONFIG_PATH = PROJECT_ROOT / "configs/ffhq_pixel_curriculum.yaml"
STEP_OVERRIDE = None
TELEGRAM_ENABLED = False
TELEGRAM_UPDATE_INTERVAL = 100
MEMORY_ABORT_AVAILABLE_GIB = 20.0


def cgroup_memory_value(name: str) -> int | str | None:
    entries = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    unified = next((entry.split("::", 1)[1] for entry in entries if "::" in entry), None)
    if unified is None:
        return None
    path = Path("/sys/fs/cgroup") / unified.lstrip("/") / name
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value if value == "max" else int(value)


def process_swap_bytes(process: psutil.Process) -> int:
    status = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8").splitlines()
    value = next(line.split()[1] for line in status if line.startswith("VmSwap:"))
    return int(value) * 1024


def run_training(
    config_path: Path,
    step_override: int | None = None,
    telegram_enabled: bool = False,
    output_override: Path | None = None,
    load_checkpoint_override: Path | None = None,
    checkpoint_interval_override: int | None = None,
    dataloader_workers_override: int | None = None,
    optimization_steps_override: int | None = None,
    preprocessed_data_root_override: Path | None = None,
    validation_interval_override: int | None = None,
    skip_initial_validation_override: bool | None = None,
    resume_training_state_override: bool | None = None,
    no_resume_override: bool | None = None,
    learning_rate_override: float | None = None,
    scheduler_type_override: str | None = None,
    telegram_update_interval: int = TELEGRAM_UPDATE_INTERVAL,
    disable_progress_bars: bool = False,
    trainer_class: type[CurriculumLtxTrainer] = CurriculumLtxTrainer,
    sample_validator: Callable[[list[Path]], None] | None = None,
    allow_distilled_long_run: bool = False,
) -> tuple[Path | None, object]:
    config = load_training_config(config_path, step_override)
    if output_override is not None:
        config.output_dir = str(output_override.resolve())
    if load_checkpoint_override is not None:
        resolved_checkpoint = load_checkpoint_override.resolve()
        config.model.load_checkpoint = str(resolved_checkpoint)
        output_checkpoints = (Path(config.output_dir).resolve() / "checkpoints")
        resume_training_state = (
            resolved_checkpoint.parent == output_checkpoints
            if resume_training_state_override is None
            else resume_training_state_override
        )
        config.checkpoints.no_resume = not resume_training_state
    if no_resume_override is not None:
        config.checkpoints.no_resume = no_resume_override
    if checkpoint_interval_override is not None:
        config.checkpoints.interval = checkpoint_interval_override
    if dataloader_workers_override is not None:
        config.data.num_dataloader_workers = dataloader_workers_override
    if optimization_steps_override is not None:
        config.optimization.steps = optimization_steps_override
    if preprocessed_data_root_override is not None:
        config.data.preprocessed_data_root = str(preprocessed_data_root_override.resolve())
    if validation_interval_override is not None:
        config.validation.interval = validation_interval_override
    if skip_initial_validation_override is not None:
        config.validation.skip_initial_validation = skip_initial_validation_override
    if learning_rate_override is not None:
        config.optimization.learning_rate = learning_rate_override
    if scheduler_type_override is not None:
        config.optimization.scheduler_type = scheduler_type_override
    output_dir = Path(config.output_dir)
    if (
        config.optimization.steps > 10
        and "distilled" in Path(config.model.model_path).name.lower()
        and not allow_distilled_long_run
    ):
        raise RuntimeError("Long flow-matching training on the fixed-step distilled transformer is blocked; use the dev transformer")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    if telegram_enabled and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    state = {
        "status": "running",
        "config": str(config_path),
        "steps": config.optimization.steps,
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "long_run": config.optimization.steps > 10,
        "checkpoint_mode": (
            "base"
            if load_checkpoint_override is None
            else "resume" if not config.checkpoints.no_resume else "weights_only"
        ),
        "trainer_class": trainer_class.__name__,
    }
    atomic_json(output_dir / "run_state.json", state)
    bot = TelegramBot(auto_start=True, working_directory=str(PROJECT_ROOT)) if telegram_enabled else None

    def send_telegram(
        text: str,
        video: str | None = None,
        required: bool = False,
    ) -> None:
        if bot is None:
            return
        last_error = None
        for attempt in range(1, 4):
            try:
                bot.send_message(text=text, video=video, wait=True)
                return
            except Exception as error:
                last_error = error
                append_jsonl(
                    output_dir / "telegram_errors.jsonl",
                    {
                        "attempt": attempt,
                        "text": text,
                        "video": video,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                if attempt < 3:
                    time.sleep(5 * attempt)
        if required:
            raise RuntimeError(f"Telegram delivery failed after three attempts: {last_error}")

    if bot is not None:
        bot.set_state(state)
        send_telegram(
            f"Started {output_dir.name} for {config.optimization.steps} optimizer steps",
            required=True,
        )
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    peak_process_swap = [process_swap_bytes(process)]
    minimum_available = [psutil.virtual_memory().available]
    stop_monitor = threading.Event()
    memory_abort_reason = [None]

    def monitor_rss() -> None:
        while not stop_monitor.wait(0.25):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            peak_process_swap[0] = max(
                peak_process_swap[0],
                process_swap_bytes(process),
            )
            minimum_available[0] = min(minimum_available[0], psutil.virtual_memory().available)
            available_gib = psutil.virtual_memory().available / 2**30
            process_swap = process_swap_bytes(process)
            if memory_abort_reason[0] is None and available_gib < MEMORY_ABORT_AVAILABLE_GIB:
                memory_abort_reason[0] = f"MemAvailable fell below {MEMORY_ABORT_AVAILABLE_GIB:.1f} GiB"
                os.kill(process.pid, signal.SIGINT)
            if memory_abort_reason[0] is None and process_swap > 0:
                memory_abort_reason[0] = "Training process began using swap"
                os.kill(process.pid, signal.SIGINT)

    monitor = threading.Thread(target=monitor_rss, daemon=True)
    monitor.start()
    swap_before = psutil.swap_memory()
    torch.cuda.reset_peak_memory_stats()
    trainer = trainer_class(config)
    callback_started = time.monotonic()
    sent_sample_paths = set()

    def step_callback(step: int, total_steps: int, sample_paths) -> None:
        elapsed = time.monotonic() - callback_started
        eta_seconds = elapsed / max(1, step) * max(0, total_steps - step)
        state.update(
            {
                "current_step": step,
                "progress_fraction": step / total_steps,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
            }
        )
        if bot is not None:
            bot.update_state(**state)
            if step % telegram_update_interval == 0 or step == total_steps:
                send_telegram(
                    f"{output_dir.name}: step {step}/{total_steps}, elapsed {elapsed / 3600:.2f}h, ETA {eta_seconds / 3600:.2f}h"
                )
        new_samples = [Path(path) for path in sample_paths or [] if str(path) not in sent_sample_paths]
        if new_samples and sample_validator is not None:
            sample_validator(new_samples)
        for path in new_samples:
            sent_sample_paths.add(str(path))
            state["last_sample_video"] = str(path)
            if bot is not None:
                send_telegram(
                    text=f"{output_dir.name}: validation sample at step {step}",
                    video=str(path),
                )
        if step % telegram_update_interval == 0 or new_samples or step == total_steps:
            atomic_json(output_dir / "run_state.json", state)
    try:
        checkpoint, stats = trainer.train(
            disable_progress_bars=disable_progress_bars,
            step_callback=step_callback,
        )
        state.update(
            {
                "status": "complete",
                "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "stats": str(stats),
            }
        )
        if bot is not None:
            send_telegram(f"Completed {output_dir.name}: {checkpoint}")
        return checkpoint, stats
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state.update(
            {
                "status": status,
                "failed_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        if bot is not None:
            send_telegram(f"{status.title()} {output_dir.name}: {type(error).__name__}: {error}")
        raise
    finally:
        stop_monitor.set()
        monitor.join()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        swap_in_delta = max(0, swap.sin - swap_before.sin)
        swap_out_delta = max(0, swap.sout - swap_before.sout)
        cgroup_swap_max = cgroup_memory_value("memory.swap.max")
        cgroup_swap_peak = cgroup_memory_value("memory.swap.peak")
        process_swap_passed = peak_process_swap[0] == 0 and cgroup_swap_peak in {None, 0}
        state["memory"] = {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "process_peak_rss_bytes": int(peak_rss[0]),
            "process_peak_swap_bytes": int(peak_process_swap[0]),
            "cgroup_swap_max_bytes": cgroup_swap_max,
            "cgroup_swap_peak_bytes": cgroup_swap_peak,
            "available_gib": memory.available / 2**30,
            "minimum_available_gib": minimum_available[0] / 2**30,
            "swap_used_gib": swap.used / 2**30,
            "swap_in_delta_bytes": swap_in_delta,
            "swap_out_delta_bytes": swap_out_delta,
            "global_swap_activity_observed": swap_in_delta > 0 or swap_out_delta > 0,
            "attributable_swap_passed": process_swap_passed,
            "headroom_passed": (
                minimum_available[0] / 2**30 >= 12.0
                and process_swap_passed
            ),
            "abort_reason": memory_abort_reason[0],
        }
        atomic_json(output_dir / "run_state.json", state)
        if bot is not None:
            bot.stop(wait=True)


def main() -> None:
    checkpoint, stats = run_training(CONFIG_PATH, STEP_OVERRIDE, TELEGRAM_ENABLED)
    print(json.dumps({"checkpoint": str(checkpoint), "stats": str(stats)}, indent=2))


if __name__ == "__main__":
    main()
