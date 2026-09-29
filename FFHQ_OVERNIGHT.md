# FFHQ overnight depixelation run

The single launcher is `run_ffhq_overnight.sh`. It does not contain credentials and refuses to start until Telegram variables are present.

```bash
ssh dgx-spark
cd /home/boni/projects/video_training/degradation_undoing
export TELEGRAM_BOT_TOKEN='your existing bot token'
export TELEGRAM_CHAT_ID='your existing chat id'
./run_ffhq_overnight.sh
```

The command returns after creating the tmux session. Training was not launched during implementation.

```bash
tmux -L vtir-ffhq-moderate attach -t ffhq-depixelation
tail -f /home/boni/projects/video_training/degradation_undoing/logs/ffhq_overnight_latest.log
cat /home/boni/projects/video_training/degradation_undoing/outputs/ffhq_pixel_curriculum/run_state.json
tail -f /home/boni/projects/video_training/degradation_undoing/outputs/ffhq_pixel_curriculum/metrics.jsonl
nvidia-smi
```

The launcher performs, in order:

1. Live CUDA, BF16, dependency, idle-GPU, memory, and model checks.
2. Deterministic FFHQ manifests and fixed validation conditions.
3. Resumable LTX text/latent precomputation.
4. Full manifest, identity-isolation, tier, endpoint, tensor, hardlink, and count validation.
5. Final launch gate including Telegram, disk, memory, curriculum smoke, MP4, and checkpoint evidence.
6. A 6,000-step rank-64 LoRA run in a user-owned `MemorySwapMax=0` scope.

The experiment uses 4,096 train identities, 128 validation identities, and 512 test identities from the 52,001 local FFHQ PNGs. Every training identity appears once in each severity tier, producing 12,288 precomputed trajectories without coupling identity to difficulty:

| Tier | Pixel factors |
|---|---|
| Easy | 2, 3, 4 |
| Medium | 4, 6, 8 |
| Hard | 8, 12, 16 |

The sampler linearly interpolates between these anchors:

| Step | Easy | Medium | Hard |
|---:|---:|---:|---:|
| 1 | 100% | 0% | 0% |
| 1,000 | 65% | 30% | 5% |
| 2,500 | 40% | 45% | 15% |
| 4,000 | 25% | 40% | 35% |
| 6,000 | 15% | 30% | 55% |

The deterministic 6,000-step plan contains 2,512 easy, 1,980 medium, and 1,508 hard samples. In its final 1,000 steps it contains 180 easy, 295 medium, and 525 hard samples, so difficult examples become predominant while easy examples remain represented.

Telegram receives pipeline start/failure/completion, precompute updates every 500 completed samples, training progress every 100 steps, and each fixed easy/medium/hard validation MP4 initially and every 500 steps.

Checkpoints are saved every 500 steps with the latest three retained. Full optimizer, scheduler, RNG, and sampler state supports exact resume. Re-running the launcher repeats deterministic manifest validation, skips completed precomputation, and resumes the training output when a valid state exists.

The measured precompute rate projects roughly 3.5 hours for 12,288 trajectories. The measured steady training rate plus validation/checkpoint overhead projects approximately 6–7 hours, for an expected end-to-end duration near 10 hours. Actual time depends on HDD and host contention.
