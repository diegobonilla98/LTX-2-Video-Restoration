# Aggressive FFHQ continuation

The completed 6,000-step moderate run is preserved under `outputs/releases/ffhq_moderate_v1`. The aggressive stage initializes from its adapter-only copy, writes only to `outputs/ffhq_pixel_aggressive`, and never resumes into or overwrites the moderate output tree. Later launches automatically select the newest aggressive checkpoint and its matching training state.

```bash
ssh dgx-spark
cd /home/boni/projects/video_training/degradation_undoing
export TELEGRAM_BOT_TOKEN='your existing bot token'
export TELEGRAM_CHAT_ID='your existing chat id'
./run_ffhq_aggressive_overnight.sh
```

```bash
tmux -L vtir-ffhq-aggressive attach -t ffhq-depixelation-aggressive
tail -f /home/boni/projects/video_training/degradation_undoing/logs/ffhq_aggressive_overnight_latest.log
tail -f /home/boni/projects/video_training/degradation_undoing/outputs/ffhq_pixel_aggressive/metrics.jsonl
cat /home/boni/projects/video_training/degradation_undoing/outputs/ffhq_pixel_aggressive/run_state.json
```

| Tier | Pixel factors | Effective input sizes |
|---|---|---|
| Easy bridge | 8, 12, 16 | 64×64, 42×42, 32×32 |
| Aggressive | 12, 16, 24 | 42×42, 32×32, 21×21 |
| Very aggressive | 16, 24, 32 | 32×32, 21×21, 16×16 |

| Step | Easy bridge | Aggressive | Extreme |
|---:|---:|---:|---:|
| 1 | 70% | 25% | 5% |
| 1,000 | 45% | 40% | 15% |
| 2,500 | 30% | 40% | 30% |
| 4,000 | 20% | 35% | 45% |
| 5,000 | 15% | 30% | 55% |

The stage starts from the preserved moderate step-6000 LoRA at a reduced `2e-5` learning rate. It trains for 5,000 new optimizer steps, checkpoints every 500 steps, and produces fixed easy/aggressive/very-aggressive 25-frame videos initially and every 500 steps. Telegram receives progress every 100 steps and each validation MP4.

The deterministic plan contains 1,658 easy-bridge, 1,806 aggressive, and 1,536 very-aggressive samples. Its final 1,000 steps contain 170 easy-bridge, 327 aggressive, and 503 very-aggressive samples.
