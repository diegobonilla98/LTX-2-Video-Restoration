#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/home/boni/projects/video_training/degradation_undoing
ENVIRONMENT=/home/boni/ai/envs/dgx-dl/bin/activate
SESSION_NAME=ffhq-depixelation
TMUX_SOCKET=vtir-ffhq-moderate
UNIT_NAME=vtir-ffhq-sft

if [[ "${FFHQ_TMUX_INNER:-0}" != "1" ]]; then
    if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
        echo "Export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before launching."
        exit 1
    fi
    if tmux -L "$TMUX_SOCKET" has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session $SESSION_NAME already exists."
        echo "Attach with: tmux -L $TMUX_SOCKET attach -t $SESSION_NAME"
        exit 1
    fi
    tmux -L "$TMUX_SOCKET" new-session -d \
        -s "$SESSION_NAME" \
        "FFHQ_TMUX_INNER=1 bash '$PROJECT_ROOT/run_ffhq_overnight.sh'"
    echo "Launched tmux session: $SESSION_NAME"
    echo "Attach: tmux -L $TMUX_SOCKET attach -t $SESSION_NAME"
    echo "Log: $PROJECT_ROOT/logs/ffhq_overnight_latest.log"
    exit 0
fi

source "$ENVIRONMENT"
cd "$PROJECT_ROOT"
mkdir -p logs
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG="logs/ffhq_overnight_${RUN_STAMP}.log"
ln -sfn "$(basename "$RUN_LOG")" logs/ffhq_overnight_latest.log
exec > >(tee -a "$RUN_LOG") 2>&1

notify() {
    VTIR_TELEGRAM_MESSAGE="$1" PYTHONPATH=. python readiness/telegram_notify.py
}

finish() {
    status=$?
    trap - EXIT
    if [[ $status -eq 0 ]]; then
        notify "FFHQ overnight pipeline completed successfully." || true
    else
        notify "FFHQ overnight pipeline failed with exit code $status. Check $RUN_LOG." || true
    fi
    exit $status
}

trap finish EXIT

notify "FFHQ overnight pipeline started in tmux session $SESSION_NAME."
PYTHONPATH=. python readiness/environment_check.py
df -h "$PROJECT_ROOT" /mnt/hdd
PYTHONPATH=. python data_gen/build_ffhq_dataset.py
notify "FFHQ manifests and curriculum were built. Starting LTX precomputation."
PYTHONPATH=. python data_gen/precompute_ffhq.py
PYTHONPATH=. python data_gen/validate_ffhq.py
PYTHONPATH=. python readiness/validate_ffhq_launch.py
notify "FFHQ data and latents passed validation. Starting 6000-step SFT."
systemd-run --user --scope \
    -p MemorySwapMax=0 \
    --unit="$UNIT_NAME" \
    /bin/bash -lc "source '$ENVIRONMENT' && cd '$PROJECT_ROOT' && PYTHONPATH=. python train/train_ffhq.py"
