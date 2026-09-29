#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/home/boni/projects/video_training/degradation_undoing
ENVIRONMENT=/home/boni/ai/envs/dgx-dl/bin/activate
SESSION_NAME=coco-depixelation-aggressive-49
TMUX_SOCKET=vtir-coco-aggressive-49

if [[ "${COCO_AGGRESSIVE_49_TMUX_INNER:-0}" != "1" ]]; then
    if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
        echo "Export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before launching."
        exit 1
    fi
    if tmux -L "$TMUX_SOCKET" has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session $SESSION_NAME already exists."
        echo "Attach: tmux -L $TMUX_SOCKET attach -t $SESSION_NAME"
        exit 1
    fi
    tmux -L "$TMUX_SOCKET" new-session -d \
        -s "$SESSION_NAME" \
        "COCO_AGGRESSIVE_49_TMUX_INNER=1 bash '$PROJECT_ROOT/run_coco_aggressive_49_overnight.sh'"
    echo "Launched COCO aggressive 49-frame training pipeline."
    echo "Attach: tmux -L $TMUX_SOCKET attach -t $SESSION_NAME"
    echo "List: tmux -L $TMUX_SOCKET ls"
    echo "Log: $PROJECT_ROOT/logs/coco_aggressive_49_overnight_latest.log"
    exit 0
fi

source "$ENVIRONMENT"
cd "$PROJECT_ROOT"
mkdir -p logs
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG="logs/coco_aggressive_49_overnight_${RUN_STAMP}.log"
ln -sfn "$(basename "$RUN_LOG")" logs/coco_aggressive_49_overnight_latest.log
exec > >(tee -a "$RUN_LOG") 2>&1

notify() {
    VTIR_TELEGRAM_MESSAGE="$1" PYTHONPATH=. python -m readiness.telegram_notify
}

run_scope() {
    unit=$1
    module=$2
    systemd-run --user --scope -p MemorySwapMax=0 --unit="$unit-$RUN_STAMP" /bin/bash -lc "source '$ENVIRONMENT' && cd '$PROJECT_ROOT' && PYTHONPATH=. python -m '$module'"
}

completed() {
    state=$1
    [[ -f "$state" ]] && grep -q '"status": "complete"' "$state"
}

cache_complete() {
    root=data/precomputed/coco_pixel_aggressive_49/train/.precomputed
    latent_count=$(find "$root/latents" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)
    condition_count=$(find "$root/conditions" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)
    [[ "$latent_count" -eq 24576 && "$condition_count" -eq 24576 ]]
}

finish() {
    status=$?
    trap - EXIT
    if [[ $status -eq 0 ]]; then
        notify "COCO aggressive 49-frame depixelation pipeline completed successfully." || true
    else
        notify "COCO aggressive 49-frame depixelation pipeline failed with exit code $status. Check $RUN_LOG." || true
    fi
    exit $status
}

trap finish EXIT
notify "COCO aggressive 49-frame depixelation pipeline started."
PYTHONPATH=. python -m readiness.environment_check
df -h "$PROJECT_ROOT" /mnt/hdd
PYTHONPATH=. python -m data_gen.build_coco_aggressive_49_dataset
notify "COCO manifests and 49-frame curriculum are ready. Running bounded precompute and training smoke."
if [[ ! -f outputs/readiness/coco_aggressive_49/summary.json ]] || ! grep -q '"passed": true' outputs/readiness/coco_aggressive_49/summary.json; then
    run_scope vtir-coco-aggressive-49-smoke-precompute readiness.precompute_coco_aggressive_49_smoke
    run_scope vtir-coco-aggressive-49-smoke-train readiness.smoke_coco_aggressive_49
fi
PYTHONPATH=. python -m readiness.validate_coco_aggressive_49_smoke
notify "Real 49-frame optimizer smoke passed. Checking the full COCO trajectory cache."
if cache_complete; then
    notify "Reusing the completed 24576-item COCO latent and condition cache."
else
    run_scope vtir-coco-aggressive-49-precompute data_gen.precompute_coco_aggressive_49
fi
PYTHONPATH=. python -m data_gen.validate_coco_aggressive_49
PYTHONPATH=. python -m readiness.validate_coco_aggressive_49_launch
notify "COCO aggressive 49-frame data passed every gate. Starting 8000-step supervised continuation."
if ! completed outputs/coco_pixel_aggressive_49/run_state.json; then
    run_scope vtir-coco-aggressive-49-train train.train_coco_aggressive_49
fi
