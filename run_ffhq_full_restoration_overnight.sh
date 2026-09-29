#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/home/boni/projects/video_training/degradation_undoing
ENVIRONMENT=/home/boni/ai/envs/dgx-dl/bin/activate
SESSION_NAME=ffhq-full-restoration
TMUX_SOCKET=vtir-ffhq-full-restoration

if [[ "${FFHQ_RESTORATION_TMUX_INNER:-0}" != "1" ]]; then
    if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
        if [[ "${FFHQ_RESTORATION_PROFILE_LOADED:-0}" != "1" ]]; then
            exec env FFHQ_RESTORATION_PROFILE_LOADED=1 bash -ic "cd '$PROJECT_ROOT' && exec ./run_ffhq_full_restoration_overnight.sh"
        fi
        echo "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are unavailable after loading the interactive shell."
        exit 1
    fi
    if tmux -L "$TMUX_SOCKET" has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session $SESSION_NAME already exists."
        echo "Attach: tmux -L $TMUX_SOCKET attach -t $SESSION_NAME"
        exit 1
    fi
    tmux -L "$TMUX_SOCKET" new-session -d \
        -s "$SESSION_NAME" \
        "FFHQ_RESTORATION_TMUX_INNER=1 bash '$PROJECT_ROOT/run_ffhq_full_restoration_overnight.sh'"
    echo "Launched FFHQ full restoration pipeline."
    echo "Attach: tmux -L $TMUX_SOCKET attach -t $SESSION_NAME"
    echo "List: tmux -L $TMUX_SOCKET ls"
    echo "Log: $PROJECT_ROOT/logs/ffhq_full_restoration_overnight_latest.log"
    exit 0
fi

source "$ENVIRONMENT"
cd "$PROJECT_ROOT"
mkdir -p logs
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG="logs/ffhq_full_restoration_overnight_${RUN_STAMP}.log"
ln -sfn "$(basename "$RUN_LOG")" logs/ffhq_full_restoration_overnight_latest.log
exec > >(tee -a "$RUN_LOG") 2>&1

notify() {
    VTIR_TELEGRAM_MESSAGE="$1" PYTHONPATH=. python -m readiness.telegram_notify
}

run_scope() {
    unit=$1
    module=$2
    systemd-run --user --scope \
        -p MemoryHigh=70G \
        -p MemoryMax=90G \
        -p MemorySwapMax=0 \
        --unit="$unit-$RUN_STAMP" \
        /bin/bash -lc "source '$ENVIRONMENT' && cd '$PROJECT_ROOT' && PYTHONPATH=. python -m '$module'"
}

completed() {
    state=$1
    [[ -f "$state" ]] && grep -q '"status": "complete"' "$state" && grep -q '"current_step": 5000' "$state"
}

cache_complete() {
    root=data/precomputed/ffhq_full_restoration/train/.precomputed
    latent_count=$(find "$root/latents" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)
    condition_count=$(find "$root/conditions" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)
    [[ "$latent_count" -eq 12288 && "$condition_count" -eq 12288 ]]
}

smoke_current() {
    state=outputs/readiness/ffhq_full_restoration/summary.json
    [[ -f "$state" ]] && grep -q '"protocol_version": 2' "$state" && grep -q '"passed": true' "$state"
}

finish() {
    status=$?
    trap - EXIT
    if [[ $status -eq 0 ]]; then
        notify "FFHQ full restoration pipeline completed successfully." || true
    else
        notify "FFHQ full restoration pipeline failed with exit code $status. Check $RUN_LOG." || true
    fi
    exit $status
}

trap finish EXIT
notify "FFHQ full restoration pipeline started: manifest, adaptive cache, real smoke, launch gate, then 5000 supervised steps."
PYTHONPATH=. python -m readiness.environment_check
df -h "$PROJECT_ROOT" /mnt/hdd
PYTHONPATH=. python -m data_gen.build_ffhq_full_restoration_dataset
PYTHONPATH=. python -m data_gen.validate_ffhq_full_restoration
notify "FFHQ restoration manifests passed. Running the bounded 17/25/33/49-frame precompute and optimizer smoke."
if ! smoke_current; then
    run_scope vtir-ffhq-restoration-smoke-precompute readiness.precompute_ffhq_full_restoration_smoke
    run_scope vtir-ffhq-restoration-smoke-train readiness.smoke_ffhq_full_restoration
fi
PYTHONPATH=. python -m readiness.validate_ffhq_full_restoration_smoke
notify "Real adaptive-frame optimizer smoke passed with finite loss and changed LoRA weights. Building or resuming the full cache."
if cache_complete; then
    notify "Reusing the completed 12288-item FFHQ restoration latent and condition cache."
else
    run_scope vtir-ffhq-restoration-precompute data_gen.precompute_ffhq_full_restoration
fi
PYTHONPATH=. python -m data_gen.validate_ffhq_full_restoration_cache
PYTHONPATH=. python -m readiness.validate_ffhq_full_restoration_launch
notify "All FFHQ restoration gates passed. Starting the 5000-step supervised continuation from FFHQ aggressive step 5000."
if ! completed outputs/ffhq_full_restoration/run_state.json; then
    run_scope vtir-ffhq-restoration-train train.train_ffhq_full_restoration
fi
