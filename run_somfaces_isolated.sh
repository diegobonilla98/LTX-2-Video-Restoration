#!/usr/bin/env bash
set -euo pipefail

cd /home/boni/projects/video_training/degradation_undoing
source /home/boni/ai/envs/dgx-dl/bin/activate

for sample_dir in outputs/somfaces_aligned_degradation_evaluation/samples/*; do
    sample_id="$(basename "$sample_dir")"
    if [[ ! -f "$sample_dir/ltx.png" ]]; then
        echo "ISOLATED_START:$sample_id" | tee -a logs/somfaces_aligned_isolated.log
        SOMFACES_SAMPLE_ID="$sample_id" python -m evaluation.generate_somfaces_degradation 2>&1 | tee -a logs/somfaces_aligned_isolated.log
    fi
done

SOMFACES_EVALUATE_ONLY=1 python -m evaluation.generate_somfaces_degradation 2>&1 | tee -a logs/somfaces_aligned_isolated.log
