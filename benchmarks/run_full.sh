#!/usr/bin/env bash
# Run the full benchmark suite for tinyquant.
#
# Each step is an independent `uv run python ...` invocation, so the GPU is
# fully released between runs. Re-running this script skips outputs that
# already exist (delete the JSON if you want to redo a step).
#
# Env:
#   MODEL          (default NousResearch/Llama-2-7b-hf)
#   PATTERN        (default model.layers.*)
#   SKIP_EXISTING  (default 1; set to 0 to overwrite)

set -u

MODEL="${MODEL:-NousResearch/Llama-2-7b-hf}"
PATTERN="${PATTERN:-model.layers.*}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

NF4_KW='{"block_size": 64}'
HQQ_KW='{"nbits": 4, "group_size": 64}'

run() {
    local output="$1"; shift
    if [[ "$SKIP_EXISTING" = "1" && -f "$output" ]]; then
        echo "=== SKIP $output (exists) ==="
        return 0
    fi
    echo
    echo "=== $(date +%H:%M:%S) START $output ==="
    if uv run python -m tinyquant_bench "$@" --output "$output"; then
        echo "=== $(date +%H:%M:%S) OK    $output ==="
    else
        echo "=== $(date +%H:%M:%S) FAIL  $output (continuing) ==="
    fi
}

# --- Quality (lm-eval-harness) ---

run baseline.json eval \
    --model "$MODEL" --dtype bfloat16 \
    --method none --backend none \
    --tasks zero_shot --batch-size 4

run tq_nf4.json eval \
    --model "$MODEL" --dtype bfloat16 \
    --method nf4 --backend tinyquant --pattern "$PATTERN" \
    --method-kwargs "$NF4_KW" \
    --tasks zero_shot --batch-size 8

run native_nf4.json eval \
    --model "$MODEL" --dtype bfloat16 \
    --method nf4 --backend native --pattern "$PATTERN" \
    --method-kwargs "$NF4_KW" \
    --tasks zero_shot --batch-size 8

run tq_hqq.json eval \
    --model "$MODEL" --dtype bfloat16 \
    --method hqq --backend tinyquant --pattern "$PATTERN" \
    --method-kwargs "$HQQ_KW" \
    --tasks zero_shot --batch-size 8

run native_hqq.json eval \
    --model "$MODEL" --dtype bfloat16 \
    --method hqq --backend native --pattern "$PATTERN" \
    --method-kwargs "$HQQ_KW" \
    --tasks zero_shot --batch-size 8

# --- Speed (forward latency) ---

run speed_baseline.json speed \
    --model "$MODEL" --dtype bfloat16 \
    --method none --backend none \
    --batch-size 1 --seq-len 128 --n-iters 100 --n-warmup 10

run speed_tinyquant_nf4.json speed \
    --model "$MODEL" --dtype bfloat16 \
    --method nf4 --backend tinyquant --pattern "$PATTERN" \
    --method-kwargs "$NF4_KW" \
    --batch-size 1 --seq-len 128 --n-iters 100 --n-warmup 10

run speed_native_nf4.json speed \
    --model "$MODEL" --dtype bfloat16 \
    --method nf4 --backend native --pattern "$PATTERN" \
    --method-kwargs "$NF4_KW" \
    --batch-size 1 --seq-len 128 --n-iters 100 --n-warmup 10

run speed_tinyquant_hqq.json speed \
    --model "$MODEL" --dtype bfloat16 \
    --method hqq --backend tinyquant --pattern "$PATTERN" \
    --method-kwargs "$HQQ_KW" \
    --batch-size 1 --seq-len 128 --n-iters 100 --n-warmup 10

run speed_native_hqq.json speed \
    --model "$MODEL" --dtype bfloat16 \
    --method hqq --backend native --pattern "$PATTERN" \
    --method-kwargs "$HQQ_KW" \
    --batch-size 1 --seq-len 128 --n-iters 100 --n-warmup 10

echo
echo "=== done ==="
ls -la *.json 2>/dev/null
