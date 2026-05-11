#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# EAGLE2 Training Script for Qwen2.5-Coder-14B-Instruct
# Two-stage: ShareGPT (Stage A) → BIRD SQL (Stage B)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Paths (override via env) ─────────────────────────────────────────────────
WORKDIR="${WORKDIR:-/mnt/nj-aigc/usr/wangtong2/eagle_sql}"
BASEPATH="${BASEPATH:-$WORKDIR/model/Qwen2.5-Coder-14B-Instruct}"
DATADIR="${DATADIR:-$WORKDIR/data/sharegpt}"
CPDIR="${CPDIR:-$WORKDIR/artifacts/eagle2/checkpoints}"
INPUT_JSONL="${INPUT_JSONL:-$WORKDIR/data/sharegpt_train.jsonl}"
CODEDIR="$WORKDIR/EAGLE-main"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

# ─── Training hyperparams ─────────────────────────────────────────────────────
LR="${LR:-3e-5}"
BS="${BS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
MAX_LEN="${MAX_LEN:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
SAVE_FREQ="${SAVE_FREQ:-5}"

# ─── Feature flags ────────────────────────────────────────────────────────────
SKIP_GE_DATA="${SKIP_GE_DATA:-0}"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE="${LOG_FILE:-$WORKDIR/artifacts/eagle2/run_eagle2_train.log}"
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== EAGLE2 Training Pipeline ==="
log "BASEPATH=$BASEPATH"
log "DATADIR=$DATADIR"
log "INPUT_JSONL=$INPUT_JSONL"
log "CPDIR=$CPDIR"
log "NUM_GPUS=$NUM_GPUS, BS=$BS, LR=$LR, EPOCHS=$NUM_EPOCHS"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Environment preflight
# ═══════════════════════════════════════════════════════════════════════════════
log "--- Step 1: Preflight checks ---"

python3 -c "
import torch, transformers, accelerate
print(f'torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}')
print(f'transformers={transformers.__version__} accelerate={accelerate.__version__}')
assert torch.cuda.device_count() >= 1, 'No GPU available'
" 2>&1 | tee -a "$LOG_FILE"

if [ ! -d "$BASEPATH" ]; then
    log "ERROR: Base model not found at $BASEPATH"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Extract hidden states (skip if already done)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_GE_DATA" = "0" ]; then
    log "--- Step 2: Extract hidden states ---"

    if [ ! -f "$INPUT_JSONL" ]; then
        log "ERROR: Input JSONL not found: $INPUT_JSONL"
        exit 1
    fi

    EXISTING_COUNT=$(find "$DATADIR" -name "*.pt" 2>/dev/null | wc -l) || EXISTING_COUNT=0
    if [ "$EXISTING_COUNT" -gt 0 ]; then
        log "Found $EXISTING_COUNT existing .pt files in $DATADIR, skipping extraction."
        log "Set SKIP_GE_DATA=0 and remove files to re-extract."
    else
        mkdir -p "$DATADIR"
        PYTHONUNBUFFERED=1 python3 -m eagle.ge_data.ge_data_qwen25 \
            --basepath "$BASEPATH" \
            --input-jsonl "$INPUT_JSONL" \
            --output-dir "$DATADIR" \
            --max-len "$MAX_LEN" \
            --batch-size 1 \
            --num-proc 4 \
            2>&1 | tee -a "$LOG_FILE"
    fi
else
    log "--- Step 2: SKIPPED (SKIP_GE_DATA=1) ---"
fi

# Verify data exists
PT_COUNT=$(find "$DATADIR" -name "*.pt" 2>/dev/null | wc -l) || PT_COUNT=0
if [ "$PT_COUNT" -eq 0 ]; then
    log "ERROR: No .pt files found in $DATADIR after extraction"
    exit 1
fi
log "Data ready: $PT_COUNT samples in $DATADIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Train EAGLE2 head
# ═══════════════════════════════════════════════════════════════════════════════
log "--- Step 3: Train EAGLE2 head ---"
mkdir -p "$CPDIR"

cd "$CODEDIR"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    -m eagle.train2.main \
    --basepath "$BASEPATH" \
    --datadir "$DATADIR" \
    --cpdir "$CPDIR" \
    --lr "$LR" \
    --bs "$BS" \
    --num-epochs "$NUM_EPOCHS" \
    --max-len "$MAX_LEN" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --save-freq "$SAVE_FREQ" \
    2>&1 | tee -a "$LOG_FILE"

log "=== Training complete. Checkpoints at: $CPDIR ==="
