#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1
unset PYTHONPATH

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "ERROR: Python executable not found: $PYTHON"
    echo "Activate the HypLoRA environment first."
    exit 1
}

echo "Project root: $PROJECT_ROOT"
echo "Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

"$PYTHON" -P - <<'PY'
import importlib
from pathlib import Path
import peft

peft_root = Path(peft.__file__).resolve().parent

if "site-packages" not in peft_root.parts:
    raise SystemExit(
        f"ERROR: Expected installed PEFT, got: {peft_root}"
    )

print("PEFT:", peft.__file__)

for name in (
    "peft.tuners.lora.layer",
    "peft.tuners.lora.model",
    "peft.tuners.lora.config",
):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()

    if not path.is_relative_to(peft_root):
        raise SystemExit(f"ERROR: Unexpected module path: {path}")

    print(f"Verified: {name} -> {path}")
PY

"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'boolq' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'piqa' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'social_i_qa' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'hellaswag' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'winogrande' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'ARC-Easy' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'ARC-Challenge' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
"$PYTHON" -P -u "$PROJECT_ROOT/commonsense_evaluate.py" \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'openbookqa' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 64 --batch_size 1
