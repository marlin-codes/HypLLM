#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="hyplora_test"
PYTHON_VERSION="3.12"

TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

FLASH_ATTN_VERSION="2.8.3.post1"

TRANSFORMERS_VERSION="4.50.0"
ACCELERATE_VERSION="1.10.1"
SAFETENSORS_VERSION="0.6.2"
DATASETS_VERSION="4.2.0"
HUGGINGFACE_HUB_VERSION="0.35.3"
TOKENIZERS_VERSION="0.21.4"
NUMPY_VERSION="1.26.4"
PANDAS_VERSION="2.2.3"
SCIPY_VERSION="1.15.3"
TQDM_VERSION="4.67.1"
FIRE_VERSION="0.7.1"
WANDB_VERSION="0.22.2"
PROTOBUF_VERSION="4.25.8"
PYARROW_VERSION="25.0.0"

PEFT_VERSION="0.15.2"

# -----------------------------------------------------------------------------
# Locate repository root
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "HypLoRA Environment Setup"
echo "============================================================"
echo "Repository: $REPO_ROOT"
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "PyTorch: $TORCH_VERSION"
echo "CUDA wheel: cu128"
echo "FlashAttention-2: $FLASH_ATTN_VERSION"
echo "Transformers: $TRANSFORMERS_VERSION"
echo "PEFT: $PEFT_VERSION"
echo "============================================================"
echo

# -----------------------------------------------------------------------------
# Check Conda
# -----------------------------------------------------------------------------

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found."
    exit 1
fi

# Make conda activate available inside this script.
eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------------
# Create environment if it does not already exist
# -----------------------------------------------------------------------------

echo "[1/7] Checking Conda environment..."

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Reusing it."
else
    echo "Creating environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo
echo "Using Python:"
which python
python --version
echo

# -----------------------------------------------------------------------------
# Upgrade packaging tools
# -----------------------------------------------------------------------------

echo "[2/7] Preparing Python packaging tools..."

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# -----------------------------------------------------------------------------
# Install PyTorch
# -----------------------------------------------------------------------------

echo "[3/7] Installing PyTorch $TORCH_VERSION + CUDA 12.8..."

python -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX_URL"

python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# -----------------------------------------------------------------------------
# Install FlashAttention 2
# -----------------------------------------------------------------------------

echo
echo "[4/7] Installing FlashAttention-2 $FLASH_ATTN_VERSION..."

python -m pip install \
    packaging \
    psutil \
    ninja

MAX_JOBS="${MAX_JOBS:-4}" \
python -m pip install \
    "flash-attn==$FLASH_ATTN_VERSION" \
    --no-build-isolation \
    --no-cache-dir

python - <<'PY'
import flash_attn
print("FlashAttention:", flash_attn.__version__)
PY

# -----------------------------------------------------------------------------
# Install pinned dependencies
# -----------------------------------------------------------------------------

echo
echo "[5/7] Installing pinned Python dependencies..."

python -m pip install --no-cache-dir \
    "transformers==$TRANSFORMERS_VERSION" \
    "accelerate==$ACCELERATE_VERSION" \
    "safetensors==$SAFETENSORS_VERSION" \
    "datasets==$DATASETS_VERSION" \
    "pyarrow==$PYARROW_VERSION" \
    "huggingface-hub==$HUGGINGFACE_HUB_VERSION" \
    "tokenizers==$TOKENIZERS_VERSION" \
    "numpy==$NUMPY_VERSION" \
    "pandas==$PANDAS_VERSION" \
    "scipy==$SCIPY_VERSION" \
    "tqdm==$TQDM_VERSION" \
    "fire==$FIRE_VERSION" \
    "wandb==$WANDB_VERSION" \
    "protobuf==$PROTOBUF_VERSION"

# -----------------------------------------------------------------------------
# Install the pinned upstream PEFT version
# -----------------------------------------------------------------------------

echo
echo "[6/7] Installing PEFT $PEFT_VERSION..."

python -m pip install --no-cache-dir "peft==$PEFT_VERSION"

# Check for dependency conflicts
python -m pip check

# -----------------------------------------------------------------------------
# Apply HypLoRA's custom LoRA implementation
# -----------------------------------------------------------------------------

echo
echo "[7/7] Patching and verifying HypLoRA PEFT files..."

SRC="$REPO_ROOT/peft/tuners/lora"

PEFT_DIR=$(python -P -c \
    'import importlib.metadata as m; print(m.distribution("peft").locate_file("peft"))')

DST="$PEFT_DIR/tuners/lora"
FILES=(layer.py model.py config.py)

# Ensure the destination is the installed package
case "$PEFT_DIR" in
    */site-packages/peft) ;;
    *)
        echo "ERROR: Unexpected PEFT installation: $PEFT_DIR"
        exit 1
        ;;
esac

# Validate all files before patching
for f in "${FILES[@]}"; do
    test -f "$SRC/$f" || {
        echo "ERROR: Missing custom source: $SRC/$f"
        exit 1
    }

    test -f "$DST/$f" || {
        echo "ERROR: Missing installed file: $DST/$f"
        exit 1
    }
done

# Back up the installed files before modifying them
STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP="$HOME/.cache/hyplora/peft-backups/$STAMP"
mkdir -p "$BACKUP"

for f in "${FILES[@]}"; do
    cp -p "$DST/$f" "$BACKUP/$f"
done

echo "Custom source: $SRC"
echo "Installed PEFT: $PEFT_DIR"
echo "Backup: $BACKUP"

# Copy only the three customized files
for f in "${FILES[@]}"; do
    cp "$SRC/$f" "$DST/$f"

    cmp -s "$SRC/$f" "$DST/$f" || {
        echo "ERROR: Verification failed for $f"
        exit 1
    }

    rm -f "$DST/__pycache__/${f%.py}".*.pyc
    echo "PATCH OK: $f"
done

# Verify the imports from outside the repository
(
    cd /tmp
    unset PYTHONPATH

    HYPLORA_PEFT_SOURCE="$SRC" python -P - <<'PY'
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import peft

installed = Path(
    importlib.metadata.distribution("peft").locate_file("peft")
).resolve()

actual = Path(peft.__file__).resolve().parent

assert actual == installed, (
    f"Unexpected PEFT import: {actual}"
)

source = Path(os.environ["HYPLORA_PEFT_SOURCE"])

print("PEFT version:", peft.__version__)
print("PEFT package:", actual)

for filename in ("layer.py", "model.py", "config.py"):
    name = f"peft.tuners.lora.{filename[:-3]}"
    module = importlib.import_module(name)

    loaded = Path(module.__file__).resolve()
    expected = (installed / "tuners" / "lora" / filename).resolve()

    assert loaded == expected, f"Unexpected module path: {loaded}"

    source_hash = hashlib.sha256(
        (source / filename).read_bytes()
    ).hexdigest()

    installed_hash = hashlib.sha256(
        loaded.read_bytes()
    ).hexdigest()

    assert source_hash == installed_hash, (
        f"Patched contents differ: {filename}"
    )

    print(f"VERIFIED: {name} -> {loaded}")

print("SUCCESS: All three customized PEFT files are active.")
PY
)

echo
echo "HypLoRA environment setup completed successfully."
