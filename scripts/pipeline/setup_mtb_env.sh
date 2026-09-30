#!/usr/bin/env bash
# Setup script for the 'mtb' conda environment
# CUDA 12.5 driver; PyTorch cu124 wheels match the installed cuDNN 9.x
set -e

ENV_NAME="mtb"
PYTHON_VERSION="3.10"

echo "=== Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION) ==="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "=== Installing PyTorch 2.4 with CUDA 12.4 ==="
conda run -n "$ENV_NAME" pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing WhisperX (ASR + alignment) ==="
conda run -n "$ENV_NAME" pip install whisperx

echo "=== Installing pyannote.audio (speaker diarization) ==="
conda run -n "$ENV_NAME" pip install pyannote.audio

echo "=== Installing video / image processing ==="
conda run -n "$ENV_NAME" pip install \
    opencv-python-headless \
    "scenedetect[opencv]"

echo "=== Installing slides dependencies (CLIP + SAM2) ==="
conda run -n "$ENV_NAME" pip install \
    transformers \
    Pillow

# SAM2 / SAM2.1 (Meta, 2024) — used by --detector sam2 and gsam2
conda run -n "$ENV_NAME" pip install "sam2" || \
  echo "  sam2 not on PyPI — install from https://github.com/facebookresearch/sam2"

# Grounding DINO — used by --detector gsam2 (text-prompt slide detection)
# Also needs torchvision (already installed above).
conda run -n "$ENV_NAME" pip install groundingdino-py || \
  echo "  groundingdino-py failed — try: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"

# SAM1 (optional legacy fallback)
conda run -n "$ENV_NAME" pip install "git+https://github.com/facebookresearch/segment-anything.git" || \
  echo "  segment-anything not installed (optional; use --detector gsam2 or --detector cv)"

echo "=== Installing yt-dlp (retrieval and download) ==="
conda run -n "$ENV_NAME" pip install yt-dlp

echo "=== Installing OpenAI SDK + Azure Identity + Pydantic ==="
conda run -n "$ENV_NAME" pip install \
    openai \
    azure-identity \
    "pydantic>=2.0"

echo "=== Exposing cuDNN libs to LD_LIBRARY_PATH ==="
# nvidia-cudnn-cu12 installs cuDNN under site-packages/nvidia/cudnn/lib but conda run
# does not add it to LD_LIBRARY_PATH automatically, causing CUDNN_STATUS_NOT_INITIALIZED.
CUDNN_LIB=$(conda run -n "$ENV_NAME" python -c \
  "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))")
conda env config vars set -n "$ENV_NAME" LD_LIBRARY_PATH="$CUDNN_LIB"

echo ""
echo "=== Environment '$ENV_NAME' is ready! ==="
echo "Verify with: conda run -n $ENV_NAME python -c \"import torch, whisperx, cv2, scenedetect, yt_dlp, openai, azure.identity, pydantic; print('All OK')\""
