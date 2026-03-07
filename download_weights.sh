#!/bin/bash
# Download LINO_UniPS pretrained weights
#
# Downloads the model weights from HuggingFace into the
# weights/ directory of this plugin.
#
# Usage:
#   cd mrLINOUniPS
#   bash download_weights.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
WEIGHTS_FILE="${WEIGHTS_DIR}/lino.pth"

HF_URL="https://huggingface.co/houyuanchen/lino/resolve/main/lino.pth"

echo "=== LINO_UniPS Weights Download ==="

# Check if already downloaded (plugin dir or torch cache)
TORCH_CACHE="${HOME}/.cache/torch/hub/checkpoints/lino.pth"

if [ -f "${WEIGHTS_FILE}" ]; then
    echo "Weights already present: ${WEIGHTS_FILE}"
    echo "  Size: $(du -h "${WEIGHTS_FILE}" | cut -f1)"
    echo "To re-download, remove the file first."
    exit 0
fi

if [ -f "${TORCH_CACHE}" ]; then
    echo "Weights found in torch cache: ${TORCH_CACHE}"
    echo "  Size: $(du -h "${TORCH_CACHE}" | cut -f1)"
    echo "Creating symlink: ${WEIGHTS_FILE} -> ${TORCH_CACHE}"
    mkdir -p "${WEIGHTS_DIR}"
    ln -sf "${TORCH_CACHE}" "${WEIGHTS_FILE}"
    exit 0
fi

mkdir -p "${WEIGHTS_DIR}"

echo "Downloading weights from HuggingFace (~338 MB)..."
curl -L -o "${WEIGHTS_FILE}" "${HF_URL}"

# Also copy to torch cache so hubconf.py finds it
mkdir -p "$(dirname "${TORCH_CACHE}")"
cp "${WEIGHTS_FILE}" "${TORCH_CACHE}"

# Verify
if [ -f "${WEIGHTS_FILE}" ] && [ -s "${WEIGHTS_FILE}" ]; then
    echo ""
    echo "Download complete:"
    echo "  ${WEIGHTS_FILE} ($(du -h "${WEIGHTS_FILE}" | cut -f1))"
    echo "  Also cached at: ${TORCH_CACHE}"
else
    echo "ERROR: Download failed or file is empty."
    exit 1
fi
