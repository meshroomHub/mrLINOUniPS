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

# Check if already downloaded
if [ -f "${WEIGHTS_FILE}" ]; then
    echo "Weights already present: ${WEIGHTS_FILE}"
    echo "  Size: $(du -h "${WEIGHTS_FILE}" | cut -f1)"
    echo "To re-download, remove the file first."
    exit 0
fi

mkdir -p "${WEIGHTS_DIR}"

echo "Downloading weights from HuggingFace (~338 MB)..."
curl -L -o "${WEIGHTS_FILE}" "${HF_URL}"

# Verify
if [ -f "${WEIGHTS_FILE}" ] && [ -s "${WEIGHTS_FILE}" ]; then
    echo ""
    echo "Download complete:"
    echo "  ${WEIGHTS_FILE} ($(du -h "${WEIGHTS_FILE}" | cut -f1))"
else
    echo "ERROR: Download failed or file is empty."
    exit 1
fi
