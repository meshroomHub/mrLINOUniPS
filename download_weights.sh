#!/bin/bash
# Download LINO_UniPS pretrained weights
#
# Downloads the model weights from HuggingFace into the
# weights/ directory of this plugin:
#   - lino.pth:     LINO-UniPS normal model (default)
#   - lino_pbr.pth: LINO-UniPS PBR model (normals + albedo), for the
#                   'Estimate Albedo' option or the 'pbr' normal model
#
# Usage:
#   cd mrLINOUniPS
#   bash download_weights.sh          # lino.pth
#   bash download_weights.sh pbr      # lino_pbr.pth
#   bash download_weights.sh all      # both

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
HF_BASE_URL="https://huggingface.co/houyuanchen/lino/resolve/main"

download() {
    local name="$1"
    local file="${WEIGHTS_DIR}/${name}"

    echo "=== LINO_UniPS Weights Download: ${name} ==="

    # Check if already downloaded
    if [ -f "${file}" ]; then
        echo "Weights already present: ${file}"
        echo "  Size: $(du -h "${file}" | cut -f1)"
        echo "To re-download, remove the file first."
        return 0
    fi

    mkdir -p "${WEIGHTS_DIR}"

    echo "Downloading weights from HuggingFace (~338 MB)..."
    curl -L -o "${file}" "${HF_BASE_URL}/${name}"

    # Verify
    if [ -f "${file}" ] && [ -s "${file}" ]; then
        echo ""
        echo "Download complete:"
        echo "  ${file} ($(du -h "${file}" | cut -f1))"
    else
        echo "ERROR: Download failed or file is empty."
        exit 1
    fi
}

case "${1:-normal}" in
    normal) download lino.pth ;;
    pbr) download lino_pbr.pth ;;
    all) download lino.pth; download lino_pbr.pth ;;
    *) echo "Usage: bash download_weights.sh [normal|pbr|all]"; exit 1 ;;
esac
