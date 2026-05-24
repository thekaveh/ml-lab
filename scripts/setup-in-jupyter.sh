#!/usr/bin/env bash
# Run inside the running jupyterhub container (or via docker exec) to
# install the nnx submodule as an editable package.
#
# Usage (from the host):
#   docker exec -it <project>-jupyterhub /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh
#
# Or (from inside the container terminal, e.g. attached VS Code):
#   /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh
set -euo pipefail

REPO_ROOT="/home/jovyan/work/ml-lab"

if [ ! -d "$REPO_ROOT/nnx" ]; then
    echo "ERROR: $REPO_ROOT/nnx not found." >&2
    echo "Likely causes:" >&2
    echo "  - The ml-lab repo isn't mounted (check that ML_REPO_PATH is set in genai-vanilla/.env)" >&2
    echo "  - The nnx submodule isn't initialized on the host (run: git submodule update --init)" >&2
    exit 1
fi

echo "Installing nnx editable..."
pip install -e "$REPO_ROOT/nnx"

echo
echo "Verifying import..."
python -c "import nnx; print(f'nnx imported from {nnx.__file__}')"

echo
echo "Setup complete. The editable install persists in the jupyterhub-data"
echo "volume; you won't need to re-run this until the image is rebuilt."
