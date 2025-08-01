#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo ""
    echo "To install uv, run one of the following:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  brew install uv"
    echo "  pip install uv"
    echo ""
    echo "For more options, visit: https://docs.astral.sh/uv/installation/"
    exit 1
fi

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
deactivate

