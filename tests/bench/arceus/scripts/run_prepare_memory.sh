#!/bin/bash
# Wrapper script to run memory preparation with the correct Honcho environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HONCHO_ROOT="$SCRIPT_DIR/../../.."

# Activate Honcho's virtual environment
source "$HONCHO_ROOT/.venv/bin/activate"

# Set PYTHONPATH to include Honcho's SDK
export PYTHONPATH="$HONCHO_ROOT/sdks/python/src:$PYTHONPATH"

# Run prepare_memory with all passed arguments
cd "$SCRIPT_DIR/.."
python -m arceus.prepare_memory "$@"
