#!/bin/bash
set -e

# Step 1: Ensure uv is installed
if ! command -v uv &> /dev/null
then
    echo "⚠️  uv not found. Installing uv..."
    pip3 install uv
fi

# Step 2: Create virtual env if missing
if [ ! -d ".venv" ]; then
    echo "🪄 Creating virtual environment..."
    uv venv
    uv pip install -e .
fi

# Step 3: Sync dependencies
echo "📦 Installing dependencies..."
uv sync

echo "✅ Environment setup complete!"
echo "👉 To activate: source .venv/bin/activate"
echo "👉 or use : uv run <>"
