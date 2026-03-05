#!/usr/bin/env bash
# setup.sh — one-command setup for HandNotes RAG
set -e

echo "=== HandNotes RAG — Setup ==="

# 1. Python deps
echo "[1/4] Installing Python dependencies …"
pip install -r requirements.txt

# 2. .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[2/4] Created .env — please add your Mistral API key!"
else
    echo "[2/4] .env already exists, skipping."
fi

# 3. Ollama
if ! command -v ollama &> /dev/null; then
    echo "[3/4] Ollama not found. Install from: https://ollama.ai"
else
    echo "[3/4] Ollama found. Pulling llama3.2 …"
    ollama pull llama3.2
fi

# 4. Done
echo ""
echo "=== Setup complete! ==="
echo "  1. Edit .env and add your MISTRAL_API_KEY"
echo "  2. Start Ollama:   ollama serve"
echo "  3. Run the app:    streamlit run app.py"
