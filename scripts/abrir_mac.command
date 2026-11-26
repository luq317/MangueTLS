#!/bin/bash
# Duplo clique no Finder: cria o ambiente (1a vez) e abre o MangueTLS Viewer.
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "Usando: $PY ($($PY --version))"
if [ ! -d "$HOME/.mangue_venv" ]; then
  "$PY" -m venv "$HOME/.mangue_venv" || exit 1
  source "$HOME/.mangue_venv/bin/activate"
  pip install --upgrade pip && pip install -r requirements.txt || { echo "Falha na instalacao"; read -p "Enter"; exit 1; }
else
  source "$HOME/.mangue_venv/bin/activate"
fi
python iniciar.py "$@"
read -p "Fechado. Enter para sair."
