#!/usr/bin/env bash
# 03 — Fine-tuning com LoRA de um LLM pequeno REAL usando mlx-lm.
#
# Usa o dataset em data/processed/ (train.jsonl / valid.jsonl) gerado pelo
# script 01. Baixa um modelo base pequeno do Hugging Face (na 1a vez) e treina
# só os adaptadores LoRA — cabe tranquilo em 24 GB.
#
# Uso:
#   ./scripts/03_finetune_lora.sh
#   MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit" ITERS=300 ./scripts/03_finetune_lora.sh
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Modelo base pequeno já em MLX (4-bit). Troque via variável MODEL se quiser.
MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
ITERS="${ITERS:-200}"
ADAPTER_DIR="models/finetuned/lora-adapters"

echo "==> Modelo base : $MODEL"
echo "==> Iterações   : $ITERS"
echo "==> Dados       : data/processed/{train,valid}.jsonl"
echo "==> Adaptadores : $ADAPTER_DIR"
echo

mkdir -p "$ADAPTER_DIR"

# mlx_lm.lora espera uma pasta 'data' com train.jsonl/valid.jsonl.
python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data data/processed \
  --adapter-path "$ADAPTER_DIR" \
  --iters "$ITERS" \
  --batch-size 4 \
  --num-layers 8 \
  --learning-rate 1e-4 \
  --steps-per-report 20 \
  --steps-per-eval 100 \
  --save-every 100

echo
echo "==> LoRA treinado. Teste com:"
echo "    python -m mlx_lm generate --model $MODEL \\"
echo "        --adapter-path $ADAPTER_DIR \\"
echo "        --prompt 'O que é MLOps?'"
echo
echo "==> Para FUNDIR os adaptadores no modelo (gerar um modelo standalone):"
echo "    python -m mlx_lm fuse --model $MODEL \\"
echo "        --adapter-path $ADAPTER_DIR \\"
echo "        --save-path models/finetuned/merged-model"
