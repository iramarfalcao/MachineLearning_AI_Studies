"""
04 — Inferência / geração de texto com um modelo MLX (com ou sem LoRA).

Uso:
    # modelo base puro
    python scripts/04_generate.py --prompt "O que é MLOps?"

    # com os adaptadores LoRA treinados no script 03
    python scripts/04_generate.py --prompt "O que é LoRA?" \
        --adapter models/finetuned/lora-adapters
"""
import argparse

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--adapter", default=None, help="pasta com adaptadores LoRA")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    # Aplica o template de chat se o tokenizer tiver um.
    messages = [{"role": "user", "content": args.prompt}]
    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        prompt = args.prompt

    sampler = make_sampler(temp=args.temp)
    print("=" * 60)
    text = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=args.max_tokens, sampler=sampler, verbose=True,
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
