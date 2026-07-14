"""
05 — Conversão de modelos.

Cobre os dois caminhos mais comuns do sandbox:

  1) Hugging Face (safetensors)  ->  MLX      [recomendado, com quantização]
     python scripts/05_convert_models.py hf \
         --repo Qwen/Qwen2.5-0.5B-Instruct \
         --out models/mlx/qwen2.5-0.5b-4bit --quantize

  2) GGUF (llama.cpp)  ->  MLX (safetensors + inspeção dos tensores)
     python scripts/05_convert_models.py gguf \
         --file models/gguf/modelo.gguf \
         --out models/mlx/do-gguf

Notas importantes:
  - O caminho HF->MLX é o oficial (mlx_lm.convert) e reconstrói a config.
  - GGUF->MLX é intrinsecamente parcial: o GGUF guarda os pesos + metadados,
    mas a config de arquitetura precisa ser remontada por modelo. Este script
    lê o GGUF com o MLX nativo (dequantiza os pesos) e salva em safetensors,
    além de exportar os metadados — suficiente para inspeção e para casos em
    que você tem o config.json do modelo original.
  - Para o caminho INVERSO (MLX/HF -> GGUF), use o llama.cpp:
        python convert_hf_to_gguf.py <pasta_hf> --outfile modelo.gguf
    (veja o README, seção "GGUF <-> MLX").
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert_hf(args):
    """HF safetensors -> MLX usando mlx_lm.convert."""
    from mlx_lm import convert

    q = bool(args.quantize)
    print(f"Convertendo {args.repo} -> {args.out} "
          f"(quantize={'4-bit' if q else 'não'})")
    convert(
        hf_path=args.repo,
        mlx_path=args.out,
        quantize=q,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )
    print(f"OK. Modelo MLX em {args.out}")
    print("Teste: python scripts/04_generate.py --model", args.out,
          "--prompt 'Olá!'")


def convert_gguf(args):
    """GGUF -> MLX: lê com mlx.core nativo e salva safetensors + metadados."""
    import mlx.core as mx

    if not os.path.exists(args.file):
        raise SystemExit(f"Arquivo GGUF não encontrado: {args.file}")

    print(f"Lendo GGUF: {args.file}")
    # mx.load entende .gguf nativamente: retorna (weights, metadata).
    # ATENÇÃO: o loader do MLX só dequantiza alguns tipos (F16, F32, Q4_0,
    # Q4_1, Q8_0). k-quants (Q4_K_M, Q6_K, etc.) NÃO são suportados e falham
    # com 'gguf_tensor_to_f16 failed'.
    try:
        weights, metadata = mx.load(args.file, return_metadata=True)
    except RuntimeError as e:
        raise SystemExit(
            f"Falha ao ler o GGUF: {e}\n"
            "Provável causa: quantização não suportada pelo MLX (ex.: Q4_K_M, "
            "Q6_K). Baixe uma variante F16/F32/Q8_0/Q4_0 do mesmo modelo, ou "
            "reconverta com o llama.cpp para um desses tipos."
        )

    os.makedirs(args.out, exist_ok=True)
    n = len(weights)
    total = sum(v.size for v in weights.values())
    print(f"Tensores: {n} | parâmetros (dequantizados): {total:,}")

    # Salva pesos em safetensors (formato que o MLX/HF entende).
    mx.save_safetensors(os.path.join(args.out, "model.safetensors"), weights)

    # Salva os metadados do GGUF (arquitetura, hiperparâmetros, etc.)
    meta_serializable = {}
    for k, v in metadata.items():
        try:
            meta_serializable[k] = v.tolist() if hasattr(v, "tolist") else v
        except Exception:  # noqa: BLE001
            meta_serializable[k] = str(v)
    with open(os.path.join(args.out, "gguf_metadata.json"), "w") as f:
        json.dump(meta_serializable, f, indent=2, ensure_ascii=False, default=str)

    # Amostra dos nomes de tensores pra você conferir a arquitetura.
    print("\nPrimeiros tensores:")
    for k in list(weights.keys())[:8]:
        print(f"  {k:40s} {tuple(weights[k].shape)} {weights[k].dtype}")

    print(f"\nOK. Pesos + metadados em {args.out}/")
    print("Nota: para rodar como modelo completo você precisa do config.json "
          "da arquitetura correspondente (ex.: copie do repo HF original).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_hf = sub.add_parser("hf", help="HF safetensors -> MLX")
    p_hf.add_argument("--repo", required=True, help="repo HF ou pasta local")
    p_hf.add_argument("--out", required=True)
    p_hf.add_argument("--quantize", action="store_true")
    p_hf.add_argument("--q-bits", type=int, default=4)
    p_hf.add_argument("--q-group-size", type=int, default=64)
    p_hf.set_defaults(func=convert_hf)

    p_gg = sub.add_parser("gguf", help="GGUF -> MLX (safetensors + metadados)")
    p_gg.add_argument("--file", required=True)
    p_gg.add_argument("--out", required=True)
    p_gg.set_defaults(func=convert_gguf)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
