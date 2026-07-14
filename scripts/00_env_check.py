"""
00 — Diagnóstico do ambiente.
Confirma que MLX enxerga a GPU do Apple Silicon (Metal) e lista versões.
Rode: python scripts/00_env_check.py
"""
import platform
import sys


def check(name, fn):
    try:
        print(f"  ✓ {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {name}: ERRO -> {e}")


print("=" * 60)
print("DIAGNÓSTICO DO AMBIENTE — LLM Sandbox")
print("=" * 60)
print(f"  Python      : {sys.version.split()[0]}")
print(f"  Plataforma  : {platform.platform()}")
print(f"  Arquitetura : {platform.machine()}")
print("-" * 60)

from importlib.metadata import version  # noqa: E402

import mlx.core as mx  # noqa: E402

check("mlx (versão)", lambda: version("mlx"))
check("device padrão", lambda: mx.default_device())


def bench():
    # Multiplicação de matrizes na GPU só pra provar que Metal funciona.
    a = mx.random.normal((2048, 2048))
    b = mx.random.normal((2048, 2048))
    c = a @ b
    mx.eval(c)
    return f"matmul 2048x2048 ok, soma={float(c.sum()):.2f}"


check("teste GPU (Metal)", bench)

for pkg in ["mlx_lm", "transformers", "datasets", "mlflow", "numpy"]:
    check(f"import {pkg}", lambda p=pkg: __import__(p).__version__)

print("=" * 60)
print("Se tudo acima está ✓, o ambiente está pronto. 🚀")
print("=" * 60)
