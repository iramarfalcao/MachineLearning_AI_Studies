"""
01 — Preparação de dados.

Faz duas coisas:
  (a) Gera um corpus de texto simples (data/processed/corpus.txt) para o
      treino "from scratch" do mini-GPT (script 02).
  (b) Converte/gera um dataset de chat no formato que o mlx_lm.lora espera
      (train.jsonl / valid.jsonl em data/processed/), para o fine-tuning (script 03).

Rode: python scripts/01_prepare_data.py
"""
import json
import os
import random

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# ---------------------------------------------------------------------------
# (a) Corpus para treino from-scratch (char/BPE-level). Um texto pequeno em PT.
# ---------------------------------------------------------------------------
CORPUS = (
    "MLOps é a prática de levar modelos de machine learning para produção de "
    "forma confiável e eficiente. Envolve versionamento de dados, versionamento "
    "de modelos, pipelines de treino reproduzíveis, monitoramento e automação. "
    "Um bom fluxo de MLOps registra cada experimento: hiperparâmetros, métricas "
    "e artefatos. Modelos pequenos podem ser treinados localmente em Apple Silicon "
    "usando MLX, que aproveita a GPU via Metal. Fine-tuning com LoRA ajusta apenas "
    "matrizes de baixo posto, economizando memória. Quantização reduz o tamanho do "
    "modelo convertendo pesos para menos bits. GGUF é um formato popular do "
    "llama.cpp; MLX tem seu próprio formato baseado em safetensors. "
) * 60  # repete pra ter volume suficiente pro treino de brinquedo

with open(os.path.join(PROC, "corpus.txt"), "w", encoding="utf-8") as f:
    f.write(CORPUS)
print(f"[a] corpus.txt escrito ({len(CORPUS)} chars) em {PROC}/corpus.txt")

# ---------------------------------------------------------------------------
# (b) Dataset de chat p/ fine-tuning LoRA (formato mlx_lm: {"messages": [...]})
# ---------------------------------------------------------------------------
PARES = [
    ("O que é MLOps?",
     "MLOps é o conjunto de práticas para levar modelos de machine learning "
     "para produção com confiabilidade: versionamento, pipelines reproduzíveis, "
     "monitoramento e automação."),
    ("O que é LoRA?",
     "LoRA (Low-Rank Adaptation) é uma técnica de fine-tuning que treina apenas "
     "pequenas matrizes de baixo posto adicionadas ao modelo, economizando "
     "memória e tempo em vez de ajustar todos os pesos."),
    ("O que é quantização?",
     "Quantização reduz a precisão dos pesos do modelo (ex.: de 16 para 4 bits), "
     "diminuindo o tamanho e o uso de memória com pequena perda de qualidade."),
    ("O que é GGUF?",
     "GGUF é um formato de arquivo do llama.cpp para armazenar modelos "
     "quantizados de forma eficiente para inferência em CPU e GPU."),
    ("O que é MLX?",
     "MLX é um framework de machine learning da Apple, otimizado para Apple "
     "Silicon, que usa memória unificada e a GPU via Metal."),
    ("Como registrar experimentos?",
     "Use uma ferramenta de tracking como o MLflow para registrar "
     "hiperparâmetros, métricas e artefatos de cada execução de treino."),
    ("O que é fine-tuning?",
     "Fine-tuning é continuar o treino de um modelo pré-treinado em dados "
     "específicos para adaptá-lo a uma tarefa ou domínio."),
    ("O que é um checkpoint?",
     "Um checkpoint é um snapshot dos pesos do modelo salvo durante o treino, "
     "permitindo retomar ou avaliar o modelo naquele ponto."),
]

# Data augmentation trivial pra ter linhas suficientes (mlx_lm gosta de >= algumas centenas).
def build_rows(n_repeat=40):
    rows = []
    for _ in range(n_repeat):
        for q, a in PARES:
            rows.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]})
    random.shuffle(rows)
    return rows


rows = build_rows()
split = int(len(rows) * 0.9)
train, valid = rows[:split], rows[split:]

for name, data in [("train", train), ("valid", valid)]:
    path = os.path.join(PROC, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[b] {name}.jsonl: {len(data)} exemplos -> {path}")

print("\nPronto. Dados em data/processed/ prontos para os scripts 02 e 03.")
