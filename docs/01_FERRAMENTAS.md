# 🧰 Ferramentas Instaladas — o que é cada uma e para que serve

Tudo foi instalado dentro de um **ambiente virtual isolado** (`.venv/`), então nada
"suja" o Python do sistema. Para usar, sempre ative primeiro:

```bash
cd /Users/iramarfalcao/ZCodeProject
source .venv/bin/activate     # o prompt passa a mostrar (.venv)
```

Para sair do ambiente: `deactivate`.

---

## 🗺️ Visão geral em uma frase

| Camada | Ferramentas | Papel |
|--------|-------------|-------|
| **Motor de cálculo** | MLX | Roda a matemática dos modelos na GPU do seu Mac |
| **LLMs prontos** | mlx-lm | Baixa, treina (LoRA) e roda LLMs em MLX |
| **Ecossistema de modelos/dados** | transformers, datasets, huggingface-hub, tokenizers, safetensors, sentencepiece | Fornece modelos, datasets e utilidades padrão do mercado |
| **Dados numéricos** | numpy, pandas | Manipular arrays e tabelas |
| **MLOps** | mlflow | Registrar e comparar experimentos |
| **Exploração** | jupyterlab, matplotlib, ipywidgets | Notebooks e gráficos |
| **Utilidades** | tqdm, pyyaml | Barras de progresso e arquivos de config |

---

## 🔧 Detalhe de cada ferramenta

### Motor de cálculo

- **MLX** `0.31.2` — Framework de machine learning **da Apple**, feito para
  Apple Silicon (M1–M4). É o equivalente ao PyTorch/TensorFlow, mas usa a
  **memória unificada** e a **GPU via Metal** do seu Mac. É o que faz seu M4
  treinar modelos de verdade. Você o usa diretamente no script do mini-GPT
  (`02_train_tiny_gpt.py`).
  - `mlx.core` (mx) → arrays/tensores e operações (como o numpy, mas na GPU)
  - `mlx.nn` → camadas de rede neural (Linear, Embedding, LayerNorm…)
  - `mlx.optimizers` → otimizadores (AdamW, SGD…)

### LLMs prontos

- **mlx-lm** `0.31.3` — Biblioteca em cima do MLX especializada em **Large
  Language Models**. É o "canivete suíço" do sandbox. Faz:
  - `python -m mlx_lm generate` → gerar texto (inferência)
  - `python -m mlx_lm lora` → fine-tuning com LoRA
  - `python -m mlx_lm fuse` → fundir adaptadores LoRA no modelo
  - `mlx_lm.convert` → converter modelos Hugging Face → MLX (com quantização)
  - Baixa modelos prontos da comunidade `mlx-community` no Hugging Face.

### Ecossistema de modelos e dados (Hugging Face)

- **transformers** `5.12.1` — Biblioteca padrão da indústria com milhares de
  arquiteturas de modelos. Aqui é usada nos bastidores (tokenizers, configs).
- **datasets** `5.0.0` — Baixar e processar datasets públicos com uma linha.
- **huggingface-hub** `1.21.0` — Cliente para baixar/enviar modelos e dados do
  Hugging Face Hub (é o que puxa o Qwen2.5 quando você roda o fine-tuning).
- **tokenizers** `0.22.2` — Tokenização rápida (transforma texto ↔ números).
- **sentencepiece** `0.2.1` — Algoritmo de tokenização subword usado por muitos
  modelos (Llama, T5…).
- **safetensors** `0.8.0` — Formato seguro e rápido para salvar pesos de modelo
  (é o `.safetensors` que aparece nas pastas `models/`).

### Dados numéricos

- **numpy** `2.5.0` — Arrays n-dimensionais; base de quase tudo em ML.
- **pandas** `2.3.3` — Tabelas (DataFrames) para limpar e preparar datasets.

### MLOps

- **mlflow** `3.14.0` — Plataforma de **experiment tracking**. Registra
  hiperparâmetros, métricas (loss, etc.) e artefatos de cada treino numa pasta
  local (`mlruns/`). A UI (`mlflow ui`, em http://127.0.0.1:5000) deixa você
  **comparar execuções** lado a lado. É o coração da parte de MLOps.

### Exploração e visualização

- **jupyterlab** `4.6.1` — Ambiente de notebooks para experimentar de forma
  interativa (`make lab`).
- **matplotlib** `3.11.0` — Gráficos (curvas de loss, distribuições…).
- **ipywidgets** `8.1.8` — Widgets interativos dentro dos notebooks.

### Utilidades

- **tqdm** `4.68.3` — Barras de progresso nos loops.
- **pyyaml** `6.0.3` — Ler/escrever arquivos `.yaml` (ex.: `configs/lora_config.yaml`).

---

## 🍺 Ferramentas do sistema (fora do venv)

- **Homebrew** — gerenciador de pacotes do macOS (já instalado). Usado para
  instalar coisas de sistema, como o `llama.cpp`.
- **llama.cpp** *(opcional, para GGUF)* — Instale com `brew install llama.cpp`
  quando quiser o caminho **MLX/HF → GGUF** ou rodar modelos GGUF. Não é Python;
  é uma ferramenta separada em C++.

---

## 📌 Regra de ouro

> Sempre que abrir um terminal novo para trabalhar no sandbox:
> ```bash
> cd /Users/iramarfalcao/ZCodeProject && source .venv/bin/activate
> ```
> Se um comando `python`/`mlx_lm` "não encontra o módulo", quase sempre é porque
> o venv não está ativado.
