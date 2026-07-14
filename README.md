# 🧪 LLM Sandbox — Mini-LLMs, Fine-tuning e MLOps no Apple Silicon

Ambiente completo para **criar mini-LLMs, treinar modelos pequenos do zero,
fazer fine-tuning com LoRA, converter modelos (GGUF ↔ MLX) e aprender MLOps** —
tudo rodando localmente na GPU do seu Mac (Apple M4 / Metal) via **MLX**.

> Testado em: Apple M4, 24 GB RAM, macOS, Python 3.12, MLX + mlx-lm.

> 📚 **Novo por aqui? Comece pela documentação de aprendizado:** [docs/README.md](docs/README.md)
> — resumo das ferramentas, currículo em 8 módulos e diário de bordo.

---

## 🚀 Início rápido

```bash
# 1. Instalar tudo (venv + dependências)   -> já feito no setup inicial
make setup

# 2. Conferir que MLX enxerga a GPU
make check

# 3. Gerar os datasets de exemplo
make data

# 4a. Treinar um mini-GPT DO ZERO (didático, ~1 min)
make train-tiny

# 4b. Fine-tuning LoRA de um LLM pequeno real
make finetune

# 5. Gerar texto
make generate PROMPT="O que é quantização?"

# 6. Ver experimentos no MLflow
make mlflow    # abre http://127.0.0.1:5000
```

Rode `make help` para ver todos os atalhos.

---

## 📂 Estrutura

```
ZCodeProject/
├── Makefile                 # atalhos (make help)
├── requirements.txt         # dependências
├── configs/
│   └── lora_config.yaml     # config de referência p/ LoRA
├── data/
│   ├── raw/                 # seus dados brutos
│   └── processed/           # corpus.txt + train/valid.jsonl (gerados)
├── models/
│   ├── base/                # modelos base
│   ├── finetuned/           # saídas de treino (mini-gpt, adaptadores LoRA)
│   ├── gguf/                # coloque .gguf aqui para converter
│   └── mlx/                 # modelos convertidos p/ MLX
├── scripts/
│   ├── 00_env_check.py      # diagnóstico GPU/pacotes
│   ├── 01_prepare_data.py   # gera datasets
│   ├── 02_train_tiny_gpt.py # mini-GPT do ZERO em MLX puro (nanoGPT-style)
│   ├── 03_finetune_lora.sh  # fine-tuning LoRA com mlx-lm
│   ├── 04_generate.py       # inferência (base ou com LoRA)
│   └── 05_convert_models.py # HF->MLX e GGUF->MLX
├── mlops/
│   └── mlflow_demo.py       # demo de experiment tracking
└── notebooks/               # JupyterLab (make lab)
```

---

## 🧠 Os 3 fluxos de aprendizado

### 1. Treinar um modelo DO ZERO (entender por dentro)
`scripts/02_train_tiny_gpt.py` implementa um Transformer completo em MLX puro
(attention, blocos, embeddings, loop de treino, geração). É o melhor lugar para
entender **como um LLM realmente funciona**. Char-level, treina em segundos.

```bash
python scripts/02_train_tiny_gpt.py --steps 2000 --n_layer 6 --n_embd 256
```

### 2. Fine-tuning de um LLM real com LoRA (o que se usa na prática)
`scripts/03_finetune_lora.sh` usa `mlx-lm` para ajustar um modelo pequeno
(Qwen2.5-0.5B por padrão) nos seus dados de chat, treinando só os adaptadores
LoRA. Depois você pode **fundir** (fuse) os adaptadores num modelo standalone:

```bash
# treinar
./scripts/03_finetune_lora.sh

# testar com o adaptador
python scripts/04_generate.py --prompt "O que é LoRA?" \
    --adapter models/finetuned/lora-adapters

# fundir adaptador no modelo
python -m mlx_lm fuse \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --adapter-path models/finetuned/lora-adapters \
    --save-path models/finetuned/merged-model
```

Formato dos dados (`data/processed/train.jsonl`), um exemplo por linha:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 3. MLOps — versionar e rastrear experimentos
Todo treino registra params/métricas no **MLflow** (backend SQLite em `mlflow.db`).

```bash
python mlops/mlflow_demo.py   # gera runs de exemplo
make mlflow                   # UI em http://127.0.0.1:5000
```

> O MLflow 3.x exige um backend em banco; o sandbox usa SQLite (`mlflow.db`).
> Para explorar os experimentos **dentro de um notebook** (gráficos de comparação),
> use `notebooks/05_mlflow_visual.ipynb`.

Conceitos de MLOps praticados aqui: **experiment tracking**, versionamento de
dados/modelos (pastas separadas + `.gitignore`), pipelines reproduzíveis
(Makefile), quantização e empacotamento de modelos.

---

## 🔄 GGUF ↔ MLX

### GGUF → MLX
```bash
# coloque o .gguf em models/gguf/ e:
python scripts/05_convert_models.py gguf \
    --file models/gguf/modelo.gguf \
    --out models/mlx/do-gguf
```
Lê o GGUF nativamente com o MLX, dequantiza os pesos, salva em `safetensors` e
exporta os metadados. (Para rodar como modelo completo você precisa do
`config.json` da arquitetura correspondente — copie do repo HF original.)

### HF → MLX (caminho recomendado, com quantização)
```bash
python scripts/05_convert_models.py hf \
    --repo Qwen/Qwen2.5-0.5B-Instruct \
    --out models/mlx/qwen2.5-0.5b-4bit --quantize
```

### MLX/HF → GGUF (caminho inverso, via llama.cpp)
O MLX não exporta GGUF diretamente. Use o `llama.cpp`:
```bash
brew install llama.cpp
# a partir de uma pasta de modelo HF/MLX-fundido:
python convert_hf_to_gguf.py models/finetuned/merged-model \
    --outfile models/gguf/meu-modelo.gguf --outtype q4_0
```

---

## 🛠️ Modelos pequenos recomendados (cabem em 24 GB)

| Modelo | Tamanho | Uso |
|--------|---------|-----|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~0.5B | fine-tuning rápido (default) |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | ~1.5B | melhor qualidade |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | ~1B   | alternativa Llama |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~3B   | topo do que cabe confortável |

Troque com a variável de ambiente `MODEL`:
```bash
MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit" ITERS=300 ./scripts/03_finetune_lora.sh
```

---

## 📚 Próximos passos sugeridos
1. Rode `make train-tiny` e leia `scripts/02_train_tiny_gpt.py` linha a linha.
2. Substitua `data/processed/train.jsonl` pelos **seus** dados e refaça o LoRA.
3. Abra o MLflow e compare runs com hiperparâmetros diferentes.
4. Baixe um `.gguf` do Hugging Face e converta para MLX.
5. Explore quantização: `05_convert_models.py hf ... --quantize --q-bits 4`.
