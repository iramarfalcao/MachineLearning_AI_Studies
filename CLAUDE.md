# CLAUDE.md

Orientações para agentes trabalhando neste repositório.

## O que é este projeto

Sandbox de aprendizado de **mini-LLMs, fine-tuning, conversão de modelos e MLOps**,
rodando localmente em **Apple Silicon (M4, 24 GB) via MLX/Metal**. É didático:
cada script/notebook corresponde a um módulo do currículo em `docs/`.

## Setup e ambiente

- Todo o Python vive num venv em `.venv/`. **Sempre ative antes de rodar qualquer coisa:**
  ```bash
  cd /Users/iramarfalcao/Github/MachineLearning_AI_Studies && source .venv/bin/activate
  ```
- Se um `import`/`mlx_lm` falhar por "módulo não encontrado", quase sempre o venv não está ativo.
- Dependências em `requirements.txt`. Reinstalar: `make setup`.
- O `Makefile` é o orquestrador — rode `make help` para ver os atalhos (`check`, `data`,
  `train-tiny`, `finetune`, `generate`, `mlflow`, `lab`, `clean`).

## Estrutura

- `scripts/00..05` — pipeline numerado (env check → dados → treino do zero → LoRA →
  geração → conversão). Rodam via `python scripts/NN_*.py` ou `bash`.
- `notebooks/02,04,05,06,07` — versões visuais interativas dos módulos (matplotlib).
- `mlops/mlflow_demo.py` — demo de experiment tracking.
- `configs/lora_config.yaml` — config de referência do `mlx_lm lora`.
- `docs/` — documentação de aprendizado: `01_FERRAMENTAS`, `02_GUIA_DE_APRENDIZADO`
  (currículo de 8 módulos), `03_DIARIO`. É a fonte de verdade da didática.
- `data/`, `models/{base,finetuned,mlx,gguf}` — dados e artefatos (gitignored).

## Convenções

- **Idioma:** todo código, comentário e documentação em **português**. Mantenha assim.
- Scripts numerados refletem a ordem do currículo — ao adicionar um passo, siga a numeração.
- Cada notebook é auto-contido: detecta `ROOT` via `os.getcwd()` assumindo cwd em
  `notebooks/` (comportamento do JupyterLab), e roda com poucos `iters`/`steps` por padrão.
- Artefatos grandes (modelos, `.gguf`, `mlflow.db`) são **gitignored** e regeneráveis;
  não os comite. `make clean` remove todos.

## Gotchas importantes (aprendidos na prática)

- **MLflow 3.x desativou o backend de arquivo.** Use SQLite: o tracking URI é sempre
  `sqlite:///mlflow.db`. A UI abre com `make mlflow` (passa `--backend-store-uri`).
  Não volte para `file://mlruns` — lança exceção.
- **GGUF → MLX só suporta F16, F32, Q8_0, Q4_0, Q4_1.** k-quants (Q4_K_M, Q6_K) falham
  com `gguf_tensor_to_f16 failed`. O MLX **não exporta** GGUF — o caminho inverso é via
  `llama.cpp` (`convert_hf_to_gguf.py`).
- **Modelos padrão:** fine-tuning usa `mlx-community/Qwen2.5-0.5B-Instruct-4bit`;
  conversão usa `HuggingFaceTB/SmolLM2-135M-Instruct` (bem pequeno, rápido de baixar).
- Versão do MLX se consulta com `importlib.metadata.version("mlx")` — o módulo `mlx`
  não tem `__version__`.

## Validação

Ao mexer em scripts/notebooks, valide executando de fato (não só leia). Para notebooks:
`jupyter nbconvert --to notebook --execute <nb> --output /dev/null` (de dentro de
`notebooks/`), idealmente com `iters`/`steps` reduzidos numa cópia temporária.
