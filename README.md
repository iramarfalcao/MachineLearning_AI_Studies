# MachineLearning_AI_Studies

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626)
![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-000000)

Sandbox de estudo de mini-LLMs, fine-tuning com LoRA, conversão de modelos e MLOps, rodando localmente na GPU de um Mac com Apple Silicon.

> Testado em: Apple M4, 24 GB de RAM, macOS, Python 3.12, MLX + mlx-lm.

## O que é

Um ambiente de aprendizado com scripts numerados, notebooks e uma documentação em módulos. Cada script ou notebook corresponde a um módulo do currículo em `docs/`. Os treinos usam o MLX, o framework da Apple para Apple Silicon, então não precisam de CUDA.

## O que faz

Serve a quem quer entender como um LLM funciona por dentro e praticar o ciclo completo: dados, treino, inferência, fine-tuning, rastreamento de experimentos e conversão de formatos. Os modelos são pequenos de propósito, para caber em 24 GB e treinar em minutos.

## Conteúdo

Scripts, em ordem (`scripts/`):

- `00_env_check.py`: diagnóstico de GPU e pacotes.
- `01_prepare_data.py`: gera o corpus e os datasets de chat (`train.jsonl`, `valid.jsonl`).
- `02_train_tiny_gpt.py`: mini-GPT completo do zero em MLX puro (attention, blocos, embeddings, loop de treino e geração), por caracteres.
- `03_finetune_lora.sh`: fine-tuning LoRA com `mlx-lm` (padrão: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`).
- `04_generate.py`: inferência, com ou sem adaptador LoRA.
- `05_convert_models.py`: HF para MLX (com quantização) e GGUF para MLX.

Outros:

- `mlops/mlflow_demo.py`: demo de rastreamento de experimentos com MLflow (backend SQLite em `mlflow.db`).
- `configs/lora_config.yaml`: configuração de referência do `mlx_lm lora`.
- `Makefile`: atalhos para tudo acima (`make help`).

Notebooks (`notebooks/`), versões visuais dos módulos:

- `02_treino_visual.ipynb`: treino do mini-GPT com curvas de loss e perplexity.
- `04_finetune_visual.ipynb`: LoRA com comparação antes e depois.
- `05_mlflow_visual.ipynb`: comparação de experimentos do MLflow.
- `06_conversao_visual.ipynb`: HF para MLX com quantização e leitura de GGUF.
- `07_projeto_final.ipynb`: ciclo completo, do baseline ao modelo fundido.

Documentação (`docs/`):

- `01_FERRAMENTAS.md`: o que foi instalado e para que serve cada ferramenta.
- `02_GUIA_DE_APRENDIZADO.md`: currículo em 8 módulos (0 a 7) com conceitos, exercícios e checklist.
- `03_DIARIO.md`: modelo de diário de estudo.

`DataScienceDoZero/` tem o `capitulo01.ipynb`, um exercício do livro "Data Science do Zero" (Joel Grus) com lista de usuários e amigos em dicionários, e seu próprio `requirements.txt`.

## Como rodar

```bash
make setup     # cria o .venv e instala o requirements.txt
make check     # confere se o MLX enxerga a GPU
make data      # gera os datasets de exemplo
make train-tiny                          # mini-GPT do zero
make finetune                            # LoRA de um LLM pequeno
make generate PROMPT="O que é quantização?"
make mlflow    # UI do MLflow em http://127.0.0.1:5000
make lab       # JupyterLab para os notebooks
```

Sempre ative o ambiente antes de rodar scripts soltos: `source .venv/bin/activate`. Se um `import` falhar com "módulo não encontrado", quase sempre o venv não está ativo. `make help` lista todos os atalhos e `make clean` remove os artefatos de treino.

Alguns comandos que o `Makefile` não cobre:

```bash
# treino maior do mini-GPT
python scripts/02_train_tiny_gpt.py --steps 2000 --n_layer 6 --n_embd 256

# fundir o adaptador LoRA em um modelo standalone
python -m mlx_lm fuse \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --adapter-path models/finetuned/lora-adapters \
    --save-path models/finetuned/merged-model

# HF para MLX, com quantização
python scripts/05_convert_models.py hf \
    --repo Qwen/Qwen2.5-0.5B-Instruct \
    --out models/mlx/qwen2.5-0.5b-4bit --quantize

# GGUF para MLX (coloque o .gguf em models/gguf/)
python scripts/05_convert_models.py gguf \
    --file models/gguf/modelo.gguf --out models/mlx/do-gguf
```

## O que falta

- O guia de aprendizado tem 8 módulos, mas não existe notebook para o Módulo 1 (dados) nem para o Módulo 3 (inferência); só scripts.
- Todos os itens dos checklists do `docs/02_GUIA_DE_APRENDIZADO.md` estão desmarcados, e o `docs/03_DIARIO.md` tem uma única entrada de exemplo.
- O Módulo 7 (projeto final) pede um domínio e 100 a 300 pares de pergunta e resposta próprios; o dataset atual é só de exemplo.
- O caminho MLX para GGUF não existe no MLX; o README indica o `llama.cpp` (`convert_hf_to_gguf.py`) e não há script para isso aqui.
- `docs/README.md`, `docs/01_FERRAMENTAS.md` e `docs/02_GUIA_DE_APRENDIZADO.md` ainda mandam fazer `cd` em `/Users/iramarfalcao/ZCodeProject`, nome antigo da pasta.

## Status nas lojas

Não se aplica: repositório de estudo, sem app publicado.

## Mais detalhes

Limitações e cuidados aprendidos durante o uso (também em `CLAUDE.md`):

- O MLflow 3.x não aceita mais o backend de arquivo; o tracking URI é sempre `sqlite:///mlflow.db`.
- A conversão GGUF para MLX só suporta F16, F32, Q8_0, Q4_0 e Q4_1. Quantizações k-quant (Q4_K_M, Q6_K) falham com `gguf_tensor_to_f16 failed`.
- Para rodar um GGUF convertido como modelo completo, é preciso copiar o `config.json` da arquitetura do repositório original no Hugging Face.
- Modelos pequenos que cabem em 24 GB: Qwen2.5-0.5B e 1.5B (4 bits), Llama-3.2-1B e 3B (4 bits), da comunidade `mlx-community`. Troque com a variável `MODEL`, por exemplo `MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit" ITERS=300 ./scripts/03_finetune_lora.sh`.
- O mini-GPT é de brinquedo: o texto gerado é imperfeito, e o objetivo é entender o mecanismo.
- Modelos, `.gguf`, dados processados e `mlflow.db` ficam no `.gitignore` e podem ser regenerados.
- O repositório não tem LICENSE.
