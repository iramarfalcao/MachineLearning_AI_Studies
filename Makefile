# ===================================================================
# LLM Sandbox — atalhos. Rode `make help` para ver tudo.
# ===================================================================
PY := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := help

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Cria venv e instala dependências
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Ambiente pronto. Rode 'make check'."

check: ## Diagnóstico do ambiente (MLX/GPU/pacotes)
	$(PY) scripts/00_env_check.py

data: ## Gera os datasets (corpus + chat jsonl)
	$(PY) scripts/01_prepare_data.py

train-tiny: ## Treina o mini-GPT do zero em MLX (didático)
	$(PY) scripts/02_train_tiny_gpt.py --steps 500

finetune: ## Fine-tuning LoRA de um LLM pequeno real
	bash scripts/03_finetune_lora.sh

generate: ## Gera texto (use PROMPT="..." e opcional ADAPTER=...)
	$(PY) scripts/04_generate.py --prompt "$(or $(PROMPT),O que é MLOps?)" \
		$(if $(ADAPTER),--adapter $(ADAPTER),)

mlflow: ## Abre a UI do MLflow em http://127.0.0.1:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

mlops-demo: ## Roda a demo de tracking com MLflow
	$(PY) mlops/mlflow_demo.py

lab: ## Abre o JupyterLab
	.venv/bin/jupyter lab

clean: ## Remove artefatos de treino (mantém venv e código)
	rm -rf mlruns mlartifacts mlflow.db models/finetuned/* models/mlx/* data/processed/*
	@echo "Limpo."

.PHONY: help setup check data train-tiny finetune generate mlflow mlops-demo lab clean
