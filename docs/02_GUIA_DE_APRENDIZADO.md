# 🎓 Guia de Aprendizado — do zero ao MLOps

Este é o seu **currículo prático**. Cada módulo tem: o conceito, o que rodar, o
que observar e um checklist. Marque `[x]` conforme avança. Vá na ordem — cada
módulo prepara o próximo.

> Antes de tudo: `cd /Users/iramarfalcao/ZCodeProject && source .venv/bin/activate`

Legenda de tempo: ⏱️ rápido (<5 min) · ⏳ médio (5–20 min) · 🕰️ longo (>20 min)

---

## Módulo 0 — Ambiente e hardware ⏱️
**Objetivo:** entender por que o Mac consegue treinar modelos.

- [ ] Rodar `make check`
- [ ] Confirmar que aparece `device: Device(gpu, 0)` e o teste de GPU passa

**Conceitos:**
- **Memória unificada**: no Apple Silicon, CPU e GPU compartilham a mesma RAM.
  Por isso um M4 com 24 GB roda modelos que exigiriam uma GPU dedicada cara.
- **Metal**: a API gráfica/compute da Apple que o MLX usa por baixo.

**Pergunta para você responder:** por que não precisamos de CUDA aqui?

---

## Módulo 1 — Dados e tokenização ⏱️
**Objetivo:** entender como texto vira número (a entrada de todo LLM).

- [ ] Rodar `make data`
- [ ] Abrir `data/processed/corpus.txt` e `data/processed/train.jsonl`
- [ ] Ler o script `scripts/01_prepare_data.py`

**Conceitos:**
- **Token**: unidade que o modelo lê (caractere, subpalavra ou palavra).
- **Vocabulário**: o conjunto de tokens possíveis; cada um vira um número (id).
- **Formato de chat** (`{"messages": [...]}`): o padrão para fine-tuning
  instrucional (par usuário → assistente).
- **Split train/valid**: separar dados de treino e de validação para medir se o
  modelo generaliza (e não só decora).

**Exercício:** edite os `PARES` em `01_prepare_data.py`, adicione 2 perguntas
suas e rode `make data` de novo.

---

## Módulo 2 — Treinar um LLM DO ZERO ⏳
**Objetivo:** ver, por dentro, como um Transformer aprende.

- [ ] Rodar `python scripts/02_train_tiny_gpt.py --steps 500`
- [ ] Observar a **loss caindo** e a **perplexity (ppl)** diminuindo
- [ ] Ler o script inteiro — é um GPT completo em ~150 linhas

**Conceitos (a espinha dorsal de todo LLM):**
- **Embedding**: transforma id de token num vetor aprendível.
- **Self-attention**: cada token "olha" para os anteriores e decide o que é
  relevante. É o coração do Transformer.
- **Máscara causal**: impede o modelo de "ver o futuro" (só olha para trás).
- **Bloco Transformer**: attention + rede feed-forward + normalização.
- **Loss (cross-entropy)**: mede o erro entre a previsão e o token correto.
- **Backpropagation / gradiente**: como o modelo ajusta os pesos para errar menos.
- **Perplexity**: `exp(loss)` — quão "surpreso" o modelo fica; menor é melhor.

**Exercícios:**
1. Treine mais: `--steps 3000 --n_layer 6 --n_embd 256`. A geração melhora?
2. Compare a amostra gerada com 100 vs 3000 steps.

> 💡 **Versão visual:** para acompanhar o treino com **gráficos de loss e
> perplexity ao vivo** em matplotlib, abra `notebooks/02_treino_visual.ipynb`
> com `make lab`. É o mesmo modelo, de forma interativa.

> ⚠️ Este modelo é de brinquedo (char-level). O texto gerado será imperfeito —
> o objetivo é **entender o mecanismo**, não a qualidade.

---

## Módulo 3 — Inferência com um LLM real ⏱️
**Objetivo:** usar um modelo pré-treinado de verdade.

- [ ] Rodar `make generate PROMPT="Explique o que é overfitting."`
- [ ] Observar tokens/seg e o pico de memória no fim da saída

**Conceitos:**
- **Modelo pré-treinado**: já aprendeu linguagem com bilhões de tokens; você só
  usa (ou ajusta).
- **Quantização** (`-4bit` no nome do modelo): pesos comprimidos para 4 bits →
  ocupa menos memória, roda mais rápido, com pequena perda de qualidade.
- **Temperatura**: controla criatividade (baixa = mais determinístico).
- **Chat template**: a formatação especial que modelos instruct esperam.

**Exercício:** rode o mesmo prompt com `--temp 0.2` e `--temp 1.0` (edite o
comando direto: `python scripts/04_generate.py --prompt "..." --temp 1.0`).
Compare as respostas.

---

## Módulo 4 — Fine-tuning com LoRA ⏳
**Objetivo:** adaptar um LLM real aos SEUS dados.

- [ ] Rodar `make finetune` (ou `ITERS=200 ./scripts/03_finetune_lora.sh`)
- [ ] Observar a **val loss caindo** ao longo das iterações
- [ ] Testar o modelo ajustado:
  ```bash
  python scripts/04_generate.py --prompt "O que é LoRA?" \
      --adapter models/finetuned/lora-adapters
  ```

**Conceitos:**
- **Fine-tuning**: continuar o treino de um modelo pronto em dados específicos.
- **LoRA (Low-Rank Adaptation)**: em vez de ajustar todos os pesos, treina
  pequenas matrizes extras (só ~0.3% dos parâmetros!). Rápido e leve.
- **Adaptador**: o arquivo pequeno com o "delta" que o LoRA aprendeu.
- **Fuse (fundir)**: juntar o adaptador ao modelo base para gerar um modelo
  standalone.
- **Trainable parameters**: no seu teste, só 1.4M de 494M pesos foram treinados.

> 💡 **Versão visual:** `notebooks/04_finetune_visual.ipynb` mostra a resposta do
> modelo **antes e depois** do fine-tuning e plota a curva de loss do LoRA ao vivo.
> Abra com `make lab`.

**Exercícios:**
1. Troque seus dados em `data/processed/` e re-treine.
2. Funda o adaptador:
   ```bash
   python -m mlx_lm fuse --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
       --adapter-path models/finetuned/lora-adapters \
       --save-path models/finetuned/merged-model
   ```

---

## Módulo 5 — MLOps: rastrear experimentos ⏳
**Objetivo:** parar de "chutar" e começar a **medir e comparar**.

- [ ] Rodar `python mlops/mlflow_demo.py`
- [ ] Rodar `make mlflow` e abrir http://127.0.0.1:5000
- [ ] Comparar os 3 runs com learning rates diferentes
- [ ] Reparar que o Módulo 2 também gravou runs (experimento `mini-gpt-from-scratch`)

**Conceitos de MLOps:**
- **Experiment tracking**: registrar params, métricas e artefatos de cada treino.
- **Reprodutibilidade**: mesmo código + mesmos dados + mesma config = mesmo
  resultado. O Makefile e os configs ajudam nisso.
- **Versionamento**: dados, modelos e código separados e versionados
  (veja o `.gitignore` — artefatos grandes ficam fora do git).
- **Métrica vs. hiperparâmetro**: hiperparâmetro é o que você escolhe (lr,
  batch); métrica é o que você mede (loss).

**Exercício:** rode 2 fine-tunings com `learning-rate` diferente (edite o
`03_finetune_lora.sh` ou o `configs/lora_config.yaml`) e compare no MLflow.

> 💡 **Versão visual:** `notebooks/05_mlflow_visual.ipynb` lê os experimentos via
> API e monta os gráficos de comparação (curvas de loss e barras de loss final)
> dentro do notebook — o que um engenheiro de MLOps faz para relatórios.
>
> ℹ️ **Backend:** o MLflow 3.x guarda tudo num SQLite (`mlflow.db`). A UI abre com
> `make mlflow` (que já passa `--backend-store-uri sqlite:///mlflow.db`).

---

## Módulo 6 — Conversão de formatos (GGUF ↔ MLX) ⏳
**Objetivo:** mover modelos entre ecossistemas.

- [ ] HF → MLX com quantização:
  ```bash
  python scripts/05_convert_models.py hf \
      --repo Qwen/Qwen2.5-0.5B-Instruct \
      --out models/mlx/qwen-4bit --quantize
  ```
- [ ] (Opcional) Baixar um `.gguf` do Hugging Face, pôr em `models/gguf/` e:
  ```bash
  python scripts/05_convert_models.py gguf \
      --file models/gguf/SEU_MODELO.gguf --out models/mlx/do-gguf
  ```

**Conceitos:**
- **GGUF**: formato do `llama.cpp`, popular para rodar modelos quantizados em
  CPU/GPU em qualquer plataforma.
- **MLX/safetensors**: formato nativo do nosso ecossistema.
- **Por que converter?** Cada runtime (llama.cpp, MLX, Ollama) prefere um
  formato. Saber converter te dá liberdade.
- **Limite do GGUF→MLX**: os pesos vêm, mas a config de arquitetura precisa ser
  remontada (detalhes no README, seção "GGUF ↔ MLX"). Além disso, o MLX só
  dequantiza **F16, F32, Q8_0, Q4_0, Q4_1** — k-quants (Q4_K_M, Q6_K) falham.

> 💡 **Versão visual:** `notebooks/06_conversao_visual.ipynb` mede a **economia
> de espaço** da quantização num gráfico e inspeciona os tensores de um GGUF.
> Abra com `make lab`.

---

## Módulo 7 — Projeto final (junte tudo) 🕰️
**Objetivo:** um ciclo completo de ML de ponta a ponta.

- [ ] Escolher um domínio (ex.: um assistente sobre um tema que você domina)
- [ ] Criar ~100–300 pares de Q&A em `data/processed/train.jsonl`
- [ ] Fazer fine-tuning LoRA e registrar no MLflow
- [ ] Comparar pelo menos 2 configurações de hiperparâmetros
- [ ] Fundir o melhor adaptador num modelo standalone
- [ ] (Bônus) Exportar para GGUF com `llama.cpp` e rodar no Ollama

**Você terá praticado:** dados → treino → avaliação → tracking → empacotamento.
Esse é o ciclo de vida de MLOps na prática.

> 💡 **Versão guiada:** `notebooks/07_projeto_final.ipynb` executa esse ciclo
> inteiro num só fluxo (baseline → LoRA com tracking → avaliação antes/depois →
> fuse → relatório). Use como esqueleto e troque pelos seus dados. Abra com `make lab`.

---

## 📖 Glossário rápido

| Termo | Significado curto |
|-------|-------------------|
| **Época (epoch)** | Uma passada completa por todo o dataset |
| **Batch** | Grupo de exemplos processados de uma vez |
| **Learning rate** | Tamanho do passo ao ajustar os pesos |
| **Overfitting** | Modelo decora o treino e vai mal em dados novos |
| **Checkpoint** | Snapshot dos pesos salvo durante o treino |
| **Inferência** | Usar o modelo treinado para gerar/predizer |
| **Gradiente** | Direção para ajustar os pesos e reduzir a loss |
| **Parâmetros** | Os pesos aprendíveis do modelo |
| **Quantização** | Comprimir pesos para menos bits |
| **LoRA** | Fine-tuning leve treinando matrizes de baixo posto |

---

## ✅ Onde estou? (marque seu progresso)

- [ ] Módulo 0 — Ambiente
- [ ] Módulo 1 — Dados
- [ ] Módulo 2 — Treino do zero
- [ ] Módulo 3 — Inferência
- [ ] Módulo 4 — Fine-tuning LoRA
- [ ] Módulo 5 — MLOps
- [ ] Módulo 6 — Conversão
- [ ] Módulo 7 — Projeto final

Use o `docs/03_DIARIO.md` para anotar o que aprendeu em cada sessão.
