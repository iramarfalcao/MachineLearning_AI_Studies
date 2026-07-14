# 📚 Documentação do LLM Sandbox

Comece por aqui. Leia na ordem:

1. **[01_FERRAMENTAS.md](01_FERRAMENTAS.md)** — o que foi instalado, o que cada
   ferramenta faz e como ativar o ambiente.
2. **[02_GUIA_DE_APRENDIZADO.md](02_GUIA_DE_APRENDIZADO.md)** — currículo prático
   em 8 módulos, do ambiente ao projeto final. Com checklists e exercícios.
3. **[03_DIARIO.md](03_DIARIO.md)** — onde você anota cada sessão de estudo.

Notebooks interativos (abra com `make lab`):
- **[../notebooks/02_treino_visual.ipynb](../notebooks/02_treino_visual.ipynb)** — Módulo 2: treino do mini-GPT com gráficos de loss ao vivo.
- **[../notebooks/04_finetune_visual.ipynb](../notebooks/04_finetune_visual.ipynb)** — Módulo 4: LoRA com comparação antes/depois.
- **[../notebooks/05_mlflow_visual.ipynb](../notebooks/05_mlflow_visual.ipynb)** — Módulo 5: comparar experimentos do MLflow.
- **[../notebooks/06_conversao_visual.ipynb](../notebooks/06_conversao_visual.ipynb)** — Módulo 6: HF→MLX (quantização) e GGUF→MLX.
- **[../notebooks/07_projeto_final.ipynb](../notebooks/07_projeto_final.ipynb)** — Módulo 7: ciclo completo dados→treino→tracking→empacotamento.

Documentação técnica do projeto (comandos, estrutura de pastas, GGUF↔MLX):
veja o **[../README.md](../README.md)** na raiz.

---

## Fluxo recomendado para hoje

```bash
cd /Users/iramarfalcao/ZCodeProject
source .venv/bin/activate

make check        # Módulo 0
make data         # Módulo 1
make train-tiny   # Módulo 2
```

Depois abra `docs/02_GUIA_DE_APRENDIZADO.md` e siga módulo a módulo.
