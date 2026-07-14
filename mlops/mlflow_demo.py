"""
MLOps — Demo de experiment tracking com MLflow.

Mostra o padrão de registrar params, métricas e artefatos de um "treino"
(aqui simulado) para você entender o fluxo antes de aplicá-lo nos scripts reais.
Os scripts 02 e 03 já registram no mesmo backend (pasta ./mlruns).

Uso:
    python mlops/mlflow_demo.py
    mlflow ui            # abre http://127.0.0.1:5000 para ver os experimentos
"""
import math
import os

import mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
mlflow.set_experiment("demo-mlops")

# Simula 3 execuções com hiperparâmetros diferentes.
for lr in [1e-3, 3e-3, 1e-2]:
    with mlflow.start_run(run_name=f"lr={lr}"):
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("optimizer", "adamw")
        mlflow.log_param("model", "mini-gpt")

        # curva de loss fake: menor lr converge mais suave
        loss = 4.0
        for step in range(100):
            loss = max(0.2, loss - lr * (5 + 2 * math.sin(step / 5)))
            mlflow.log_metric("loss", loss, step=step)

        # registra um artefato (ex.: um resumo do run)
        art = os.path.join(ROOT, "mlartifacts", "summary.txt")
        os.makedirs(os.path.dirname(art), exist_ok=True)
        with open(art, "w") as f:
            f.write(f"final_loss={loss:.3f} lr={lr}\n")
        mlflow.log_artifact(art)
        print(f"run lr={lr}: final_loss={loss:.3f}")

print("\nPronto. Rode 'mlflow ui' e abra http://127.0.0.1:5000")
