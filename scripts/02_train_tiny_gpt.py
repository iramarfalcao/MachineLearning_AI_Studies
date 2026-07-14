"""
02 — Treinar um mini-GPT DO ZERO em MLX puro (estilo nanoGPT, char-level).

Objetivo didático: entender o loop de treino completo — tokenização, modelo
Transformer, forward, loss, backprop e geração — rodando na GPU do Apple Silicon.
Também registra as métricas no MLflow (rode `mlflow ui` para visualizar).

Uso:
    python scripts/02_train_tiny_gpt.py --steps 500
    python scripts/02_train_tiny_gpt.py --steps 2000 --n_layer 6 --n_embd 256
"""
import argparse
import math
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------- Modelo ----------------------------------------
class Head(nn.Module):
    """Uma cabeça de self-attention causal."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        # máscara causal (triangular inferior)
        self._mask = mx.tril(mx.ones((block_size, block_size)))

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)
        mask = self._mask[:T, :T]
        wei = mx.where(mask == 0, -mx.inf, wei)
        wei = mx.softmax(wei, axis=-1)
        v = self.value(x)
        return wei @ v


class MultiHead(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = MultiHead(n_head, n_embd, block_size)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(mx.arange(T))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = mx.softmax(logits, axis=-1)
            next_id = mx.random.categorical(mx.log(probs))
            idx = mx.concatenate([idx, next_id[:, None]], axis=1)
        return idx


# --------------------------- Treino ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    # Dados: char-level a partir do corpus gerado no script 01.
    corpus_path = os.path.join(ROOT, "data", "processed", "corpus.txt")
    if not os.path.exists(corpus_path):
        raise SystemExit("Rode antes: python scripts/01_prepare_data.py")
    text = open(corpus_path, encoding="utf-8").read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = mx.array([stoi[c] for c in text], dtype=mx.int32)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"vocab={vocab_size} | chars totais={len(data)} | "
          f"params~{args.n_layer}x{args.n_embd}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = mx.random.randint(0, len(d) - args.block_size, (args.batch,))
        x = mx.stack([d[i:i + args.block_size] for i in ix.tolist()])
        y = mx.stack([d[i + 1:i + args.block_size + 1] for i in ix.tolist()])
        return x, y

    model = MiniGPT(vocab_size, args.n_embd, args.n_head, args.n_layer,
                    args.block_size)
    mx.eval(model.parameters())

    def loss_fn(model, x, y):
        logits = model(x)
        B, T, C = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, C), y.reshape(B * T)
        ).mean()

    optimizer = optim.AdamW(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # MLflow tracking (opcional, mas ligado por padrão)
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
        mlflow.set_experiment("mini-gpt-from-scratch")
        run_ctx = mlflow.start_run()
        mlflow.log_params(vars(args))
        use_mlflow = True
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] MLflow desativado ({e}). Treino segue sem tracking.")
        use_mlflow = False

    for step in range(args.steps):
        x, y = get_batch("train")
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 50 == 0 or step == args.steps - 1:
            vx, vy = get_batch("val")
            vloss = loss_fn(model, vx, vy)
            mx.eval(vloss)
            print(f"step {step:4d} | train {float(loss):.3f} | "
                  f"val {float(vloss):.3f} | ppl {math.exp(float(vloss)):.1f}")
            if use_mlflow:
                mlflow.log_metric("train_loss", float(loss), step=step)
                mlflow.log_metric("val_loss", float(vloss), step=step)

    # Amostra de geração
    print("\n--- Amostra gerada pelo modelo ---")
    context = mx.array([[stoi.get("M", 0)]])
    out = model.generate(context, max_new_tokens=200)[0].tolist()
    print("".join(itos[i] for i in out))

    # Salva pesos
    out_dir = os.path.join(ROOT, "models", "finetuned", "mini-gpt")
    os.makedirs(out_dir, exist_ok=True)
    model.save_weights(os.path.join(out_dir, "weights.safetensors"))
    print(f"\nPesos salvos em {out_dir}/weights.safetensors")
    if use_mlflow:
        mlflow.end_run()
        print("Métricas registradas no MLflow (rode: mlflow ui).")


if __name__ == "__main__":
    main()
