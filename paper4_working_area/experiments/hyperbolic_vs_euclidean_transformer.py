# hyperbolic_vs_euclidean_transformer.py
# THE REAL EXPERIMENT — Fully Hyperbolic vs Euclidean from Scratch
# November 24, 2025 — You are running history.

import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
from geoopt import PoincareBall
from datetime import datetime
import os
import numpy as np

# === HYPERBOLIC BUILDING BLOCKS ===
class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.ball = PoincareBall(c=c)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.ball.logmap0(x)
        x = self.linear(x)
        x = self.ball.expmap0(x)
        return x


class FullyHyperbolicAttention(nn.Module):
    def __init__(self, dim, heads=8, c=1.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.ball = PoincareBall(c=c)
        self.to_qkv = HypLinear(dim, dim*3, c=c, bias=False)
        self.to_out = HypLinear(dim, dim, c=c)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [tensor_chunk.view(b, t, self.heads, self.dim_head) for tensor_chunk in qkv]

        q = self.ball.logmap0(q)
        k = self.ball.logmap0(k)
        v = self.ball.logmap0(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        out = self.ball.expmap0(out)
        return self.to_out(out)


class HyperbolicModel(nn.Module):
    def __init__(self, dim=128, depth=4, heads=8, c=1.0):
        super().__init__()
        self.ball = PoincareBall(c=c)
        self.emb = nn.Embedding(1000, dim)  # your vocab
        self.layers = nn.ModuleList([
            FullyHyperbolicAttention(dim, heads, c=c) for _ in range(depth)
        ])
        self.proj = HypLinear(dim, 20, c=c)  # to your 20D space

    def forward(self, idx):
        x = self.emb(idx)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.ball.expmap0(x * 0.1)  # start small
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x.mean(dim=1))  # pool
        return x


class EuclideanModel(nn.Module):
    def __init__(self, dim=128, depth=4, heads=8):
        super().__init__()
        self.emb = nn.Embedding(1000, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.proj = nn.Linear(dim, 20)

    def forward(self, idx):
        x = self.emb(idx)
        x = x.unsqueeze(1) # Add sequence dimension
        for _ in range(4):
            h, _ = self.attn(x, x, x)
            x = x + h
            x = x + self.ff(x)
        return self.proj(x.mean(dim=1))


# === TRAINING LOOP ===
def train_model(model, pairs, epochs=500, lr=0.01, name="model"):
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr) if hasattr(model, 'ball') else optim.Adam(model.parameters(), lr=lr)
    ball = model.ball if hasattr(model, 'ball') else None

    losses = []
    print(f"Training {name} from scratch...")

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        for c1, c2, target_dist in pairs:
            v1 = model(torch.tensor([c1]))
            v2 = model(torch.tensor([c2]))

            if ball:
                dist = ball.dist(v1, v2)
            else:
                dist = torch.cdist(v1, v2)

            loss = (dist - target_dist).pow(2).mean()
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        if epoch % 100 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss.item():.6f}")

    print(f"{name} trained! Final loss: {losses[-1]:.6f}")
    return losses


# === RUN EXPERIMENT ===
if __name__ == "__main__":
    print("Starting Hyperbolic vs Euclidean Transformer Showdown")

    # Dummy data — replace with your real 75 concepts
    concepts = list(range(75))
    pairs = [(i, j, np.random.rand()) for i in range(75) for j in range(i+1, 75)][:500]

    # Train both
    hyp_model = HyperbolicModel(dim=128, depth=4)
    eucl_model = EuclideanModel(dim=128, depth=4)

    hyp_losses = train_model(hyp_model, pairs, epochs=400, name="Fully Hyperbolic")
    eucl_losses = train_model(eucl_model, pairs, epochs=400, name="Euclidean")

    # Final summary
    final_results = {
        "name": "Hyperbolic vs Euclidean — From Scratch",
        "backend": "Classical",
        "correlation": "In progress",
        "quantum_hits": 0,
        "conclusion": f"Fully hyperbolic model: final loss {hyp_losses[-1]:.6f}\n"
                     f"Euclidean model: final loss {eucl_losses[-1]:.6f}\n"
                     f"{'HYPERBOLIC WINS' if hyp_losses[-1] < eucl_losses[-1] else 'EUCLIDEAN STILL AHEAD'}"
    }

    def generate_final_summary(results, file):
        os.makedirs("../results", exist_ok=True)
        with open(file, "w") as f:
            f.write(f"# Hyperbolic vs Euclidean — Final Result\n")
            f.write(f"- Date: {datetime.now().isoformat()}\n")
            f.write(f"{results['conclusion']}\n")
        print(f"Summary saved to {file}")

    generate_final_summary(final_results, "../results/hyperbolic_vs_euclidean_final.txt")

    print("EXPERIMENT COMPLETE — You just ran the future.")