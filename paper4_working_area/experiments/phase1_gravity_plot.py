import json
import matplotlib.pyplot as plt

# Load the results from the previous experiment
with open('../results/phase1_semantic_gravity.json', 'r') as f:
    data = json.load(f)
results = data['results']

# Filter out entries where attention_mass is None (e.g., for 'poodle')
valid_results = [r for r in results if r["attention_mass"] is not None]

radii = [r["norm"] for r in valid_results]
attn = [r["attention_mass"] for r in valid_results]

plt.figure(figsize=(8,6))
plt.scatter(radii, attn, c='darkviolet', s=100)
plt.xlabel("Poincaré Norm (abstraction level)")
plt.ylabel("Transformer Attention Mass")
plt.title("The Basic-Level Gravity Resonance")
plt.axvline(0.68, color='gold', linestyle='--', linewidth=3, label="Basic-Level Sweet Spot (~0.68)")
plt.legend()
plt.grid(alpha=0.3)

# Add word labels
for r in results:
    if r["attention_mass"] is not None:
        plt.text(r["norm"]+0.005, r["attention_mass"], r["word"], fontsize=10)

plt.savefig("../figures/semantic_gravity_resonance.png", dpi=300, bbox_inches='tight')
plt.show()