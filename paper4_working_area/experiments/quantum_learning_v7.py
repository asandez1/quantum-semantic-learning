#!/usr/bin/env python3
"""
QUANTUM LEARNING V7: The Definitive Model
=====================================================

COMBINING THE WINNERS:
1. ARCHITECTURE: V4 (Dense Re-uploading + Ancilla + CNOTs)
   - Proven to learn with a neutral baseline.
   - CNOTs provide the strong signal needed to avoid mumbling.

2. SCALING: [0, π] (The V5 Fix)
   - Removes the aliasing bug where 0 ≈ 2π.
   - Maximizes contrast between dissimilar vectors.

3. HYPERPARAMETERS:
   - Higher Learning Rate (0.20) to force decision boundary separation.

GOAL:
- The rigor of V4 (Neutral Baseline).
- The performance of V3 (High Correlation).
"""

import numpy as np
import json
import os
import subprocess
from datetime import datetime
from scipy import stats

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "experiment_name": "quantum_learning_v7_definitive",
    "n_qubits_per_vector": 3,
    "n_features": 6,            # Dense (Ry+Rz)
    "n_ancilla": 1,
    "shots": 4096,
    "spsa_iterations": 300,
    "spsa_lr": 0.20,            # Boosted LR to break mumbling
    "reupload_layers": 2
}

# ============================================================================
# UTILITIES
# ============================================================================

def get_git_info():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except:
        return "nogit"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

# ============================================================================
# V7 ARCHITECTURE (V4 Logic + V5 Scaling)
# ============================================================================

def build_definitive_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    n_q = CONFIG['n_qubits_per_vector'] # 3
    n_total = n_q * 2 + 1               # 7
    ancilla = n_total - 1

    qc = QuantumCircuit(n_total)

    # === LAYER 1: ENCODING ===
    # Dense Encode v1
    for i in range(n_q):
        qc.ry(float(v1[i]), i)
        qc.rz(float(v1[i+n_q]), i)

    # Dense Encode v2
    for i in range(n_q):
        qc.ry(float(v2[i]), n_q + i)
        qc.rz(float(v2[i+n_q]), n_q + i)

    qc.barrier()

    # === PROCESSING 1 (Strong Entanglement) ===
    p_idx = 0
    for i in range(n_total):
        qc.rx(float(theta[p_idx]), i)
        p_idx += 1

    # V4 STYLE: CNOT RING (Strong Digital Signal)
    for i in range(n_total - 1):
        qc.cx(i, i+1)
    qc.cx(n_total-1, 0)

    # Entangle with Ancilla
    qc.cx(0, ancilla)
    qc.cx(n_q, ancilla)

    qc.barrier()

    # === LAYER 2: RE-UPLOADING ===
    # Re-Encode v1 (Scaled)
    for i in range(n_q):
        qc.ry(float(v1[i]) * 0.5, i)

    # Re-Encode v2 (Scaled)
    for i in range(n_q):
        qc.ry(float(v2[i]) * 0.5, n_q + i)

    qc.barrier()

    # === PROCESSING 2 ===
    for i in range(n_total):
        qc.ry(float(theta[p_idx]), i)
        p_idx += 1

    # Final CNOTs to Ancilla (Crucial for decision)
    for i in range(0, n_total-1, 2):
        qc.cx(i, ancilla)

    # Hadamard sandwich on Ancilla to measure phase interference
    qc.h(ancilla)
    qc.measure_all()

    return qc

# ============================================================================
# TRAINING LOGIC
# ============================================================================

def get_ancilla_prob(counts):
    """P(Ancilla=1)"""
    total = sum(counts.values())
    hits = 0
    for k, v in counts.items():
        if k[0] == '1': hits += v
    return hits / total

def train_v7(sampler, train_data, n_params):
    theta = np.random.uniform(-0.01, 0.01, n_params)
    best_theta = theta.copy()
    best_loss = float('inf')
    losses = []

    print(f"  Training V7 ({CONFIG['spsa_iterations']} iters, {n_params} params, LR={CONFIG['spsa_lr']})...")

    for k in range(CONFIG['spsa_iterations']):
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        ck = 0.1 / (k+1)**0.101
        ak = CONFIG['spsa_lr'] / (k+1)**0.602

        tp = theta + ck*delta
        tm = theta - ck*delta

        circuits = []
        for v1, v2, _ in train_data:
            circuits.append(build_definitive_circuit(v1, v2, tp))
            circuits.append(build_definitive_circuit(v1, v2, tm))

        res = sampler.run(circuits, shots=CONFIG['shots']).result()

        loss_p, loss_m = 0, 0
        for i, (_, _, label) in enumerate(train_data):
            pp = get_ancilla_prob(res[i*2].data.meas.get_counts())
            pm = get_ancilla_prob(res[i*2+1].data.meas.get_counts())

            eps = 1e-7
            pp = np.clip(pp, eps, 1-eps)
            pm = np.clip(pm, eps, 1-eps)

            loss_p += -(label * np.log(pp) + (1-label)*np.log(1-pp))
            loss_m += -(label * np.log(pm) + (1-label)*np.log(1-pm))

        loss_p /= len(train_data)
        loss_m /= len(train_data)

        grad = (loss_p - loss_m) / (2*ck) * delta
        theta = theta - ak*grad

        current_loss = (loss_p + loss_m)/2
        losses.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta.copy()

        if (k+1) % 50 == 0:
            print(f"    Iter {k+1}: Loss = {current_loss:.4f} (best={best_loss:.4f})")

    return best_theta, losses

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("QUANTUM LEARNING V7: THE DEFINITIVE MODEL")
    print("="*70)
    print("Strategy: V4 Architecture + V5 Scaling + Boosted LR")
    print(f"Qubits: {CONFIG['n_qubits_per_vector']*2 + 1} (6 data + 1 ancilla)")
    print(f"Features: {CONFIG['n_features']} (dense encoding)")
    print(f"Scaling: [0, π] (fixing aliasing)")
    print(f"Learning Rate: {CONFIG['spsa_lr']} (boosted)")
    print("="*70)

    np.random.seed(42)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Dataset
    train_sim = [
        ('dog', 'puppy'),
        ('cat', 'kitten'),
        ('king', 'queen'),
        ('car', 'truck'),
        ('happy', 'joy'),
        ('run', 'walk'),
        ('ocean', 'sea'),
        ('book', 'novel'),
    ]

    train_diff = [
        ('dog', 'car'),
        ('cat', 'stone'),
        ('king', 'banana'),
        ('run', 'sleep'),
        ('happy', 'sad'),
        ('book', 'fire'),
        ('water', 'computer'),
        ('music', 'chair'),
    ]

    test_pairs = [
        ('wolf', 'dog'),       # Similar
        ('eagle', 'hawk'),     # Similar
        ('apple', 'orange'),   # Similar
        ('wolf', 'table'),     # Dissimilar
        ('eagle', 'phone'),    # Dissimilar
        ('apple', 'rocket'),   # Dissimilar
        ('mouse', 'rat'),      # Similar
        ('tree', 'computer'),  # Dissimilar
    ]

    all_words = list(set([w for p in train_sim + train_diff + test_pairs for w in p]))
    print(f"\nTotal words: {len(all_words)}")
    print(f"Training similar: {len(train_sim)}")
    print(f"Training dissimilar: {len(train_diff)}")
    print(f"Test pairs: {len(test_pairs)}")

    embeddings = model.encode(all_words, show_progress_bar=False)

    # 1. PCA (Dense)
    print(f"\nApplying PCA: 384D → {CONFIG['n_features']}D...")
    pca = PCA(n_components=CONFIG['n_features'])
    vecs_pca = pca.fit_transform(embeddings)
    variance = np.sum(pca.explained_variance_ratio_)
    print(f"Variance Explained: {variance:.1%}")

    # 2. SCALING: [0, π] (The Fix)
    print("Scaling to [0, π] (Correcting Aliasing)...")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    vectors = scaler.fit_transform(vecs_pca)

    def get_vec(word): return vectors[all_words.index(word)]
    def get_raw(word): return embeddings[all_words.index(word)]

    train_data = []
    for p in train_sim:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 0.0))
    for p in train_diff:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 1.0))
    np.random.shuffle(train_data)

    print(f"Training labels: {[d[2] for d in train_data]}")

    backend = AerSimulator()
    sampler = SamplerV2(mode=backend)
    n_params = 14  # Same params count as V4

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n" + "="*70)
    print("[BASELINE] RANDOM PARAMETERS")
    print("="*70)

    theta_rand = np.random.uniform(-np.pi, np.pi, n_params)
    circuits = [build_definitive_circuit(get_vec(p[0]), get_vec(p[1]), theta_rand) for p in test_pairs]
    res_rand = sampler.run(circuits, shots=CONFIG['shots']).result()

    preds_rand = []
    targets = []
    for i, (c1, c2) in enumerate(test_pairs):
        pred = 1.0 - get_ancilla_prob(res_rand[i].data.meas.get_counts())
        preds_rand.append(pred)

        v1, v2 = get_raw(c1), get_raw(c2)
        truth = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        targets.append(truth)

    corr_rand, _ = stats.pearsonr(preds_rand, targets)
    print(f"Random params correlation: {corr_rand:.4f}")
    print(f"Random preds range: [{min(preds_rand):.3f}, {max(preds_rand):.3f}]")

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("[TRAINING] V7 DEFINITIVE CIRCUIT")
    print("="*70)

    theta_opt, losses = train_v7(sampler, train_data, n_params)

    # =========================================================================
    # EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("[EVALUATION]")
    print("="*70)

    circuits = [build_definitive_circuit(get_vec(p[0]), get_vec(p[1]), theta_opt) for p in test_pairs]
    res = sampler.run(circuits, shots=CONFIG['shots']).result()

    preds = []
    print(f"\n{'Pair':<20} {'Truth':<10} {'Pred':<10} {'Match'}")
    print("-"*55)

    for i, (c1, c2) in enumerate(test_pairs):
        pred = 1.0 - get_ancilla_prob(res[i].data.meas.get_counts())
        preds.append(pred)

        truth = targets[i]
        match = "✅" if (truth > 0.4) == (pred > 0.5) else "❌"
        print(f"{c1+'-'+c2:<20} {truth:.4f}     {pred:.4f}     {match}")

    corr, p_value = stats.pearsonr(preds, targets)
    span = max(preds) - min(preds)
    training_effect = corr - corr_rand

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("QUANTUM LEARNING V7 - SUMMARY")
    print("="*70)

    learning_threshold = 0.15

    print(f"\n{'Metric':<35} {'Value':<12} {'Verdict'}")
    print("-"*60)
    print(f"{'Variance Explained':<35} {variance:.1%}")
    print(f"{'Random params correlation':<35} {corr_rand:+.4f}       (baseline)")
    print(f"{'Trained params correlation':<35} {corr:+.4f}       {'✅' if corr > 0.5 else '❌'}")
    print(f"{'Training improvement':<35} {training_effect:+.4f}       {'✅ LEARNING' if training_effect > learning_threshold else '❌'}")
    print(f"{'Final loss':<35} {losses[-1]:.4f}")
    print(f"{'Prediction span':<35} {span:.4f}        {'✅ FIXED' if span > 0.3 else '❌ MUMBLING'}")

    learning_demonstrated = (
        corr > corr_rand + learning_threshold and
        corr > 0.3
    )

    print("\n" + "="*70)
    if learning_demonstrated:
        print("✅ QUANTUM LEARNING DEMONSTRATED!")
        print(f"   - Random baseline: {corr_rand:.4f}")
        print(f"   - After training:  {corr:.4f}")
        print(f"   - Improvement:     {training_effect:+.4f}")
        if span > 0.3:
            print(f"   - Mumbling FIXED: span = {span:.3f}")
    else:
        print("❌ NO LEARNING DEMONSTRATED")
        if training_effect <= learning_threshold:
            print(f"   - Training improvement ({training_effect:+.4f}) below threshold")
        if corr <= 0.3:
            print(f"   - Correlation ({corr:.4f}) too low")
    print("="*70)

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "config": CONFIG,
        "git_commit": get_git_info(),
        "timestamp": datetime.now().isoformat(),
        "results": {
            "corr_random": float(corr_rand),
            "corr_trained": float(corr),
            "training_effect": float(training_effect),
            "pred_span": float(span),
            "final_loss": float(losses[-1]),
            "best_loss": float(min(losses)),
            "variance_explained": float(variance),
            "preds_trained": preds,
            "preds_random": preds_rand,
            "targets": targets,
        },
        "learning_demonstrated": bool(learning_demonstrated),
    }

    filename = f"{CONFIG['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
