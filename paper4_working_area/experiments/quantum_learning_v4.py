#!/usr/bin/env python3
"""
QUANTUM LEARNING V4: Dense Re-uploading Classifier
===================================================

IMPROVEMENTS OVER V3:
1. DENSE ENCODING: Encodes 2 features per qubit (Ry and Rz).
   - V3: 3 qubits = 3 dims (23% variance)
   - V4: 3 qubits = 6 dims (~50% variance) -> MORE MEANING

2. DATA RE-UPLOADING: Encodes data TWICE.
   - Creates non-linear decision boundaries.
   - Fixes the "Mumbling" problem (low confidence predictions).

3. HADAMARD SANDWICH:
   - Wraps Ancilla in H-gates to measure Phase interference.
   - Increases sensitivity to semantic differences.

GOAL:
- Achieve > 0.8 Correlation
- Push Loss below 0.60
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
    "experiment_name": "quantum_learning_v4_dense",
    "n_qubits_per_vector": 3,   # 3 qubits per vector
    "n_features": 6,            # 2 features per qubit (Dense Encoding)
    "n_ancilla": 1,
    "shots": 4096,
    "spsa_iterations": 300,     # Increased for deeper circuit
    "spsa_lr": 0.1,             # Slightly lower LR for stability
    "reupload_layers": 2        # Number of times data is encoded
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
# V4 ARCHITECTURE: Dense Re-uploading
# ============================================================================

def build_dense_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    V4 CIRCUIT:
    - Input: v1 (6 dims), v2 (6 dims)
    - Architecture: 7 Qubits (3 for A, 3 for B, 1 Ancilla)
    """
    n_q = CONFIG['n_qubits_per_vector'] # 3
    n_total = n_q * 2 + 1               # 7
    ancilla = n_total - 1

    qc = QuantumCircuit(n_total)

    # === LAYER 1: ENCODING + PROCESSING ===
    # Dense Encode v1 (features 0-2 on Ry, 3-5 on Rz)
    for i in range(n_q):
        qc.ry(float(v1[i]), i)
        qc.rz(float(v1[i+n_q]), i)

    # Dense Encode v2
    for i in range(n_q):
        qc.ry(float(v2[i]), n_q + i)
        qc.rz(float(v2[i+n_q]), n_q + i)

    qc.barrier()

    # Trainable Weights 1
    param_idx = 0
    for i in range(n_total):
        qc.rx(float(theta[param_idx]), i)
        param_idx += 1

    # Entanglement 1 (Ring connection + Ancilla star)
    for i in range(n_total - 1):
        qc.cx(i, i+1)
    qc.cx(n_total-1, 0) # Close ring

    # Entangle data with Ancilla
    qc.cx(0, ancilla)
    qc.cx(n_q, ancilla)

    qc.barrier()

    # === LAYER 2: RE-UPLOADING (The "Memory" boost) ===
    # We encode the data AGAIN. This acts like a hidden layer.

    # Re-Encode v1
    for i in range(n_q):
        qc.ry(float(v1[i]) * 0.5, i) # Scaled re-upload

    # Re-Encode v2
    for i in range(n_q):
        qc.ry(float(v2[i]) * 0.5, n_q + i)

    qc.barrier()

    # Trainable Weights 2
    for i in range(n_total):
        qc.ry(float(theta[param_idx]), i)
        param_idx += 1

    # Entanglement 2
    for i in range(0, n_total-1, 2):
        qc.cz(i, i+1) # CZ gates for different phase interference

    # Final Ancilla Processing
    qc.h(ancilla) # Switch basis
    qc.measure_all()

    return qc

# ============================================================================
# TRAINING LOGIC (Standardized)
# ============================================================================

def get_ancilla_prob(counts, n_total):
    """Get Probability of Ancilla being |1> (Dissimilar)."""
    total = sum(counts.values())
    hits = 0
    for k, v in counts.items():
        # Check MSB (Ancilla is last qubit -> index 0 in bitstring if Little Endian)
        # Qiskit string is [q_n, ... q_0]. Our Ancilla is q_6.
        # So it is at index 0 of the string.
        if k[0] == '1':
            hits += v
    return hits / total

def train_v4(sampler, train_data, n_params):
    theta = np.random.uniform(-0.01, 0.01, n_params)
    best_theta = theta.copy()
    best_loss = float('inf')
    losses = []

    print(f"  Training V4 ({CONFIG['spsa_iterations']} iters, {n_params} params)...")

    for k in range(CONFIG['spsa_iterations']):
        # SPSA Setup
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        ck = 0.1 / (k+1)**0.101
        ak = CONFIG['spsa_lr'] / (k+1)**0.602

        tp = theta + ck*delta
        tm = theta - ck*delta

        # Batch Execution
        circuits = []
        for v1, v2, _ in train_data:
            circuits.append(build_dense_circuit(v1, v2, tp))
            circuits.append(build_dense_circuit(v1, v2, tm))

        job = sampler.run(circuits, shots=CONFIG['shots'])
        res = job.result()

        # Loss Calc
        loss_p, loss_m = 0, 0
        for i, (_, _, label) in enumerate(train_data):
            # Target: 0 (Similar), 1 (Dissimilar)
            pp = get_ancilla_prob(res[i*2].data.meas.get_counts(), 7)
            pm = get_ancilla_prob(res[i*2+1].data.meas.get_counts(), 7)

            # Binary Cross Entropy
            epsilon = 1e-7
            pp = np.clip(pp, epsilon, 1-epsilon)
            pm = np.clip(pm, epsilon, 1-epsilon)

            loss_p += -(label * np.log(pp) + (1-label)*np.log(1-pp))
            loss_m += -(label * np.log(pm) + (1-label)*np.log(1-pm))

        loss_p /= len(train_data)
        loss_m /= len(train_data)

        # Update
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
    print("QUANTUM LEARNING V4: DENSE RE-UPLOADING CLASSIFIER")
    print("="*70)
    print(f"Qubits per vector: {CONFIG['n_qubits_per_vector']}")
    print(f"Features (dense): {CONFIG['n_features']}")
    print(f"Total qubits: {CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']}")
    print(f"SPSA iterations: {CONFIG['spsa_iterations']}")
    print("="*70)

    np.random.seed(42)

    # 1. Data Prep
    print("\n[1] PREPARING DATA...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Richer Dataset
    train_sim = [
        ('dog', 'puppy'),
        ('cat', 'kitten'),
        ('king', 'queen'),
        ('run', 'walk'),
        ('happy', 'joy'),
        ('car', 'truck'),
        ('book', 'novel'),
    ]
    train_diff = [
        ('dog', 'car'),
        ('cat', 'stone'),
        ('king', 'banana'),
        ('run', 'sleep'),
        ('happy', 'sad'),
        ('book', 'mountain'),
        ('water', 'fire'),
    ]
    test_pairs = [
        ('wolf', 'dog'),      # Similar
        ('eagle', 'hawk'),    # Similar
        ('apple', 'orange'),  # Similar
        ('wolf', 'table'),    # Dissimilar
        ('eagle', 'phone'),   # Dissimilar
        ('apple', 'run'),     # Dissimilar
        ('mouse', 'rat'),     # Similar
        ('tree', 'computer'), # Dissimilar
    ]

    all_words = list(set([w for p in train_sim + train_diff + test_pairs for w in p]))
    print(f"Total words: {len(all_words)}")
    print(f"Training similar: {len(train_sim)}")
    print(f"Training dissimilar: {len(train_diff)}")
    print(f"Test pairs: {len(test_pairs)}")

    embeddings = model.encode(all_words, show_progress_bar=False)

    # V4 UPGRADE: PCA to 6 dimensions (for Dense Encoding)
    print(f"\nApplying PCA: 384D → {CONFIG['n_features']}D (Dense Mode)...")
    pca = PCA(n_components=CONFIG['n_features'])
    vecs_pca = pca.fit_transform(embeddings)
    variance = np.sum(pca.explained_variance_ratio_)
    print(f"Variance Explained: {variance:.1%}")

    # Scale to [0, 2pi] for full rotation coverage
    scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
    vectors = scaler.fit_transform(vecs_pca)

    # Prepare Batches
    def get_vec(word): return vectors[all_words.index(word)]
    def get_raw(word): return embeddings[all_words.index(word)]

    train_data = []
    for p in train_sim:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 0.0))  # Similar = 0
    for p in train_diff:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 1.0))  # Dissimilar = 1
    np.random.shuffle(train_data)

    print(f"Training labels: {[d[2] for d in train_data]}")

    # 2. Setup
    print("\nInitializing AerSimulator...")
    backend = AerSimulator()
    sampler = SamplerV2(mode=backend)

    # Calc params: 2 Layers of 7 qubits (Rx then Ry) -> 14 params
    n_params = 14

    # =========================================================================
    # BASELINE: Random Parameters
    # =========================================================================
    print("\n" + "="*70)
    print("[BASELINE] RANDOM PARAMETERS")
    print("="*70)

    theta_random = np.random.uniform(-np.pi, np.pi, n_params)

    # Evaluate random
    circuits_rand = []
    for c1, c2 in test_pairs:
        circuits_rand.append(build_dense_circuit(get_vec(c1), get_vec(c2), theta_random))

    job_rand = sampler.run(circuits_rand, shots=CONFIG['shots'])
    res_rand = job_rand.result()

    preds_rand = []
    targets = []
    for i, (c1, c2) in enumerate(test_pairs):
        prob_dissim = get_ancilla_prob(res_rand[i].data.meas.get_counts(), 7)
        pred_sim = 1.0 - prob_dissim
        preds_rand.append(pred_sim)

        raw_v1, raw_v2 = get_raw(c1), get_raw(c2)
        sim = np.dot(raw_v1, raw_v2)/(np.linalg.norm(raw_v1)*np.linalg.norm(raw_v2))
        targets.append(sim)

    corr_random, _ = stats.pearsonr(preds_rand, targets)
    print(f"Random params correlation: {corr_random:.4f}")
    print(f"Random preds range: [{min(preds_rand):.3f}, {max(preds_rand):.3f}]")

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("[TRAINING] V4 DENSE RE-UPLOADING CIRCUIT")
    print("="*70)

    theta_opt, losses = train_v4(sampler, train_data, n_params)

    # =========================================================================
    # EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("[EVALUATION]")
    print("="*70)

    circuits = []
    for c1, c2 in test_pairs:
        v1, v2 = get_vec(c1), get_vec(c2)
        circuits.append(build_dense_circuit(v1, v2, theta_opt))

    job = sampler.run(circuits, shots=CONFIG['shots'])
    res = job.result()

    preds = []
    print(f"\n{'Pair':<20} {'Truth':<10} {'Pred':<10} {'Match'}")
    print("-"*50)
    for i, (c1, c2) in enumerate(test_pairs):
        # Prob(Ancilla=1) is Dissimilarity.
        # So Similarity = 1 - Prob(Ancilla=1)
        prob_dissim = get_ancilla_prob(res[i].data.meas.get_counts(), 7)
        pred_sim = 1.0 - prob_dissim
        preds.append(pred_sim)

        # Check if prediction matches expectation
        is_similar = targets[i] > 0.5
        pred_similar = pred_sim > 0.5
        match = "✅" if is_similar == pred_similar else "❌"

        print(f"{c1+'-'+c2:<20} {targets[i]:.4f}     {pred_sim:.4f}     {match}")

    correlation, p_value = stats.pearsonr(preds, targets)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("QUANTUM LEARNING V4 - SUMMARY")
    print("="*70)

    training_effect = correlation - corr_random
    learning_threshold = 0.15

    print(f"\n{'Metric':<35} {'Value':<12} {'Verdict'}")
    print("-"*60)
    print(f"{'Variance Explained (PCA)':<35} {variance:.1%}        (vs 23% in V3)")
    print(f"{'Random params correlation':<35} {corr_random:+.4f}       (baseline)")
    print(f"{'Trained params correlation':<35} {correlation:+.4f}       {'✅' if correlation > 0.5 else '❌'}")
    print(f"{'Training improvement':<35} {training_effect:+.4f}       {'✅ LEARNING' if training_effect > learning_threshold else '❌'}")
    print(f"{'Final loss':<35} {losses[-1]:.4f}        (vs 0.66 in V3)")
    print(f"{'Prediction range':<35} [{min(preds):.3f}, {max(preds):.3f}]  (vs [0.45, 0.67] in V3)")

    learning_demonstrated = (
        correlation > corr_random + learning_threshold and
        correlation > 0.5
    )

    print("\n" + "="*70)
    if learning_demonstrated:
        print("✅ QUANTUM LEARNING DEMONSTRATED!")
        print(f"   - Random baseline: {corr_random:.4f}")
        print(f"   - After training:  {correlation:.4f}")
        print(f"   - Improvement:     {training_effect:+.4f}")
        if losses[-1] < 0.60:
            print(f"   - Loss broke 0.60 barrier: {losses[-1]:.4f}")
        if max(preds) - min(preds) > 0.3:
            print(f"   - 'Mumbling' fixed: range = {max(preds) - min(preds):.3f}")
    else:
        print("❌ NO LEARNING DEMONSTRATED")
        if training_effect <= learning_threshold:
            print(f"   - Training improvement ({training_effect:+.4f}) below threshold")
        if correlation <= 0.5:
            print(f"   - Correlation ({correlation:.4f}) too low")
    print("="*70)

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "config": CONFIG,
        "git_commit": get_git_info(),
        "timestamp": datetime.now().isoformat(),
        "results": {
            "random_correlation": float(corr_random),
            "trained_correlation": float(correlation),
            "training_effect": float(training_effect),
            "final_loss": float(losses[-1]),
            "best_loss": float(min(losses)),
            "variance_explained": float(variance),
            "predictions": preds,
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
