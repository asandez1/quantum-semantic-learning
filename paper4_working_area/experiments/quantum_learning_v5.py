#!/usr/bin/env python3
"""
QUANTUM LEARNING V5: Global Parity & Correct Scaling
=====================================================

FIXES FROM V4:
1. SCALING FIX: Changed inputs from [0, 2π] to [0, π].
   - V4 Bug: 0 and 2π are the same quantum state (|0⟩).
   - V5 Fix: 0 and π are orthogonal states (|0⟩ vs |1⟩).
   - Result: "Different" words will actually look different.

2. GLOBAL PARITY: Removed the "Bottleneck Ancilla".
   - V4: Measured 1 qubit (ignored 6 qubits of info).
   - V5: Measures ALL qubits.
   - Mechanism: Even Parity = Similar, Odd Parity = Dissimilar.

3. ZZ-PHASE ENTANGLEMENT:
   - Uses Rzz/CZ interaction which is more robust for semantic tasks
     than standard CNOT rings.

GOAL:
- Break the 0.5 "mumbling" plateau.
- Achieve Prediction Range > 0.6 (e.g., 0.2 to 0.8).
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
    "experiment_name": "quantum_learning_v5_parity",
    "n_qubits_per_vector": 3,
    "n_features": 6,            # Dense encoding (Ry + Rz)
    "n_total_qubits": 6,        # NO ANCILLA needed for Parity
    "shots": 4096,
    "spsa_iterations": 200,     # Faster convergence expected with Parity
    "spsa_lr": 0.1,
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
# V5 ARCHITECTURE: ZZ-Phase Parity Circuit
# ============================================================================

def build_parity_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    V5 CIRCUIT: Global Parity Measurement
    Input: v1 (6 dims), v2 (6 dims)
    Qubits: 6 (No Ancilla)
    """
    n = CONFIG['n_qubits_per_vector'] # 3
    n_total = n * 2                   # 6

    qc = QuantumCircuit(n_total)

    # === 1. DENSE ENCODING (Ry + Rz) ===
    # Left Register (v1)
    for i in range(n):
        qc.ry(float(v1[i]), i)
        qc.rz(float(v1[i+n]), i)

    # Right Register (v2)
    for i in range(n):
        qc.ry(float(v2[i]), n + i)
        qc.rz(float(v2[i+n]), n + i)

    qc.barrier()

    # === 2. INTERFERENCE LAYER (Trainable) ===
    # We want to mix the Left and Right registers

    # Trainable Rotations 1
    for i in range(n_total):
        qc.rx(float(theta[i]), i)

    # ZZ-Interaction (Simulating Rzz with CNOT-Rz-CNOT)
    # This creates strong phase entanglement between pairs
    for i in range(n):
        # Entangle v1[i] with v2[i] (corresponding features)
        qc.cx(i, n+i)
        qc.rz(float(theta[n_total + i]), n+i) # Parameterized interaction strength
        qc.cx(i, n+i)

    qc.barrier()

    # Trainable Rotations 2 (Mixing)
    for i in range(n_total):
        qc.ry(float(theta[n_total + n + i]), i)

    # === 3. BASIS ROTATION FOR PARITY ===
    # A Hadamard layer rotates Z-basis parity into X-basis interference
    qc.h(range(n_total))

    qc.measure_all()

    return qc

# ============================================================================
# PARITY CALCULATION
# ============================================================================

def compute_parity_prob(counts, n_qubits):
    """
    Computes the probability of EVEN parity.
    Even Parity (+1) -> Similar
    Odd Parity (-1) -> Dissimilar

    Returns: P(Even) which maps to Similarity [0, 1]
    """
    total = sum(counts.values())
    even_counts = 0

    for bitstring, count in counts.items():
        # Count number of 1s
        ones = bitstring.count('1')
        if ones % 2 == 0:
            even_counts += count

    return even_counts / total

# ============================================================================
# TRAINING
# ============================================================================

def train_v5(sampler, train_data, n_params):
    theta = np.random.uniform(-0.01, 0.01, n_params)
    best_theta = theta.copy()
    best_loss = float('inf')
    losses = []

    print(f"  Training V5 (Parity Mode, {CONFIG['spsa_iterations']} iters, {n_params} params)...")

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
            circuits.append(build_parity_circuit(v1, v2, tp))
            circuits.append(build_parity_circuit(v1, v2, tm))

        job = sampler.run(circuits, shots=CONFIG['shots'])
        res = job.result()

        # Loss Calc
        loss_p, loss_m = 0, 0
        for i, (_, _, target_sim) in enumerate(train_data):
            # Target is Similarity (0 to 1)
            # Prediction is P(Even Parity)

            pred_p = compute_parity_prob(res[i*2].data.meas.get_counts(), 6)
            pred_m = compute_parity_prob(res[i*2+1].data.meas.get_counts(), 6)

            # MSE Loss (often more stable for Parity than LogLoss)
            loss_p += (pred_p - target_sim) ** 2
            loss_m += (pred_m - target_sim) ** 2

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
    print("QUANTUM LEARNING V5: GLOBAL PARITY & SCALING FIX")
    print("="*70)
    print(f"Qubits: {CONFIG['n_total_qubits']} (no ancilla)")
    print(f"Features: {CONFIG['n_features']} (dense encoding)")
    print(f"Measurement: Global Parity (even=similar, odd=dissimilar)")
    print(f"Scaling: [0, π] (fixing aliasing bug)")
    print("="*70)

    np.random.seed(42)

    # 1. Data Prep
    print("\n[1] PREPARING DATA...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Dataset
    train_sim = [
        ('dog', 'puppy'),
        ('cat', 'kitten'),
        ('king', 'queen'),
        ('car', 'truck'),
        ('happy', 'joy'),
        ('run', 'walk'),
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
    ]

    # Harder Test Set
    test_pairs = [
        ('wolf', 'dog'),       # Similar
        ('eagle', 'hawk'),     # Similar
        ('ocean', 'sea'),      # Similar
        ('wolf', 'table'),     # Dissimilar
        ('eagle', 'phone'),    # Dissimilar
        ('ocean', 'computer'), # Dissimilar
        ('apple', 'orange'),   # Similar (Tricky)
        ('apple', 'rocket'),   # Dissimilar
    ]

    all_words = list(set([w for p in train_sim + train_diff + test_pairs for w in p]))
    print(f"Total words: {len(all_words)}")
    print(f"Training similar: {len(train_sim)}")
    print(f"Training dissimilar: {len(train_diff)}")
    print(f"Test pairs: {len(test_pairs)}")

    embeddings = model.encode(all_words, show_progress_bar=False)

    # V5 UPGRADE: PCA to 6 dims
    print(f"\nApplying PCA: 384D → {CONFIG['n_features']}D...")
    pca = PCA(n_components=CONFIG['n_features'])
    vecs_pca = pca.fit_transform(embeddings)
    variance = np.sum(pca.explained_variance_ratio_)
    print(f"Variance Explained: {variance:.1%}")

    # V5 CRITICAL FIX: Scale to [0, π] instead of [0, 2π]
    # This ensures |0> and |1> are distinct endpoints
    print(f"Scaling to [0, π] (Fixing aliasing bug from V4)...")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    vectors = scaler.fit_transform(vecs_pca)

    def get_vec(word): return vectors[all_words.index(word)]
    def get_ground_truth(w1, w2):
        v1, v2 = embeddings[all_words.index(w1)], embeddings[all_words.index(w2)]
        return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

    train_data = []
    for p in train_sim:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 1.0))  # Similar = 1.0 (Even Parity target)
    for p in train_diff:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 0.0))  # Dissimilar = 0.0 (Odd Parity target)
    np.random.shuffle(train_data)

    print(f"Training labels: {[d[2] for d in train_data]}")

    # 2. Setup
    print("\nInitializing AerSimulator...")
    backend = AerSimulator()
    sampler = SamplerV2(mode=backend)

    # Params: 6 (Rx) + 3 (Rz_interact) + 6 (Ry) = 15 params
    n_params = CONFIG['n_total_qubits'] * 2 + CONFIG['n_qubits_per_vector']

    # =========================================================================
    # BASELINE: Random Parameters
    # =========================================================================
    print("\n" + "="*70)
    print("[BASELINE] RANDOM PARAMETERS")
    print("="*70)

    theta_rand = np.random.uniform(-np.pi, np.pi, n_params)
    circuits = [build_parity_circuit(get_vec(p[0]), get_vec(p[1]), theta_rand) for p in test_pairs]
    res = sampler.run(circuits, shots=CONFIG['shots']).result()

    preds_rand = []
    targets = []
    for i, (c1, c2) in enumerate(test_pairs):
        pred = compute_parity_prob(res[i].data.meas.get_counts(), 6)
        truth = get_ground_truth(c1, c2)
        preds_rand.append(pred)
        targets.append(truth)

    corr_rand, _ = stats.pearsonr(preds_rand, targets)
    print(f"Random params correlation: {corr_rand:.4f}")
    print(f"Random preds range: [{min(preds_rand):.3f}, {max(preds_rand):.3f}]")

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("[TRAINING] V5 PARITY CIRCUIT")
    print("="*70)

    theta_opt, losses = train_v5(sampler, train_data, n_params)

    # =========================================================================
    # EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("[EVALUATION]")
    print("="*70)

    circuits = [build_parity_circuit(get_vec(p[0]), get_vec(p[1]), theta_opt) for p in test_pairs]
    res = sampler.run(circuits, shots=CONFIG['shots']).result()

    preds_trained = []
    print(f"\n{'Pair':<20} {'Truth':<10} {'Pred':<10} {'Verdict'}")
    print("-"*55)

    for i, (c1, c2) in enumerate(test_pairs):
        pred = compute_parity_prob(res[i].data.meas.get_counts(), 6)
        truth = targets[i]
        preds_trained.append(pred)

        # Verdict: Did we get the right "side"?
        is_sim_truth = truth > 0.4
        is_sim_pred = pred > 0.5
        match = "✅" if is_sim_truth == is_sim_pred else "❌"

        print(f"{c1+'-'+c2:<20} {truth:.4f}     {pred:.4f}     {match}")

    corr_train, p_value = stats.pearsonr(preds_trained, targets)
    training_effect = corr_train - corr_rand

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("QUANTUM LEARNING V5 - SUMMARY")
    print("="*70)

    pred_range = max(preds_trained) - min(preds_trained)
    learning_threshold = 0.15

    print(f"\n{'Metric':<35} {'Value':<12} {'Verdict'}")
    print("-"*60)
    print(f"{'Variance Explained':<35} {variance:.1%}")
    print(f"{'Random params correlation':<35} {corr_rand:+.4f}       (baseline)")
    print(f"{'Trained params correlation':<35} {corr_train:+.4f}       {'✅' if corr_train > 0.5 else '❌'}")
    print(f"{'Training improvement':<35} {training_effect:+.4f}       {'✅ LEARNING' if training_effect > learning_threshold else '❌'}")
    print(f"{'Final loss':<35} {losses[-1]:.4f}")
    print(f"{'Prediction range':<35} [{min(preds_trained):.3f}, {max(preds_trained):.3f}]  (span={pred_range:.3f})")
    print(f"{'Mumbling fixed?':<35} {'✅ YES' if pred_range > 0.3 else '❌ NO'} (target > 0.3)")

    learning_demonstrated = (
        corr_train > corr_rand + learning_threshold and
        corr_train > 0.3
    )

    print("\n" + "="*70)
    if learning_demonstrated:
        print("✅ QUANTUM LEARNING DEMONSTRATED!")
        print(f"   - Random baseline: {corr_rand:.4f}")
        print(f"   - After training:  {corr_train:.4f}")
        print(f"   - Improvement:     {training_effect:+.4f}")
        if pred_range > 0.3:
            print(f"   - Mumbling FIXED: range = {pred_range:.3f}")
    else:
        print("❌ NO LEARNING DEMONSTRATED")
        if training_effect <= learning_threshold:
            print(f"   - Training improvement ({training_effect:+.4f}) below threshold")
        if corr_train <= 0.3:
            print(f"   - Correlation ({corr_train:.4f}) too low")
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
            "corr_train": float(corr_train),
            "training_effect": float(training_effect),
            "preds_trained": preds_trained,
            "preds_random": preds_rand,
            "targets": targets,
            "loss": [float(l) for l in losses],
            "variance_explained": float(variance),
            "pred_range": float(pred_range),
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
