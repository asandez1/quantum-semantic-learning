#!/usr/bin/env python3
"""
QUANTUM LEARNING V6: The "Goldilocks" Fusion
=====================================================

THE STRATEGY:
1. ARCHITECTURE (From V4): Dense Re-uploading with Ancilla.
   - Proven to learn (+0.46 improvement).
   - Robust against noise (unlike V5 Parity).

2. SCALING (From V5): Inputs scaled to [0, π].
   - Fixes V4's "Aliasing Bug" where 0 ≈ 2π.
   - Max distance = Orthogonal states (|0⟩ vs |1⟩).

3. PHASE KICKBACK:
   - Uses CRz (Controlled-Rotation Z) instead of simple CNOT.
   - Allows "Analog" influence rather than "Digital" flipping.

GOAL:
- Break the 0.60 Correlation barrier.
- Fix "Mumbling" (Spread predictions > 0.4).
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
    "experiment_name": "quantum_learning_v6_goldilocks",
    "n_qubits_per_vector": 3,
    "n_features": 6,            # Dense (Ry+Rz)
    "n_ancilla": 1,             # Back to Ancilla (V4 style)
    "shots": 4096,
    "spsa_iterations": 300,     # Deep training
    "spsa_lr": 0.1,
    "layers": 2                 # Re-uploading depth
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
# V6 ARCHITECTURE: Phase-Kickback Classifier
# ============================================================================

def build_goldilocks_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    n = CONFIG['n_qubits_per_vector'] # 3
    n_total = n * 2 + 1               # 7 (Ancilla is index 6)
    ancilla = n_total - 1

    qc = QuantumCircuit(n_total)

    # Helper for Dense Encoding
    def encode_layer(vec, offset):
        for i in range(n):
            qc.ry(float(vec[i]), offset + i)      # Feature 1 -> Y rotation
            qc.rz(float(vec[i+n]), offset + i)    # Feature 2 -> Z phase

    # === LAYER 1: INITIAL UPLOAD ===
    encode_layer(v1, 0)   # Left Reg
    encode_layer(v2, n)   # Right Reg

    qc.barrier()

    # === PROCESSING BLOCK 1 ===
    # Trainable Rotations
    p_idx = 0
    for i in range(n_total):
        qc.rx(float(theta[p_idx]), i)
        p_idx += 1

    # Phase Kickback Entanglement (CRz)
    # Instead of flipping the Ancilla (CNOT), we rotate it based on Data
    # This creates a "Weighted Sum" effect
    for i in range(n * 2):
        # Control: Data Qubit [i], Target: Ancilla
        # Rotation angle is parameterized
        qc.crz(float(theta[p_idx]), i, ancilla)
        p_idx += 1

    qc.barrier()

    # === LAYER 2: RE-UPLOAD (Memory Effect) ===
    # Scale by 0.5 to allow fine-tuning
    v1_scaled = v1 * 0.5
    v2_scaled = v2 * 0.5

    encode_layer(v1_scaled, 0)
    encode_layer(v2_scaled, n)

    # === PROCESSING BLOCK 2 ===
    for i in range(n_total):
        qc.ry(float(theta[p_idx]), i)
        p_idx += 1

    # Final "Decision" Entanglement
    # Connect V1 and V2 to Ancilla using CNOTs for hard decision
    for i in range(n):
        qc.cx(i, ancilla)      # V1 -> Ancilla
        qc.cx(n + i, ancilla)  # V2 -> Ancilla

    qc.measure_all()
    return qc

# ============================================================================
# TRAINING LOGIC
# ============================================================================

def get_ancilla_prob(counts):
    """
    Get P(Ancilla=1).
    Ancilla is Qubit 6 (MSB in Qiskit Little Endian string).
    """
    total = sum(counts.values())
    hits = 0
    for k, v in counts.items():
        if k[0] == '1': # Check leftmost bit (Qubit 6)
            hits += v
    return hits / total

def train_v6(sampler, train_data, n_params):
    theta = np.random.uniform(-0.01, 0.01, n_params)
    best_theta = theta.copy()
    best_loss = float('inf')
    losses = []

    print(f"  Training V6 ({CONFIG['spsa_iterations']} iters, {n_params} params)...")

    for k in range(CONFIG['spsa_iterations']):
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        ck = 0.1 / (k+1)**0.101
        ak = CONFIG['spsa_lr'] / (k+1)**0.602

        tp = theta + ck*delta
        tm = theta - ck*delta

        circuits = []
        for v1, v2, _ in train_data:
            circuits.append(build_goldilocks_circuit(v1, v2, tp))
            circuits.append(build_goldilocks_circuit(v1, v2, tm))

        res = sampler.run(circuits, shots=CONFIG['shots']).result()

        loss_p, loss_m = 0, 0
        for i, (_, _, label) in enumerate(train_data):
            # Label 0 = Similar, 1 = Dissimilar
            # Prediction = P(Ancilla=1)
            pp = get_ancilla_prob(res[i*2].data.meas.get_counts())
            pm = get_ancilla_prob(res[i*2+1].data.meas.get_counts())

            # Clamp for stability
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
    print("QUANTUM LEARNING V6: GOLDILOCKS FUSION")
    print("="*70)
    print("Strategy: V4 Architecture + V5 Scaling + Phase Kickback")
    print(f"Qubits: {CONFIG['n_qubits_per_vector']*2 + 1} (6 data + 1 ancilla)")
    print(f"Features: {CONFIG['n_features']} (dense encoding)")
    print(f"Scaling: [0, π] (fixing aliasing)")
    print("="*70)

    np.random.seed(42)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Expanded Dataset
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

    # 1. PCA to 6 dims (Dense)
    print(f"\nApplying PCA: 384D → {CONFIG['n_features']}D...")
    pca = PCA(n_components=CONFIG['n_features'])
    vecs_pca = pca.fit_transform(embeddings)
    variance = np.sum(pca.explained_variance_ratio_)
    print(f"Variance Explained: {variance:.1%}")

    # 2. SCALING FIX: [0, π] (From V5 - Critical!)
    print("Scaling to [0, π] (The V5 Fix)...")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    vectors = scaler.fit_transform(vecs_pca)

    def get_vec(word): return vectors[all_words.index(word)]
    def get_raw(word): return embeddings[all_words.index(word)]

    train_data = []
    # Ancilla=0 means Similar, Ancilla=1 means Dissimilar
    for p in train_sim:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 0.0))
    for p in train_diff:
        train_data.append((get_vec(p[0]), get_vec(p[1]), 1.0))
    np.random.shuffle(train_data)

    print(f"Training labels: {[d[2] for d in train_data]}")

    backend = AerSimulator()
    sampler = SamplerV2(mode=backend)

    # Params:
    # Layer 1: 7 (Rx) + 6 (CRz) = 13
    # Layer 2: 7 (Ry) = 7
    # Total = 20 params
    n_params = 20

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n" + "="*70)
    print("[BASELINE] RANDOM PARAMETERS")
    print("="*70)

    theta_rand = np.random.uniform(-np.pi, np.pi, n_params)
    circuits_rand = [build_goldilocks_circuit(get_vec(p[0]), get_vec(p[1]), theta_rand) for p in test_pairs]
    res_rand = sampler.run(circuits_rand, shots=CONFIG['shots']).result()

    preds_rand = []
    targets = []
    for i, (c1, c2) in enumerate(test_pairs):
        prob_dissim = get_ancilla_prob(res_rand[i].data.meas.get_counts())
        pred_sim = 1.0 - prob_dissim
        preds_rand.append(pred_sim)

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
    print("[TRAINING] V6 GOLDILOCKS CIRCUIT")
    print("="*70)

    theta_opt, losses = train_v6(sampler, train_data, n_params)

    # =========================================================================
    # EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("[EVALUATION]")
    print("="*70)

    circuits = [build_goldilocks_circuit(get_vec(p[0]), get_vec(p[1]), theta_opt) for p in test_pairs]
    res = sampler.run(circuits, shots=CONFIG['shots']).result()

    preds = []
    print(f"\n{'Pair':<20} {'Truth':<10} {'Pred':<10} {'Verdict'}")
    print("-"*55)

    for i, (c1, c2) in enumerate(test_pairs):
        # Prediction is 1 - P(Ancilla=1)
        prob_dissim = get_ancilla_prob(res[i].data.meas.get_counts())
        pred_sim = 1.0 - prob_dissim
        preds.append(pred_sim)

        truth = targets[i]
        match = "✅" if (truth > 0.4) == (pred_sim > 0.5) else "❌"
        print(f"{c1+'-'+c2:<20} {truth:.4f}     {pred_sim:.4f}     {match}")

    corr, p_value = stats.pearsonr(preds, targets)
    span = max(preds) - min(preds)
    training_effect = corr - corr_rand

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("QUANTUM LEARNING V6 - SUMMARY")
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
