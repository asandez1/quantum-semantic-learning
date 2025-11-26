#!/usr/bin/env python3
"""
QUANTUM LEARNING TEST V3: Ancilla-Based Classification
=======================================================

WHY V1 AND V2 FAILED TO SHOW "LEARNING":
- V1: Interference pattern (RY(v1)·RY(-v2)) naturally detects similarity
- V2: Cross-register CX gates naturally create similarity-correlated outputs
- In both cases, the circuit STRUCTURE does the work, not learned parameters

V3 SOLUTION: ANCILLA-BASED CLASSIFICATION
------------------------------------------
1. Encode v1 and v2 on separate registers (like V2)
2. Add an ANCILLA qubit initialized to |0⟩
3. Trainable layers connect data qubits to ancilla
4. ONLY measure the ancilla - it must LEARN to indicate similarity

KEY INSIGHT:
- With random parameters, ancilla output is RANDOM (no similarity info)
- After training, ancilla should correlate with similarity
- This FORCES the circuit to learn - no free lunch from structure

SUCCESS CRITERIA:
- Random params: correlation ≈ 0 (ancilla knows nothing)
- Trained params: correlation > 0.5 (ancilla learned similarity)
- Improvement: trained >> random (proves learning happened)
"""

import numpy as np
import json
import os
import subprocess
from datetime import datetime
from scipy import stats

# Qiskit Imports
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

# ML Imports
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "experiment_name": "quantum_learning_v3_ancilla",
    "n_qubits_per_vector": 3,  # 3 qubits per vector
    "n_ancilla": 1,            # 1 ancilla qubit for output
    "shots": 4096,
    "spsa_iterations": 200,    # More iterations for harder task
    "spsa_lr": 0.15,
    "similarity_threshold": 0.5,  # Above this = "similar"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(dot / (norm1 * norm2))


def get_git_info():
    """Get git commit hash."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        return commit
    except:
        return "nogit"


# ============================================================================
# V3 ARCHITECTURE: Ancilla-Based Learning
# ============================================================================

def build_ancilla_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    ANCILLA-BASED CLASSIFICATION CIRCUIT

    Layout:
        Qubits 0,1,2: Encode v1
        Qubits 3,4,5: Encode v2
        Qubit 6:      ANCILLA (output) - starts as |0⟩

    The circuit must LEARN to flip the ancilla for similar pairs.
    With random theta, ancilla output is random.

    Parameters:
        - Layer 1: 7 qubits × 1 = 7 params (local rotations)
        - Layer 2: 7 qubits × 1 = 7 params (after entanglement)
        - Layer 3: 7 qubits × 1 = 7 params (before measurement)
        Total: 21 parameters
    """
    n = len(v1)  # 3
    n_data = 2 * n  # 6 data qubits
    n_total = n_data + 1  # 7 total (including ancilla)
    ancilla_idx = n_total - 1  # Index 6

    qc = QuantumCircuit(n_total)

    # === ENCODING LAYER (Fixed) ===
    # Encode v1 on qubits 0,1,2
    for i in range(n):
        qc.ry(float(v1[i]), i)

    # Encode v2 on qubits 3,4,5
    for i in range(n):
        qc.ry(float(v2[i]), n + i)

    # Ancilla starts as |0⟩ (no encoding)
    qc.barrier()

    # === TRAINABLE LAYER 1: Local rotations ===
    for i in range(n_total):
        qc.ry(float(theta[i]), i)

    qc.barrier()

    # === ENTANGLEMENT: Connect data qubits to ancilla ===
    # This is where the circuit must LEARN to route similarity info

    # Connect v1 qubits to ancilla
    for i in range(n):
        qc.cx(i, ancilla_idx)

    # Connect v2 qubits to ancilla
    for i in range(n):
        qc.cx(n + i, ancilla_idx)

    qc.barrier()

    # === TRAINABLE LAYER 2: After entanglement ===
    for i in range(n_total):
        qc.ry(float(theta[n_total + i]), i)

    # More entanglement - cross connections
    for i in range(n):
        qc.cx(i, n + i)  # v1 to v2

    qc.barrier()

    # === TRAINABLE LAYER 3: Final processing ===
    for i in range(n_total):
        qc.ry(float(theta[2 * n_total + i]), i)

    # Final connections to ancilla
    qc.cx(0, ancilla_idx)
    qc.cx(n, ancilla_idx)

    # Only measure the ANCILLA
    qc.measure_all()

    return qc


def build_ancilla_product_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    PRODUCT VERSION (no entanglement) - Ablation test.
    Same structure but NO CX gates.
    """
    n = len(v1)
    n_data = 2 * n
    n_total = n_data + 1

    qc = QuantumCircuit(n_total)

    # Encoding
    for i in range(n):
        qc.ry(float(v1[i]), i)
        qc.ry(float(v2[i]), n + i)

    qc.barrier()

    # Trainable layers (NO entanglement)
    for i in range(n_total):
        qc.ry(float(theta[i]), i)

    qc.barrier()

    for i in range(n_total):
        qc.ry(float(theta[n_total + i]), i)

    qc.barrier()

    for i in range(n_total):
        qc.ry(float(theta[2 * n_total + i]), i)

    qc.measure_all()
    return qc


# ============================================================================
# SIMILARITY EXTRACTION (Ancilla-based)
# ============================================================================

def get_ancilla_probability(counts: dict, n_total: int) -> float:
    """
    Extract P(ancilla = 1) from measurement counts.

    The ancilla is the LAST qubit (index n_total-1).
    In Qiskit's little-endian convention, it's the FIRST bit of the string.

    Returns probability that ancilla is |1⟩.
    """
    total_shots = sum(counts.values())
    ancilla_one_count = 0

    for bitstring, count in counts.items():
        # Qiskit uses little-endian: rightmost bit is qubit 0
        # Ancilla is qubit n_total-1, so it's the leftmost bit
        bs = bitstring.zfill(n_total)
        ancilla_bit = bs[0]  # Leftmost = highest index qubit

        if ancilla_bit == '1':
            ancilla_one_count += count

    return ancilla_one_count / total_shots


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def binary_cross_entropy(pred: float, target: float, eps: float = 1e-7) -> float:
    """
    Binary cross-entropy loss.
    pred: P(ancilla=1)
    target: 1 if similar, 0 if dissimilar
    """
    pred = np.clip(pred, eps, 1 - eps)
    return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))


# ============================================================================
# TRAINING
# ============================================================================

def train_circuit(sampler, train_data, circuit_builder):
    """
    Train circuit with SPSA optimizer.
    train_data: list of (v1, v2, is_similar) tuples where is_similar is 0 or 1
    """
    n_total = CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']  # 7
    n_params = n_total * 3  # 21 parameters

    # Initialize near zero
    theta = np.random.uniform(-0.01, 0.01, n_params)

    losses = []
    best_theta = theta.copy()
    best_loss = float('inf')

    print(f"  Training ({CONFIG['spsa_iterations']} iterations, {n_params} parameters)...")

    for iteration in range(CONFIG['spsa_iterations']):
        # SPSA perturbation
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        c_k = 0.1 / (iteration + 1) ** 0.101
        a_k = CONFIG['spsa_lr'] / (iteration + 1) ** 0.602

        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        # Build circuits
        circuits = []
        for v1, v2, _ in train_data:
            circuits.append(circuit_builder(v1, v2, theta_plus))
            circuits.append(circuit_builder(v1, v2, theta_minus))

        # Run
        job = sampler.run(circuits, shots=CONFIG['shots'])
        result = job.result()

        # Compute losses
        loss_plus = 0.0
        loss_minus = 0.0

        for i, (v1, v2, target) in enumerate(train_data):
            counts_plus = result[2*i].data.meas.get_counts()
            counts_minus = result[2*i + 1].data.meas.get_counts()

            pred_plus = get_ancilla_probability(counts_plus, n_total)
            pred_minus = get_ancilla_probability(counts_minus, n_total)

            loss_plus += binary_cross_entropy(pred_plus, target)
            loss_minus += binary_cross_entropy(pred_minus, target)

        loss_plus /= len(train_data)
        loss_minus /= len(train_data)

        # SPSA update
        gradient = (loss_plus - loss_minus) / (2 * c_k) * delta
        theta = theta - a_k * gradient

        current_loss = (loss_plus + loss_minus) / 2
        losses.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta.copy()

        if (iteration + 1) % 40 == 0:
            print(f"    Iter {iteration+1}: loss={current_loss:.4f} (best={best_loss:.4f})")

    return best_theta, losses


def evaluate_circuit(sampler, test_data, theta, circuit_builder):
    """
    Evaluate circuit on test pairs.
    Returns predictions and targets.

    Since we're using inverted convention (ancilla=1 means dissimilar),
    we compute P(ancilla=0) = 1 - P(ancilla=1) to get "similarity prediction"
    """
    n_total = CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']

    circuits = []
    for v1, v2, _ in test_data:
        circuits.append(circuit_builder(v1, v2, theta))

    job = sampler.run(circuits, shots=CONFIG['shots'])
    result = job.result()

    preds = []
    targets = []

    for i, (v1, v2, target_sim) in enumerate(test_data):
        counts = result[i].data.meas.get_counts()
        p_one = get_ancilla_probability(counts, n_total)
        # INVERTED: P(similar) = P(ancilla=0) = 1 - P(ancilla=1)
        pred = 1.0 - p_one
        preds.append(pred)
        targets.append(target_sim)  # Keep continuous for correlation

    preds = np.array(preds)
    targets = np.array(targets)

    # Check for constant predictions
    if np.std(preds) < 1e-10:
        print(f"    WARNING: Predictions are constant ({preds[0]:.4f})")
        return preds, targets, 0.0, 1.0

    correlation, p_value = stats.pearsonr(preds, targets)
    return preds, targets, correlation, p_value


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("QUANTUM LEARNING V3: Ancilla-Based Classification")
    print("=" * 70)
    print(f"Data qubits per vector: {CONFIG['n_qubits_per_vector']}")
    print(f"Ancilla qubits: {CONFIG['n_ancilla']}")
    print(f"Total qubits: {CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']}")
    print(f"SPSA iterations: {CONFIG['spsa_iterations']}")
    print("=" * 70)
    print("\nKEY TEST: With random params, ancilla should be random.")
    print("          After training, ancilla should indicate similarity.")
    print("=" * 70)

    np.random.seed(42)

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n[1] PREPARING DATA...")

    # Training pairs with EXPLICIT labels (similar/dissimilar)
    # More pairs for better training
    train_similar = [
        ('dog', 'puppy'),      # Very similar
        ('cat', 'kitten'),     # Very similar
        ('horse', 'pony'),     # Similar
        ('wolf', 'dog'),       # Similar
        ('lion', 'tiger'),     # Similar
        ('mouse', 'rat'),      # Similar
    ]

    train_dissimilar = [
        ('dog', 'car'),        # Very different
        ('cat', 'mountain'),   # Very different
        ('bird', 'computer'),  # Very different
        ('fish', 'happiness'), # Very different
        ('horse', 'music'),    # Very different
        ('lion', 'book'),      # Very different
    ]

    # Test pairs (held out)
    test_pairs = [
        ('eagle', 'hawk'),      # Similar (should predict ~1)
        ('shark', 'dolphin'),   # Similar
        ('cow', 'bull'),        # Similar
        ('happy', 'car'),       # Dissimilar (should predict ~0)
        ('tree', 'phone'),      # Dissimilar
        ('water', 'anger'),     # Dissimilar
        ('snake', 'lizard'),    # Similar
        ('apple', 'rocket'),    # Dissimilar
    ]

    all_concepts = list(set(
        c for pair in (train_similar + train_dissimilar + test_pairs) for c in pair
    ))

    print(f"Total concepts: {len(all_concepts)}")
    print(f"Training similar pairs: {len(train_similar)}")
    print(f"Training dissimilar pairs: {len(train_dissimilar)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Get embeddings
    print("\nLoading sentence-transformers...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_raw = model.encode(all_concepts, show_progress_bar=False)

    # PCA
    n_dim = CONFIG['n_qubits_per_vector']
    print(f"Applying PCA: 384D → {n_dim}D...")
    pca = PCA(n_components=n_dim)
    embeddings_pca = pca.fit_transform(embeddings_raw)
    print(f"Variance explained: {sum(pca.explained_variance_ratio_):.3f}")

    # Scale
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(embeddings_pca)

    # Create training data with binary labels
    # NOTE: Using INVERTED convention - similar=0, dissimilar=1
    # This aligns with quantum XOR behavior where similar inputs cancel
    def make_train_data(similar_pairs, dissimilar_pairs):
        data = []
        for c1, c2 in similar_pairs:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            data.append((vectors[idx1], vectors[idx2], 0.0))  # Label = 0 (similar → ancilla stays |0⟩)
        for c1, c2 in dissimilar_pairs:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            data.append((vectors[idx1], vectors[idx2], 1.0))  # Label = 1 (dissimilar → ancilla flips)
        return data

    # Create test data with continuous similarity (for correlation)
    def make_test_data(pairs):
        data = []
        for c1, c2 in pairs:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            sim = cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2])
            data.append((vectors[idx1], vectors[idx2], sim))
        return data

    train_data = make_train_data(train_similar, train_dissimilar)
    test_data = make_test_data(test_pairs)

    # Shuffle training data
    np.random.shuffle(train_data)

    print(f"\nTraining labels: {[d[2] for d in train_data]}")
    print(f"Test similarities: {[f'{d[2]:.2f}' for d in test_data]}")

    # =========================================================================
    # SETUP
    # =========================================================================
    print("\nInitializing AerSimulator...")
    backend = AerSimulator()
    sampler = SamplerV2(mode=backend)

    results = {}
    n_total = CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']
    n_params = n_total * 3

    # =========================================================================
    # CRITICAL TEST: Random Parameters Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("[BASELINE] RANDOM PARAMETERS - Should show NO correlation")
    print("=" * 70)

    theta_random = np.random.uniform(-np.pi, np.pi, n_params)
    preds_rand, targets_rand, corr_random, p_random = evaluate_circuit(
        sampler, test_data, theta_random, build_ancilla_circuit
    )
    print(f"\n  Random params correlation: {corr_random:.4f} (p={p_random:.3e})")
    print(f"  Predictions range: [{min(preds_rand):.3f}, {max(preds_rand):.3f}]")
    print(f"  Expected: correlation ≈ 0 (random)")

    results['random_baseline'] = {
        'correlation': float(corr_random),
        'preds': preds_rand.tolist(),
    }

    # =========================================================================
    # TEST 1: Entangled Circuit Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] TRAINING ENTANGLED CIRCUIT")
    print("=" * 70)

    theta_trained, losses = train_circuit(
        sampler, train_data, build_ancilla_circuit
    )

    print("\nEvaluating trained circuit...")
    preds_trained, targets_trained, corr_trained, p_trained = evaluate_circuit(
        sampler, test_data, theta_trained, build_ancilla_circuit
    )
    print(f"  Trained params correlation: {corr_trained:.4f} (p={p_trained:.3e})")
    print(f"  Predictions range: [{min(preds_trained):.3f}, {max(preds_trained):.3f}]")

    training_effect = corr_trained - corr_random
    print(f"\n  TRAINING EFFECT: {training_effect:+.4f}")
    print(f"  Random: {corr_random:.4f} → Trained: {corr_trained:.4f}")

    results['entangled'] = {
        'correlation': float(corr_trained),
        'training_effect': float(training_effect),
        'final_loss': float(losses[-1]),
        'best_loss': float(min(losses)),
    }

    # =========================================================================
    # TEST 2: Product Circuit (Ablation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] PRODUCT CIRCUIT (no entanglement)")
    print("=" * 70)

    theta_product, losses_prod = train_circuit(
        sampler, train_data, build_ancilla_product_circuit
    )

    preds_prod, _, corr_product, _ = evaluate_circuit(
        sampler, test_data, theta_product, build_ancilla_product_circuit
    )

    print(f"\n  Product correlation: {corr_product:.4f}")
    print(f"  Entangled correlation: {corr_trained:.4f}")

    entanglement_helps = corr_trained > corr_product
    print(f"  Entanglement helps: {'YES ✅' if entanglement_helps else 'NO ❌'}")

    results['product'] = {
        'correlation': float(corr_product),
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUANTUM LEARNING V3 - SUMMARY")
    print("=" * 70)

    # Determine if learning occurred
    learning_threshold = 0.15  # Need significant improvement over random
    learning_occurred = (
        corr_trained > corr_random + learning_threshold and
        corr_trained > 0.3
    )

    print(f"\n{'Metric':<30} {'Value':<12} {'Verdict'}")
    print("-" * 60)
    print(f"{'Random params correlation':<30} {corr_random:+.4f}       (baseline)")
    print(f"{'Trained params correlation':<30} {corr_trained:+.4f}       {'✅' if corr_trained > 0.3 else '❌'}")
    print(f"{'Training improvement':<30} {training_effect:+.4f}       {'✅ LEARNING' if training_effect > learning_threshold else '❌ NO LEARNING'}")
    print(f"{'Entanglement effect':<30} {corr_trained - corr_product:+.4f}       {'✅ HELPS' if entanglement_helps else '❌'}")

    print("\n" + "=" * 70)
    if learning_occurred:
        print("✅ QUANTUM LEARNING DEMONSTRATED!")
        print(f"   - Random baseline: {corr_random:.4f}")
        print(f"   - After training:  {corr_trained:.4f}")
        print(f"   - Improvement:     {training_effect:+.4f}")
        print("   - The circuit LEARNED to detect similarity!")
    else:
        print("❌ NO LEARNING DEMONSTRATED")
        if training_effect <= learning_threshold:
            print(f"   - Training improvement ({training_effect:+.4f}) below threshold ({learning_threshold})")
        if corr_trained <= 0.3:
            print(f"   - Trained correlation ({corr_trained:.4f}) too low")
    print("=" * 70)

    # Save results
    output = {
        "config": CONFIG,
        "git_commit": get_git_info(),
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "learning_demonstrated": bool(learning_occurred),
        "summary": {
            "random_correlation": float(corr_random),
            "trained_correlation": float(corr_trained),
            "training_effect": float(training_effect),
            "entanglement_effect": float(corr_trained - corr_product),
        }
    }

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    filename = f"{CONFIG['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
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
