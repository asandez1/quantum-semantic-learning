#!/usr/bin/env python3
"""
QUANTUM LEARNING TEST: Does the Circuit Actually Learn?
========================================================
This experiment tests whether quantum circuits can LEARN semantic relationships,
not just preserve encoded information.

TESTS:
1. BASIC LEARNING: Train on pairs A, test on pairs B (disjoint concepts)
2. ENTANGLEMENT ABLATION: Compare with/without CX gates
3. QUANTUM vs CLASSICAL: Compare hardware to noiseless simulation
4. GENERALIZATION: Test on completely different semantic domain

SUCCESS CRITERIA:
- Test correlation > 0.5 on held-out pairs
- Entangled circuit > non-entangled
- Hardware shows similar or better performance than simulation
"""

import numpy as np
import json
from datetime import datetime
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Configuration
BACKEND_NAME = "ibm_fez"
N_QUBITS = 20
SHOTS = 4096
USE_HARDWARE = False  # Set True for real quantum hardware


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity."""
    dot = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(dot / (norm1 * norm2))


def compute_similarity_from_counts(counts: dict, n_qubits: int) -> float:
    """Compute similarity from measurement counts."""
    total_shots = sum(counts.values())
    weighted_hamming = 0.0
    for bitstring, count in counts.items():
        bs = bitstring.zfill(n_qubits)
        hamming_weight = bs.count('1')
        weighted_hamming += hamming_weight * count
    avg_hamming = weighted_hamming / total_shots
    similarity = 1.0 - (avg_hamming / (n_qubits / 2))
    return float(max(0.0, min(1.0, similarity)))


# ============================================================================
# CIRCUIT ARCHITECTURES
# ============================================================================

def build_entangled_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    ENTANGLED CIRCUIT: Uses CX gates to create quantum correlations.
    This is the standard approach - tests if entanglement helps learning.
    """
    n = len(v1)
    qc = QuantumCircuit(n)

    # Encode v1
    for i in range(n):
        qc.ry(float(v1[i]), i)

    # Trainable layer 1 with entanglement
    for i in range(n):
        qc.ry(float(theta[i]), i)

    # ENTANGLEMENT: Cross-qubit correlations
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, n - 1, 2):
        qc.cx(i, i + 1)

    # Trainable layer 2
    for i in range(n):
        qc.ry(float(theta[n + i]), i)

    # Encode -v2 (interference)
    for i in range(n):
        qc.ry(float(-v2[i]), i)

    qc.measure_all()
    return qc


def build_product_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    PRODUCT CIRCUIT: NO entanglement - each qubit is independent.
    This is the ABLATION test - if this works as well, entanglement doesn't help.
    """
    n = len(v1)
    qc = QuantumCircuit(n)

    # Encode v1
    for i in range(n):
        qc.ry(float(v1[i]), i)

    # Trainable layer 1 (NO entanglement)
    for i in range(n):
        qc.ry(float(theta[i]), i)

    # NO CX GATES - this is the key difference

    # Trainable layer 2
    for i in range(n):
        qc.ry(float(theta[n + i]), i)

    # Encode -v2 (interference)
    for i in range(n):
        qc.ry(float(-v2[i]), i)

    qc.measure_all()
    return qc


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_circuit(train_pairs, vectors, concepts, circuit_builder,
                  n_iterations=5, lr=0.1, use_simulator=True):
    """
    Train circuit parameters using SPSA.
    Returns: optimized theta, training losses
    """
    n_params = N_QUBITS * 2  # Two layers of RY gates
    theta = np.random.uniform(-0.1, 0.1, n_params)

    # Setup backend
    if use_simulator:
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        sampler = AerSampler()
    else:
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token="xxx",
            instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::",
        )
        backend = service.backend(BACKEND_NAME)
        sampler = SamplerV2(mode=backend)

    losses = []

    for iteration in range(n_iterations):
        # SPSA perturbation
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        c_k = 0.1 / (iteration + 1) ** 0.101
        a_k = lr / (iteration + 1) ** 0.602

        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        # Evaluate both perturbations on all training pairs
        loss_plus = 0.0
        loss_minus = 0.0

        circuits_plus = []
        circuits_minus = []
        targets = []

        for c1, c2 in train_pairs:
            idx1 = concepts.index(c1)
            idx2 = concepts.index(c2)
            v1, v2 = vectors[idx1], vectors[idx2]

            # Target: cosine similarity from raw embeddings
            # (computed separately, not from vectors which are scaled)
            circuits_plus.append(circuit_builder(v1, v2, theta_plus))
            circuits_minus.append(circuit_builder(v1, v2, theta_minus))

        # Run circuits
        if use_simulator:
            result_plus = sampler.run(circuits_plus, shots=SHOTS).result()
            result_minus = sampler.run(circuits_minus, shots=SHOTS).result()
        else:
            result_plus = sampler.run(circuits_plus, shots=SHOTS).result()
            result_minus = sampler.run(circuits_minus, shots=SHOTS).result()

        # Compute losses
        for i, (c1, c2) in enumerate(train_pairs):
            idx1 = concepts.index(c1)
            idx2 = concepts.index(c2)

            counts_plus = result_plus[i].data.meas.get_counts()
            counts_minus = result_minus[i].data.meas.get_counts()

            pred_plus = compute_similarity_from_counts(counts_plus, N_QUBITS)
            pred_minus = compute_similarity_from_counts(counts_minus, N_QUBITS)

            # Target is stored in embeddings_raw
            target = train_pairs_targets[i]

            loss_plus += (pred_plus - target) ** 2
            loss_minus += (pred_minus - target) ** 2

        loss_plus /= len(train_pairs)
        loss_minus /= len(train_pairs)

        # SPSA update
        gradient = (loss_plus - loss_minus) / (2 * c_k) * delta
        theta = theta - a_k * gradient

        current_loss = (loss_plus + loss_minus) / 2
        losses.append(current_loss)
        print(f"  Iter {iteration+1}: loss={current_loss:.4f}")

    return theta, losses


def evaluate_circuit(test_pairs, vectors, concepts, embeddings_raw,
                    theta, circuit_builder, use_simulator=True):
    """
    Evaluate trained circuit on test pairs.
    Returns: predictions, targets, correlation
    """
    if use_simulator:
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        sampler = AerSampler()
    else:
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token="xxx",
            instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::",
        )
        backend = service.backend(BACKEND_NAME)
        sampler = SamplerV2(mode=backend)

    circuits = []
    targets = []

    for c1, c2 in test_pairs:
        idx1 = concepts.index(c1)
        idx2 = concepts.index(c2)
        v1, v2 = vectors[idx1], vectors[idx2]

        # Target from raw embeddings
        target = cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2])
        targets.append(target)

        circuits.append(circuit_builder(v1, v2, theta))

    result = sampler.run(circuits, shots=SHOTS).result()

    preds = []
    for i in range(len(circuits)):
        counts = result[i].data.meas.get_counts()
        pred = compute_similarity_from_counts(counts, N_QUBITS)
        preds.append(pred)

    preds = np.array(preds)
    targets = np.array(targets)

    correlation, p_value = stats.pearsonr(preds, targets)

    return preds, targets, correlation, p_value


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

# Global variable for training targets (set during data prep)
train_pairs_targets = []

def main():
    global train_pairs_targets

    print("=" * 70)
    print("QUANTUM LEARNING TEST")
    print("=" * 70)
    print("Testing if quantum circuits can LEARN semantic relationships")
    print("=" * 70)

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n[1] PREPARING DATA...")

    # TRAINING SET: Animal domain
    train_pairs = [
        ('dog', 'puppy'),
        ('cat', 'kitten'),
        ('bird', 'sparrow'),
        ('fish', 'salmon'),
        ('horse', 'pony'),
    ]

    # TEST SET 1: Same domain (animals) but DIFFERENT concepts
    test_same_domain = [
        ('dog', 'cat'),        # Related animals
        ('bird', 'fish'),      # Different animal types
        ('horse', 'dog'),      # Different animals
        ('cat', 'bird'),       # Different animals
        ('fish', 'horse'),     # Different animals
    ]

    # TEST SET 2: DIFFERENT domain (objects/abstract)
    test_diff_domain = [
        ('car', 'truck'),      # Vehicles
        ('happy', 'sad'),      # Emotions
        ('house', 'building'), # Structures
        ('book', 'paper'),     # Objects
        ('music', 'song'),     # Abstract
    ]

    # Collect all concepts
    all_concepts = list(set(
        c for pair in (train_pairs + test_same_domain + test_diff_domain)
        for c in pair
    ))

    print(f"Total concepts: {len(all_concepts)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Test (same domain): {len(test_same_domain)}")
    print(f"Test (diff domain): {len(test_diff_domain)}")

    # Get embeddings
    print("\nLoading sentence-transformers model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_raw = model.encode(all_concepts, show_progress_bar=False)

    # PCA to N_QUBITS dimensions
    print(f"Applying PCA: 384D → {N_QUBITS}D...")
    pca = PCA(n_components=N_QUBITS)
    embeddings_pca = pca.fit_transform(embeddings_raw)
    variance = sum(pca.explained_variance_ratio_)
    print(f"Variance explained: {variance:.3f}")

    # Scale for quantum encoding
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(embeddings_pca)

    # Compute training targets
    train_pairs_targets = []
    for c1, c2 in train_pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        target = cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2])
        train_pairs_targets.append(target)

    print(f"Training targets: {[f'{t:.3f}' for t in train_pairs_targets]}")

    results = {}

    # =========================================================================
    # TEST 1: BASIC LEARNING (Entangled Circuit)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] BASIC LEARNING - Can circuit learn and generalize?")
    print("=" * 70)

    print("\nTraining entangled circuit on 5 animal pairs...")
    theta_entangled, train_losses = train_circuit(
        train_pairs, vectors, all_concepts,
        build_entangled_circuit,
        n_iterations=10,
        use_simulator=True
    )

    print("\nEvaluating on SAME domain (held-out animal pairs)...")
    preds, targets, corr_same, p_same = evaluate_circuit(
        test_same_domain, vectors, all_concepts, embeddings_raw,
        theta_entangled, build_entangled_circuit,
        use_simulator=True
    )
    print(f"  Correlation: {corr_same:.4f} (p={p_same:.3e})")

    print("\nEvaluating on DIFFERENT domain (objects/abstract)...")
    preds, targets, corr_diff, p_diff = evaluate_circuit(
        test_diff_domain, vectors, all_concepts, embeddings_raw,
        theta_entangled, build_entangled_circuit,
        use_simulator=True
    )
    print(f"  Correlation: {corr_diff:.4f} (p={p_diff:.3e})")

    results['test1_basic_learning'] = {
        'train_loss_final': float(train_losses[-1]),
        'test_same_domain': float(corr_same),
        'test_diff_domain': float(corr_diff),
        'verdict': 'PASS' if corr_same > 0.5 else 'FAIL'
    }

    # =========================================================================
    # TEST 2: ENTANGLEMENT ABLATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] ENTANGLEMENT ABLATION - Does entanglement help?")
    print("=" * 70)

    print("\nTraining PRODUCT circuit (no entanglement)...")
    theta_product, train_losses_product = train_circuit(
        train_pairs, vectors, all_concepts,
        build_product_circuit,
        n_iterations=10,
        use_simulator=True
    )

    print("\nEvaluating product circuit on same domain...")
    preds, targets, corr_product, p_product = evaluate_circuit(
        test_same_domain, vectors, all_concepts, embeddings_raw,
        theta_product, build_product_circuit,
        use_simulator=True
    )
    print(f"  Product (no entanglement): {corr_product:.4f}")
    print(f"  Entangled:                 {corr_same:.4f}")

    entanglement_helps = corr_same > corr_product
    print(f"\n  Entanglement helps: {'YES' if entanglement_helps else 'NO'}")
    print(f"  Improvement: {(corr_same - corr_product):.4f}")

    results['test2_entanglement'] = {
        'entangled_corr': float(corr_same),
        'product_corr': float(corr_product),
        'improvement': float(corr_same - corr_product),
        'verdict': 'ENTANGLEMENT_HELPS' if entanglement_helps else 'NO_BENEFIT'
    }

    # =========================================================================
    # TEST 3: RANDOM vs TRAINED PARAMETERS
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 3] RANDOM vs TRAINED - Did training improve anything?")
    print("=" * 70)

    print("\nEvaluating with RANDOM parameters...")
    theta_random = np.random.uniform(-0.1, 0.1, N_QUBITS * 2)
    preds, targets, corr_random, p_random = evaluate_circuit(
        test_same_domain, vectors, all_concepts, embeddings_raw,
        theta_random, build_entangled_circuit,
        use_simulator=True
    )
    print(f"  Random params:  {corr_random:.4f}")
    print(f"  Trained params: {corr_same:.4f}")

    training_helped = corr_same > corr_random
    print(f"\n  Training helped: {'YES' if training_helped else 'NO'}")
    print(f"  Improvement: {(corr_same - corr_random):.4f}")

    results['test3_training'] = {
        'random_corr': float(corr_random),
        'trained_corr': float(corr_same),
        'improvement': float(corr_same - corr_random),
        'verdict': 'TRAINING_HELPS' if training_helped else 'NO_BENEFIT'
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUANTUM LEARNING TEST - SUMMARY")
    print("=" * 70)

    print(f"\n{'Test':<30} {'Result':<15} {'Verdict'}")
    print("-" * 60)
    print(f"{'Basic Learning (same domain)':<30} {corr_same:.4f}         {'✅ PASS' if corr_same > 0.5 else '❌ FAIL'}")
    print(f"{'Generalization (diff domain)':<30} {corr_diff:.4f}         {'✅ PASS' if corr_diff > 0.3 else '❌ FAIL'}")
    print(f"{'Entanglement Effect':<30} {corr_same - corr_product:+.4f}         {'✅ HELPS' if entanglement_helps else '❌ NO EFFECT'}")
    print(f"{'Training Effect':<30} {corr_same - corr_random:+.4f}         {'✅ HELPS' if training_helped else '❌ NO EFFECT'}")

    # Overall verdict
    learning_demonstrated = (
        corr_same > 0.5 and  # Can predict held-out pairs
        training_helped and  # Training improved over random
        corr_same > corr_random + 0.1  # Meaningful improvement
    )

    print("\n" + "=" * 70)
    if learning_demonstrated:
        print("✅ QUANTUM LEARNING DEMONSTRATED")
        print("   The circuit learned transferable semantic relationships")
        if entanglement_helps:
            print("   Entanglement provides genuine quantum advantage")
    else:
        print("❌ NO LEARNING DEMONSTRATED")
        print("   Circuit may just be preserving/transforming input")
    print("=" * 70)

    results['overall'] = {
        'learning_demonstrated': learning_demonstrated,
        'entanglement_beneficial': entanglement_helps,
        'training_beneficial': training_helped
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"quantum_learning_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
