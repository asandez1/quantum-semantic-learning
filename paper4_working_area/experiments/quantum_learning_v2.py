#!/usr/bin/env python3
"""
QUANTUM LEARNING TEST V2: Architecture That REQUIRES Learning
=============================================================

PROBLEM WITH V1:
- The RY(v1) → θ → RY(-v2) pattern is a HARD-CODED similarity detector
- Parameters θ are useless - random works equally well
- No actual learning happens

SOLUTION (This Version):
1. SEPARATE ENCODING: v1 and v2 on different qubit registers
2. LEARNED COMPARISON: Circuit must learn to compare (no free interference)
3. CONTRASTIVE LOSS: Push dissimilar pairs apart, pull similar together

SUCCESS CRITERIA:
- Trained params >> Random params (proves learning)
- Test correlation > 0.5 on held-out pairs (proves generalization)
- Entangled > Product (proves quantum helps)
"""

import numpy as np
import json
import os
import subprocess
import uuid
from datetime import datetime
from scipy import stats

# Qiskit Imports
from qiskit import QuantumCircuit
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
    "experiment_name": "quantum_learning_v2",
    "n_qubits_per_vector": 3,  # 3 qubits per vector = 6 total
    "shots": 4096,
    "spsa_iterations": 150,
    "spsa_lr": 0.1,
    "contrastive_margin": 0.3,  # Margin for contrastive loss
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
# NEW ARCHITECTURE: Learned Comparison Circuit
# ============================================================================

def build_learned_comparison_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    LEARNED COMPARISON CIRCUIT

    Key Design: v1 and v2 are encoded on SEPARATE qubit registers.
    The circuit MUST learn to compare them - no free interference!

    Layout:
        Qubits 0,1,2: Encode v1
        Qubits 3,4,5: Encode v2

    Trainable layers connect them and must LEARN the comparison.
    """
    n = len(v1)  # Should be 3
    total_qubits = 2 * n  # 6 qubits
    qc = QuantumCircuit(total_qubits)

    # === ENCODING LAYER (Fixed - not trainable) ===
    # Encode v1 on qubits 0,1,2
    for i in range(n):
        qc.ry(float(v1[i]), i)

    # Encode v2 on qubits 3,4,5
    for i in range(n):
        qc.ry(float(v2[i]), n + i)

    qc.barrier()

    # === TRAINABLE COMPARISON LAYER 1 ===
    # Local rotations
    for i in range(total_qubits):
        qc.ry(float(theta[i]), i)

    # Cross-register entanglement (THIS IS WHERE LEARNING HAPPENS)
    # Connect v1 qubits to v2 qubits - circuit must learn what connections matter
    for i in range(n):
        qc.cx(i, n + i)  # Connect qubit i to qubit n+i

    qc.barrier()

    # === TRAINABLE COMPARISON LAYER 2 ===
    for i in range(total_qubits):
        qc.ry(float(theta[total_qubits + i]), i)

    # More cross-register entanglement
    for i in range(n - 1):
        qc.cx(i, n + i + 1)
        qc.cx(n + i, i + 1)

    qc.barrier()

    # === TRAINABLE OUTPUT LAYER ===
    for i in range(total_qubits):
        qc.ry(float(theta[2 * total_qubits + i]), i)

    qc.measure_all()
    return qc


def build_product_comparison_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    PRODUCT (NO ENTANGLEMENT) VERSION - Ablation test
    Same structure but NO CX gates between registers.
    """
    n = len(v1)
    total_qubits = 2 * n
    qc = QuantumCircuit(total_qubits)

    # Encode v1 and v2 separately
    for i in range(n):
        qc.ry(float(v1[i]), i)
        qc.ry(float(v2[i]), n + i)

    qc.barrier()

    # Trainable layers (NO cross-register entanglement)
    for i in range(total_qubits):
        qc.ry(float(theta[i]), i)

    qc.barrier()

    for i in range(total_qubits):
        qc.ry(float(theta[total_qubits + i]), i)

    qc.barrier()

    for i in range(total_qubits):
        qc.ry(float(theta[2 * total_qubits + i]), i)

    qc.measure_all()
    return qc


# ============================================================================
# SIMILARITY EXTRACTION
# ============================================================================

def compute_similarity_from_counts(counts: dict, n_qubits: int) -> float:
    """
    Extract similarity score from measurement counts.

    Strategy: Use correlation between first-half and second-half measurements.
    If circuit learned to compare, similar inputs → correlated outputs.
    """
    total_shots = sum(counts.values())
    n_half = n_qubits // 2

    # Count how often first-half and second-half agree
    agreement = 0
    for bitstring, count in counts.items():
        bs = bitstring.zfill(n_qubits)
        first_half = bs[:n_half]
        second_half = bs[n_half:]

        # Count matching bits
        matches = sum(1 for a, b in zip(first_half, second_half) if a == b)
        agreement += (matches / n_half) * count

    similarity = agreement / total_shots
    return float(similarity)


# ============================================================================
# CONTRASTIVE LOSS
# ============================================================================

def contrastive_loss(pred_sim: float, target_sim: float, margin: float = 0.3) -> float:
    """
    Contrastive loss that:
    - Pulls similar pairs together (high target → high pred)
    - Pushes dissimilar pairs apart (low target → low pred)

    L = target * (1 - pred)^2 + (1 - target) * max(0, pred - margin)^2
    """
    # Normalize target to [0, 1]
    target_norm = (target_sim + 1) / 2  # cosine is [-1, 1]

    # Pull term: similar pairs should have high prediction
    pull = target_norm * (1 - pred_sim) ** 2

    # Push term: dissimilar pairs should have low prediction
    push = (1 - target_norm) * max(0, pred_sim - margin) ** 2

    return pull + push


# ============================================================================
# TRAINING
# ============================================================================

def train_circuit(sampler, train_data, circuit_builder, use_contrastive=True):
    """
    Train circuit with SPSA optimizer.

    train_data: list of (v1, v2, target_similarity) tuples
    """
    n_qubits = CONFIG['n_qubits_per_vector'] * 2  # 6 total
    n_params = n_qubits * 3  # 3 trainable layers

    # Initialize near zero (identity-like start)
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

        # Build circuits for all pairs
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
        n_pairs = len(train_data)

        for i, (v1, v2, target) in enumerate(train_data):
            counts_plus = result[2*i].data.meas.get_counts()
            counts_minus = result[2*i + 1].data.meas.get_counts()

            pred_plus = compute_similarity_from_counts(counts_plus, n_qubits)
            pred_minus = compute_similarity_from_counts(counts_minus, n_qubits)

            if use_contrastive:
                loss_plus += contrastive_loss(pred_plus, target, CONFIG['contrastive_margin'])
                loss_minus += contrastive_loss(pred_minus, target, CONFIG['contrastive_margin'])
            else:
                # Simple MSE
                target_norm = (target + 1) / 2
                loss_plus += (pred_plus - target_norm) ** 2
                loss_minus += (pred_minus - target_norm) ** 2

        loss_plus /= n_pairs
        loss_minus /= n_pairs

        # SPSA update
        gradient = (loss_plus - loss_minus) / (2 * c_k) * delta
        theta = theta - a_k * gradient

        current_loss = (loss_plus + loss_minus) / 2
        losses.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta.copy()

        if (iteration + 1) % 25 == 0:
            print(f"    Iter {iteration+1}: loss={current_loss:.4f} (best={best_loss:.4f})")

    return best_theta, losses


def evaluate_circuit(sampler, test_data, theta, circuit_builder):
    """Evaluate circuit on test pairs."""
    n_qubits = CONFIG['n_qubits_per_vector'] * 2

    circuits = []
    for v1, v2, _ in test_data:
        circuits.append(circuit_builder(v1, v2, theta))

    job = sampler.run(circuits, shots=CONFIG['shots'])
    result = job.result()

    preds = []
    targets = []
    for i, (v1, v2, target) in enumerate(test_data):
        counts = result[i].data.meas.get_counts()
        pred = compute_similarity_from_counts(counts, n_qubits)
        preds.append(pred)
        targets.append((target + 1) / 2)  # Normalize to [0, 1]

    preds = np.array(preds)
    targets = np.array(targets)

    if np.std(preds) < 1e-10 or np.std(targets) < 1e-10:
        return preds, targets, 0.0, 1.0

    correlation, p_value = stats.pearsonr(preds, targets)
    return preds, targets, correlation, p_value


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("QUANTUM LEARNING V2: Architecture That REQUIRES Learning")
    print("=" * 70)
    print(f"Qubits per vector: {CONFIG['n_qubits_per_vector']}")
    print(f"Total qubits: {CONFIG['n_qubits_per_vector'] * 2}")
    print(f"SPSA iterations: {CONFIG['spsa_iterations']}")
    print(f"Using contrastive loss: YES")
    print("=" * 70)

    np.random.seed(42)

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n[1] PREPARING DATA...")

    # Training pairs (animals - related concepts)
    train_concepts = [
        ('dog', 'puppy'),      # Very similar
        ('cat', 'kitten'),     # Very similar
        ('dog', 'cat'),        # Somewhat similar (both pets)
        ('bird', 'airplane'),  # Dissimilar (but both fly)
        ('fish', 'car'),       # Very dissimilar
        ('horse', 'zebra'),    # Similar
        ('lion', 'tiger'),     # Similar
        ('snake', 'rope'),     # Dissimilar
    ]

    # Test pairs (same domain - held out)
    test_same = [
        ('wolf', 'dog'),
        ('mouse', 'rat'),
        ('eagle', 'hawk'),
        ('shark', 'dolphin'),
        ('cow', 'bull'),
    ]

    # Test pairs (different domain)
    test_diff = [
        ('happy', 'sad'),
        ('computer', 'phone'),
        ('mountain', 'valley'),
        ('music', 'silence'),
        ('book', 'movie'),
    ]

    all_concepts = list(set(
        c for pair in (train_concepts + test_same + test_diff) for c in pair
    ))

    print(f"Total concepts: {len(all_concepts)}")
    print(f"Training pairs: {len(train_concepts)}")
    print(f"Test (same domain): {len(test_same)}")
    print(f"Test (diff domain): {len(test_diff)}")

    # Get embeddings
    print("\nLoading sentence-transformers...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_raw = model.encode(all_concepts, show_progress_bar=False)

    # PCA to n_qubits_per_vector dimensions
    n_dim = CONFIG['n_qubits_per_vector']
    print(f"Applying PCA: 384D → {n_dim}D...")
    pca = PCA(n_components=n_dim)
    embeddings_pca = pca.fit_transform(embeddings_raw)
    print(f"Variance explained: {sum(pca.explained_variance_ratio_):.3f}")

    # Scale for quantum encoding
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(embeddings_pca)

    # Create data tuples
    def make_data(pairs):
        data = []
        for c1, c2 in pairs:
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
            target = cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2])
            data.append((vectors[idx1], vectors[idx2], target))
        return data

    train_data = make_data(train_concepts)
    test_same_data = make_data(test_same)
    test_diff_data = make_data(test_diff)

    print(f"\nTraining targets: {[f'{d[2]:.2f}' for d in train_data]}")

    # =========================================================================
    # SETUP
    # =========================================================================
    print("\nInitializing AerSimulator...")
    backend = AerSimulator()
    sampler = SamplerV2(mode=backend)

    results = {}

    # =========================================================================
    # TEST 1: LEARNED COMPARISON (Entangled)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] LEARNED COMPARISON CIRCUIT (with entanglement)")
    print("=" * 70)

    theta_entangled, losses_ent = train_circuit(
        sampler, train_data, build_learned_comparison_circuit
    )

    print("\nEvaluating on SAME domain...")
    _, _, corr_same_ent, p_same = evaluate_circuit(
        sampler, test_same_data, theta_entangled, build_learned_comparison_circuit
    )
    print(f"  Correlation: {corr_same_ent:.4f} (p={p_same:.3e})")

    print("\nEvaluating on DIFFERENT domain...")
    _, _, corr_diff_ent, p_diff = evaluate_circuit(
        sampler, test_diff_data, theta_entangled, build_learned_comparison_circuit
    )
    print(f"  Correlation: {corr_diff_ent:.4f} (p={p_diff:.3e})")

    results['entangled'] = {
        'test_same': float(corr_same_ent),
        'test_diff': float(corr_diff_ent),
        'final_loss': float(losses_ent[-1]),
        'best_loss': float(min(losses_ent)),
    }

    # =========================================================================
    # TEST 2: PRODUCT CIRCUIT (No Entanglement)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] PRODUCT CIRCUIT (no entanglement) - Ablation")
    print("=" * 70)

    theta_product, losses_prod = train_circuit(
        sampler, train_data, build_product_comparison_circuit
    )

    print("\nEvaluating on SAME domain...")
    _, _, corr_same_prod, _ = evaluate_circuit(
        sampler, test_same_data, theta_product, build_product_comparison_circuit
    )
    print(f"  Product (no entanglement): {corr_same_prod:.4f}")
    print(f"  Entangled:                 {corr_same_ent:.4f}")

    entanglement_helps = corr_same_ent > corr_same_prod
    print(f"\n  Entanglement helps: {'YES ✅' if entanglement_helps else 'NO ❌'}")
    print(f"  Improvement: {corr_same_ent - corr_same_prod:+.4f}")

    results['product'] = {
        'test_same': float(corr_same_prod),
        'final_loss': float(losses_prod[-1]),
    }

    # =========================================================================
    # TEST 3: RANDOM vs TRAINED (Critical Learning Test)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 3] RANDOM vs TRAINED - Does training actually help?")
    print("=" * 70)

    n_params = CONFIG['n_qubits_per_vector'] * 2 * 3
    theta_random = np.random.uniform(-0.01, 0.01, n_params)

    print("\nEvaluating with RANDOM parameters...")
    _, _, corr_random, _ = evaluate_circuit(
        sampler, test_same_data, theta_random, build_learned_comparison_circuit
    )

    print(f"  Random params:  {corr_random:.4f}")
    print(f"  Trained params: {corr_same_ent:.4f}")

    training_helps = corr_same_ent > corr_random + 0.1  # Need meaningful improvement
    print(f"\n  Training helps: {'YES ✅' if training_helps else 'NO ❌'}")
    print(f"  Improvement: {corr_same_ent - corr_random:+.4f}")

    results['training_test'] = {
        'random_corr': float(corr_random),
        'trained_corr': float(corr_same_ent),
        'improvement': float(corr_same_ent - corr_random),
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUANTUM LEARNING V2 - SUMMARY")
    print("=" * 70)

    print(f"\n{'Test':<35} {'Result':<12} {'Verdict'}")
    print("-" * 65)
    print(f"{'Basic Learning (same domain)':<35} {corr_same_ent:+.4f}       {'✅ PASS' if corr_same_ent > 0.5 else '❌ FAIL'}")
    print(f"{'Generalization (diff domain)':<35} {corr_diff_ent:+.4f}       {'✅ PASS' if corr_diff_ent > 0.3 else '❌ FAIL'}")
    print(f"{'Entanglement Effect':<35} {corr_same_ent - corr_same_prod:+.4f}       {'✅ HELPS' if entanglement_helps else '❌ NO EFFECT'}")
    print(f"{'Training Effect':<35} {corr_same_ent - corr_random:+.4f}       {'✅ HELPS' if training_helps else '❌ NO EFFECT'}")

    # Final verdict
    learning_demonstrated = (
        corr_same_ent > 0.5 and
        training_helps and
        corr_same_ent > corr_random + 0.1
    )

    print("\n" + "=" * 70)
    if learning_demonstrated:
        print("✅ QUANTUM LEARNING DEMONSTRATED!")
        print("   - Training improves performance")
        print("   - Circuit learned meaningful comparison")
    else:
        print("❌ NO LEARNING DEMONSTRATED")
        if not training_helps:
            print("   - Training did not improve over random")
        if corr_same_ent <= 0.5:
            print("   - Correlation too low")
    print("=" * 70)

    # Save results
    output = {
        "config": CONFIG,
        "git_commit": get_git_info(),
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "learning_demonstrated": bool(learning_demonstrated),
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
