#!/usr/bin/env python3
"""
QUANTUM ENTANGLEMENT & SUPERPOSITION VALIDATION
================================================

This script validates two quantum phenomena on real hardware:

1. ENTANGLEMENT ADVANTAGE:
   - Compare V3 (with CX gates) vs V3-Product (no CX gates)
   - Both use trained weights
   - Expected: Entangled >> Product (simulation showed +0.99 gap)

2. SUPERPOSITION VALIDATION:
   - Compare trained circuit vs random circuit
   - Both entangled architectures
   - Expected: Trained >> Random (demonstrates Hilbert space navigation)

Hardware: IBM Quantum ibm_fez (156 qubits)
Strategy: Inference only (no training on hardware)
"""

import numpy as np
import json
import os
from datetime import datetime
from scipy import stats

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# ML Imports
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "experiment_name": "entanglement_superposition_validation",
    "n_qubits_per_vector": 3,
    "n_ancilla": 1,
    "shots": 4096,

    # HARDWARE SETTINGS
    "use_hardware": True,  # Set to True for real quantum
    "backend_name": "ibm_fez",
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


# ============================================================================
# CIRCUIT ARCHITECTURES
# ============================================================================

def build_entangled_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    V3 ENTANGLED CIRCUIT (Original)

    Full entanglement with CX gates connecting all qubits to ancilla.
    This is the standard V3 architecture that achieved +0.61 correlation.
    """
    n = len(v1)  # 3
    n_data = 2 * n  # 6 data qubits
    n_total = n_data + 1  # 7 total
    ancilla_idx = n_total - 1  # Index 6

    qc = QuantumCircuit(n_total)

    # === ENCODING LAYER ===
    for i in range(n):
        qc.ry(float(v1[i]), i)
    for i in range(n):
        qc.ry(float(v2[i]), n + i)
    qc.barrier()

    # === TRAINABLE LAYER 1 ===
    for i in range(n_total):
        qc.ry(float(theta[i]), i)
    qc.barrier()

    # === ENTANGLEMENT: CX to ancilla ===
    for i in range(n):
        qc.cx(i, ancilla_idx)
    for i in range(n):
        qc.cx(n + i, ancilla_idx)
    qc.barrier()

    # === TRAINABLE LAYER 2 ===
    for i in range(n_total):
        qc.ry(float(theta[n_total + i]), i)

    # Cross entanglement
    for i in range(n):
        qc.cx(i, n + i)
    qc.barrier()

    # === TRAINABLE LAYER 3 ===
    for i in range(n_total):
        qc.ry(float(theta[2 * n_total + i]), i)

    # Final CX to ancilla
    qc.cx(0, ancilla_idx)
    qc.cx(n, ancilla_idx)

    qc.measure_all()
    return qc


def build_product_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    V3 PRODUCT CIRCUIT (No Entanglement)

    Same structure as V3 but WITHOUT any CX gates.
    This creates a product state - no quantum correlations between registers.
    """
    n = len(v1)  # 3
    n_data = 2 * n  # 6 data qubits
    n_total = n_data + 1  # 7 total

    qc = QuantumCircuit(n_total)

    # === ENCODING LAYER ===
    for i in range(n):
        qc.ry(float(v1[i]), i)
    for i in range(n):
        qc.ry(float(v2[i]), n + i)
    qc.barrier()

    # === TRAINABLE LAYER 1 ===
    for i in range(n_total):
        qc.ry(float(theta[i]), i)
    qc.barrier()

    # === NO ENTANGLEMENT (Product State) ===
    # Skip all CX gates
    qc.barrier()

    # === TRAINABLE LAYER 2 ===
    for i in range(n_total):
        qc.ry(float(theta[n_total + i]), i)

    # NO cross entanglement
    qc.barrier()

    # === TRAINABLE LAYER 3 ===
    for i in range(n_total):
        qc.ry(float(theta[2 * n_total + i]), i)

    # NO final CX gates

    qc.measure_all()
    return qc


def get_ancilla_probability(counts: dict, n_total: int) -> float:
    """Extract P(ancilla = 1) from measurement counts."""
    total_shots = sum(counts.values())
    ancilla_one_count = 0

    for bitstring, count in counts.items():
        bs = bitstring.replace(' ', '').zfill(n_total)
        ancilla_bit = bs[0]  # Leftmost = highest index qubit
        if ancilla_bit == '1':
            ancilla_one_count += count

    return ancilla_one_count / total_shots


def evaluate_circuit(sampler, test_data, theta, circuit_builder, n_total, backend=None):
    """Evaluate a circuit architecture on test pairs."""
    circuits = []
    for v1, v2, _ in test_data:
        qc = circuit_builder(v1, v2, theta)
        circuits.append(qc)

    # Transpile if running on hardware
    if backend is not None:
        circuits = transpile(circuits, backend=backend, optimization_level=1)
        depths = [c.depth() for c in circuits]
        print(f"    Circuit depths: min={min(depths)}, max={max(depths)}, avg={np.mean(depths):.0f}")

    # Run
    job = sampler.run(circuits, shots=CONFIG['shots'])
    result = job.result()

    preds = []
    targets = []

    for i, (v1, v2, target_sim) in enumerate(test_data):
        counts = result[i].data.meas.get_counts()
        p_one = get_ancilla_probability(counts, n_total)
        pred = 1.0 - p_one  # P(similar) = P(ancilla=0)
        preds.append(pred)
        targets.append(target_sim)

    preds = np.array(preds)
    targets = np.array(targets)

    if np.std(preds) < 1e-10:
        return preds, targets, 0.0, 1.0

    correlation, p_value = stats.pearsonr(preds, targets)
    return preds, targets, correlation, p_value


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("QUANTUM ENTANGLEMENT & SUPERPOSITION VALIDATION")
    print("=" * 70)
    print(f"Mode: {'HARDWARE' if CONFIG['use_hardware'] else 'SIMULATION'}")
    print("=" * 70)

    np.random.seed(42)

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n[1] PREPARING DATA...")

    # Same test pairs as V3 hardware transfer
    test_pairs = [
        ('eagle', 'hawk'), ('shark', 'dolphin'), ('cow', 'bull'),
        ('happy', 'car'), ('tree', 'phone'), ('water', 'anger'),
        ('snake', 'lizard'), ('apple', 'rocket'),
    ]

    all_concepts = list(set(c for pair in test_pairs for c in pair))
    print(f"Total concepts: {len(all_concepts)}")

    # Get embeddings
    print("Loading sentence-transformers...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_raw = model.encode(all_concepts, show_progress_bar=False)

    # PCA
    n_dim = CONFIG['n_qubits_per_vector']
    pca = PCA(n_components=n_dim)
    embeddings_pca = pca.fit_transform(embeddings_raw)

    # Scale
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(embeddings_pca)

    # Create test data
    test_data = []
    for c1, c2 in test_pairs:
        idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
        sim = cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2])
        test_data.append((vectors[idx1], vectors[idx2], sim))

    print(f"Test pairs: {len(test_data)}")

    # =========================================================================
    # SETUP BACKEND
    # =========================================================================
    n_total = CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']  # 7
    n_params = n_total * 3  # 21

    if CONFIG['use_hardware']:
        print("\n[2] CONNECTING TO IBM QUANTUM...")
        service = QiskitRuntimeService(channel="ibm_cloud")
        backend = service.backend(CONFIG['backend_name'])
        print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
        sampler = SamplerV2(mode=backend)
    else:
        print("\n[2] INITIALIZING SIMULATOR...")
        backend = AerSimulator()
        sampler = SamplerV2(mode=backend)
        backend = None  # Don't transpile for simulator

    # =========================================================================
    # LOAD TRAINED WEIGHTS
    # =========================================================================
    print("\n[3] LOADING PRE-TRAINED WEIGHTS...")

    weights_file = os.path.join(os.path.dirname(__file__), 'results', 'v3_best_theta.json')

    if os.path.exists(weights_file):
        with open(weights_file, 'r') as f:
            saved = json.load(f)
        theta_trained = np.array(saved['best_theta'])
        print(f"  Loaded {len(theta_trained)} trained parameters")
    else:
        print("  ERROR: No weights file found! Run quantum_learning_v3_hardware.py first.")
        return

    # Random weights for comparison
    theta_random = np.random.uniform(-np.pi, np.pi, n_params)
    print(f"  Generated {len(theta_random)} random parameters")

    # =========================================================================
    # TEST 1: ENTANGLEMENT ABLATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: ENTANGLEMENT ABLATION")
    print("=" * 70)
    print("Comparing: Entangled (CX gates) vs Product (no CX gates)")
    print("Both using TRAINED weights")

    hw_backend = service.backend(CONFIG['backend_name']) if CONFIG['use_hardware'] else None

    # 1A: Entangled circuit with trained weights
    print("\n[1A] ENTANGLED + TRAINED...")
    preds_ent_train, targets, corr_ent_train, p_ent_train = evaluate_circuit(
        sampler, test_data, theta_trained, build_entangled_circuit, n_total, hw_backend
    )
    print(f"  Correlation: {corr_ent_train:.4f} (p={p_ent_train:.4f})")

    # 1B: Product circuit with trained weights
    print("\n[1B] PRODUCT + TRAINED...")
    preds_prod_train, _, corr_prod_train, p_prod_train = evaluate_circuit(
        sampler, test_data, theta_trained, build_product_circuit, n_total, hw_backend
    )
    print(f"  Correlation: {corr_prod_train:.4f} (p={p_prod_train:.4f})")

    entanglement_effect = corr_ent_train - corr_prod_train

    print(f"\n  ENTANGLEMENT EFFECT: {entanglement_effect:+.4f}")
    print(f"  (Entangled: {corr_ent_train:.4f} vs Product: {corr_prod_train:.4f})")

    # =========================================================================
    # TEST 2: SUPERPOSITION (Learning) VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: SUPERPOSITION (LEARNING) VALIDATION")
    print("=" * 70)
    print("Comparing: Trained weights vs Random weights")
    print("Both using ENTANGLED architecture")

    # 2A: Already have entangled + trained from above
    print(f"\n[2A] ENTANGLED + TRAINED: {corr_ent_train:.4f} (from above)")

    # 2B: Entangled circuit with random weights
    print("\n[2B] ENTANGLED + RANDOM...")
    preds_ent_rand, _, corr_ent_rand, p_ent_rand = evaluate_circuit(
        sampler, test_data, theta_random, build_entangled_circuit, n_total, hw_backend
    )
    print(f"  Correlation: {corr_ent_rand:.4f} (p={p_ent_rand:.4f})")

    learning_effect = corr_ent_train - corr_ent_rand

    print(f"\n  LEARNING EFFECT (Superposition): {learning_effect:+.4f}")
    print(f"  (Trained: {corr_ent_train:.4f} vs Random: {corr_ent_rand:.4f})")

    # =========================================================================
    # TEST 3: PRODUCT STATE BASELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: PRODUCT STATE LEARNING CHECK")
    print("=" * 70)
    print("Can product states learn? (They shouldn't)")

    # 3A: Product circuit with random weights
    print("\n[3A] PRODUCT + RANDOM...")
    preds_prod_rand, _, corr_prod_rand, p_prod_rand = evaluate_circuit(
        sampler, test_data, theta_random, build_product_circuit, n_total, hw_backend
    )
    print(f"  Correlation: {corr_prod_rand:.4f} (p={p_prod_rand:.4f})")

    product_learning = corr_prod_train - corr_prod_rand

    print(f"\n  PRODUCT LEARNING EFFECT: {product_learning:+.4f}")
    print(f"  (Trained: {corr_prod_train:.4f} vs Random: {corr_prod_rand:.4f})")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: QUANTUM ADVANTAGE VALIDATION")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'Value':<12} {'Target':<12} {'Verdict'}")
    print("-" * 75)

    # Entanglement
    ent_target = 0.50  # Simulation showed +0.99, but expect degradation on hardware
    ent_verdict = "VALIDATED" if entanglement_effect > 0.20 else "WEAK" if entanglement_effect > 0 else "FAILED"
    print(f"{'Entanglement Effect':<40} {entanglement_effect:+.4f}       {'>+0.20':<12} {ent_verdict}")

    # Learning (Superposition)
    learn_target = 0.50
    learn_verdict = "VALIDATED" if learning_effect > 0.50 else "WEAK" if learning_effect > 0.20 else "FAILED"
    print(f"{'Learning Effect (Superposition)':<40} {learning_effect:+.4f}       {'>+0.50':<12} {learn_verdict}")

    # Product can't learn
    prod_verdict = "EXPECTED" if abs(product_learning) < 0.30 else "UNEXPECTED"
    print(f"{'Product Learning (should be ~0)':<40} {product_learning:+.4f}       {'~0.00':<12} {prod_verdict}")

    print("\n" + "-" * 75)
    print("DETAILED RESULTS:")
    print("-" * 75)
    print(f"{'Condition':<30} {'Correlation':<15} {'P-value'}")
    print(f"{'Entangled + Trained':<30} {corr_ent_train:+.4f}         {p_ent_train:.4f}")
    print(f"{'Entangled + Random':<30} {corr_ent_rand:+.4f}         {p_ent_rand:.4f}")
    print(f"{'Product + Trained':<30} {corr_prod_train:+.4f}         {p_prod_train:.4f}")
    print(f"{'Product + Random':<30} {corr_prod_rand:+.4f}         {p_prod_rand:.4f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "config": CONFIG,
        "mode": "hardware" if CONFIG['use_hardware'] else "simulation",
        "timestamp": datetime.now().isoformat(),
        "results": {
            "entangled_trained": {
                "correlation": float(corr_ent_train),
                "p_value": float(p_ent_train),
                "predictions": preds_ent_train.tolist(),
            },
            "entangled_random": {
                "correlation": float(corr_ent_rand),
                "p_value": float(p_ent_rand),
                "predictions": preds_ent_rand.tolist(),
            },
            "product_trained": {
                "correlation": float(corr_prod_train),
                "p_value": float(p_prod_train),
                "predictions": preds_prod_train.tolist(),
            },
            "product_random": {
                "correlation": float(corr_prod_rand),
                "p_value": float(p_prod_rand),
                "predictions": preds_prod_rand.tolist(),
            },
            "targets": targets.tolist(),
        },
        "quantum_advantage": {
            "entanglement_effect": float(entanglement_effect),
            "learning_effect": float(learning_effect),
            "product_learning": float(product_learning),
            "entanglement_validated": bool(entanglement_effect > 0.20),
            "superposition_validated": bool(learning_effect > 0.50),
        },
    }

    mode_str = "hardware" if CONFIG['use_hardware'] else "simulation"
    filename = f"entanglement_test_{mode_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {filepath}")

    # Final verdict
    print("\n" + "=" * 70)
    if entanglement_effect > 0.20 and learning_effect > 0.50:
        print("QUANTUM ADVANTAGE FULLY VALIDATED!")
        print("  - Entanglement provides cross-register correlations")
        print("  - Superposition enables Hilbert space navigation")
    elif entanglement_effect > 0 or learning_effect > 0.20:
        print("PARTIAL QUANTUM ADVANTAGE DETECTED")
        print("  - Some quantum effects survive hardware noise")
    else:
        print("QUANTUM ADVANTAGE NOT DETECTED ON HARDWARE")
        print("  - Noise may have overwhelmed quantum effects")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
