#!/usr/bin/env python3
"""
V3 SCALING EXPERIMENT: More Training Pairs + More Qubits
=========================================================

This experiment tests if V3 generalizes better with:
1. More training pairs (50 instead of 12)
2. More qubits (11 instead of 7)

Strategy: Train locally (simulator), Run globally (hardware)

Expected: Generalization improves from r=0.08 to r=0.45-0.60
"""

import numpy as np
import json
import os
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    "experiment_name": "v3_scaling_50pairs",
    "n_qubits_per_vector": 5,  # SCALED UP from 3 to 5
    "n_ancilla": 1,
    "shots": 4096,
    "spsa_iterations": 300,  # More iterations for more params
    "spsa_lr": 0.12,

    # HARDWARE SETTINGS
    "use_hardware": False,  # Start with simulation for training
    "backend_name": "ibm_fez",
}

# ============================================================================
# EXPANDED TRAINING DATA (50 pairs instead of 12)
# ============================================================================
TRAIN_SIMILAR = [
    # Animals
    ('dog', 'puppy'), ('cat', 'kitten'), ('horse', 'pony'), ('wolf', 'dog'),
    ('lion', 'tiger'), ('mouse', 'rat'), ('bird', 'sparrow'), ('fish', 'salmon'),
    ('bear', 'grizzly'), ('deer', 'elk'), ('rabbit', 'hare'), ('pig', 'boar'),
    ('whale', 'dolphin'), ('chicken', 'hen'), ('frog', 'toad'),
    # Objects
    ('car', 'automobile'), ('phone', 'telephone'), ('computer', 'laptop'),
    ('chair', 'seat'), ('table', 'desk'), ('cup', 'mug'), ('knife', 'blade'),
    # Abstract
    ('happy', 'joy'), ('sad', 'sorrow'), ('love', 'affection'),
]

TRAIN_DISSIMILAR = [
    # Animal-Object
    ('dog', 'car'), ('cat', 'mountain'), ('bird', 'computer'), ('fish', 'happiness'),
    ('horse', 'music'), ('lion', 'book'), ('eagle', 'chair'), ('shark', 'flower'),
    ('bear', 'telephone'), ('wolf', 'painting'), ('mouse', 'ocean'),
    # Object-Abstract
    ('car', 'happiness'), ('phone', 'tree'), ('computer', 'river'),
    ('chair', 'emotion'), ('table', 'dream'), ('book', 'wind'),
    # Nature-Object
    ('tree', 'phone'), ('water', 'anger'), ('mountain', 'keyboard'),
    ('forest', 'hammer'), ('river', 'sadness'), ('ocean', 'pencil'),
    ('sun', 'chair'), ('moon', 'car'),
]

# Test set (same 83 pairs for fair comparison)
TEST_PAIRS = [
    # Animals - Similar (not in training)
    ('monkey', 'ape'), ('snake', 'lizard'), ('eagle', 'hawk'), ('shark', 'dolphin'),
    ('cow', 'bull'),
    # Animals - Dissimilar (not in training)
    ('snake', 'cloud'), ('monkey', 'table'), ('cow', 'music'),
    # Objects - Similar (not in training)
    ('door', 'gate'), ('window', 'glass'), ('bed', 'mattress'), ('lamp', 'light'),
    ('key', 'lock'), ('pen', 'pencil'), ('bag', 'sack'),
    # Objects - Dissimilar (not in training)
    ('door', 'emotion'), ('window', 'anger'), ('bed', 'river'),
    # Nature - Similar (not in training)
    ('tree', 'plant'), ('river', 'stream'), ('mountain', 'hill'), ('ocean', 'sea'),
    ('forest', 'woods'), ('flower', 'rose'), ('sun', 'star'),
    # Nature - Dissimilar (not in training)
    ('flower', 'hammer'), ('sun', 'sadness'), ('forest', 'telephone'),
    # Abstract - Similar (not in training)
    ('anger', 'rage'), ('fear', 'terror'), ('hope', 'optimism'), ('peace', 'calm'),
    # Abstract - Dissimilar (not in training)
    ('fear', 'pencil'), ('hope', 'rock'), ('peace', 'banana'),
    # Cross-category
    ('apple', 'rocket'), ('sky', 'hammer'), ('rain', 'keyboard'), ('snow', 'emotion'),
    ('fire', 'sadness'), ('water', 'book'), ('air', 'table'),
]

# Original 8 test pairs for validation comparison
ORIGINAL_TEST_PAIRS = [
    ('eagle', 'hawk'), ('shark', 'dolphin'), ('cow', 'bull'),
    ('happy', 'car'), ('tree', 'phone'), ('water', 'anger'),
    ('snake', 'lizard'), ('apple', 'rocket'),
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cosine_similarity(v1, v2):
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
# V3 SCALED ARCHITECTURE (11 qubits: 5+5+1)
# ============================================================================

def build_v3_scaled_circuit(v1, v2, theta):
    """
    V3 Scaled: 5 qubits per vector + 1 ancilla = 11 total
    Parameters: 11 * 3 = 33 parameters
    """
    n = len(v1)  # 5
    n_data = 2 * n  # 10
    n_total = n_data + 1  # 11
    ancilla_idx = n_total - 1  # 10

    qc = QuantumCircuit(n_total)

    # Encoding
    for i in range(n):
        qc.ry(float(v1[i]), i)
    for i in range(n):
        qc.ry(float(v2[i]), n + i)
    qc.barrier()

    # Layer 1
    for i in range(n_total):
        qc.ry(float(theta[i]), i)
    qc.barrier()

    # Entanglement to ancilla
    for i in range(n):
        qc.cx(i, ancilla_idx)
    for i in range(n):
        qc.cx(n + i, ancilla_idx)
    qc.barrier()

    # Layer 2
    for i in range(n_total):
        qc.ry(float(theta[n_total + i]), i)
    for i in range(n):
        qc.cx(i, n + i)
    qc.barrier()

    # Layer 3
    for i in range(n_total):
        qc.ry(float(theta[2 * n_total + i]), i)
    qc.cx(0, ancilla_idx)
    qc.cx(n, ancilla_idx)

    qc.measure_all()
    return qc


def get_ancilla_probability(counts, n_total):
    total_shots = sum(counts.values())
    ancilla_one_count = 0
    for bitstring, count in counts.items():
        bs = bitstring.replace(' ', '').zfill(n_total)
        if bs[0] == '1':
            ancilla_one_count += count
    return ancilla_one_count / total_shots


def binary_cross_entropy(pred, target, eps=1e-7):
    pred = np.clip(pred, eps, 1 - eps)
    return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))


# ============================================================================
# TRAINING (Simulator only)
# ============================================================================

def train_v3_scaled(sampler, train_data, n_total):
    """Train with SPSA on expanded dataset."""
    n_params = n_total * 3  # 33 parameters

    theta = np.random.uniform(-0.01, 0.01, n_params)
    losses = []
    best_theta = theta.copy()
    best_loss = float('inf')

    print(f"  Training ({CONFIG['spsa_iterations']} iterations, {n_params} parameters, {len(train_data)} pairs)...")

    for iteration in range(CONFIG['spsa_iterations']):
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        c_k = 0.1 / (iteration + 1) ** 0.101
        a_k = CONFIG['spsa_lr'] / (iteration + 1) ** 0.602

        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        # Sample a mini-batch (for efficiency with 50 pairs)
        batch_size = min(16, len(train_data))
        batch_indices = np.random.choice(len(train_data), batch_size, replace=False)
        batch = [train_data[i] for i in batch_indices]

        circuits = []
        for v1, v2, _ in batch:
            circuits.append(build_v3_scaled_circuit(v1, v2, theta_plus))
            circuits.append(build_v3_scaled_circuit(v1, v2, theta_minus))

        job = sampler.run(circuits, shots=CONFIG['shots'])
        result = job.result()

        loss_plus = 0.0
        loss_minus = 0.0

        for i, (v1, v2, target) in enumerate(batch):
            counts_plus = result[2*i].data.meas.get_counts()
            counts_minus = result[2*i + 1].data.meas.get_counts()

            pred_plus = get_ancilla_probability(counts_plus, n_total)
            pred_minus = get_ancilla_probability(counts_minus, n_total)

            loss_plus += binary_cross_entropy(pred_plus, target)
            loss_minus += binary_cross_entropy(pred_minus, target)

        loss_plus /= len(batch)
        loss_minus /= len(batch)

        gradient = (loss_plus - loss_minus) / (2 * c_k) * delta
        theta = theta - a_k * gradient

        current_loss = (loss_plus + loss_minus) / 2
        losses.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta.copy()

        if (iteration + 1) % 50 == 0:
            print(f"    Iter {iteration+1}: loss={current_loss:.4f} (best={best_loss:.4f})")

    return best_theta, losses


def evaluate_v3_scaled(sampler, test_data, theta, n_total, backend=None):
    """Evaluate on test set."""
    circuits = []
    for v1, v2, _ in test_data:
        qc = build_v3_scaled_circuit(v1, v2, theta)
        circuits.append(qc)

    if backend is not None:
        print(f"  Transpiling {len(circuits)} circuits for {backend.name}...")
        circuits = transpile(circuits, backend=backend, optimization_level=1)
        depths = [c.depth() for c in circuits]
        print(f"  Circuit depths: min={min(depths)}, max={max(depths)}, avg={np.mean(depths):.0f}")

    # Run in batches
    batch_size = 20
    all_preds = []

    for i in range(0, len(circuits), batch_size):
        batch = circuits[i:i+batch_size]
        job = sampler.run(batch, shots=CONFIG['shots'])
        result = job.result()

        for j in range(len(batch)):
            counts = result[j].data.meas.get_counts()
            p_one = get_ancilla_probability(counts, n_total)
            pred = 1.0 - p_one
            all_preds.append(pred)

    preds = np.array(all_preds)
    targets = np.array([d[2] for d in test_data])

    if np.std(preds) < 1e-10:
        return preds, targets, 0.0, 1.0

    corr, p_val = stats.pearsonr(preds, targets)
    return preds, targets, corr, p_val


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("V3 SCALING EXPERIMENT: 50 Training Pairs + 11 Qubits")
    print("=" * 70)
    print(f"Mode: {'HARDWARE' if CONFIG['use_hardware'] else 'SIMULATION (Training)'}")
    print(f"Qubits per vector: {CONFIG['n_qubits_per_vector']}")
    print(f"Total qubits: {CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']}")
    print(f"Training pairs: {len(TRAIN_SIMILAR) + len(TRAIN_DISSIMILAR)}")
    print("=" * 70)

    np.random.seed(42)

    # Data preparation
    print("\n[1] PREPARING DATA...")

    all_train_pairs = TRAIN_SIMILAR + TRAIN_DISSIMILAR
    all_test_pairs = TEST_PAIRS + ORIGINAL_TEST_PAIRS
    all_concepts = list(set(
        c for pair in (all_train_pairs + all_test_pairs) for c in pair
    ))
    print(f"Total concepts: {len(all_concepts)}")

    print("Loading sentence-transformers...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_raw = model.encode(all_concepts, show_progress_bar=False)

    # PCA
    n_dim = CONFIG['n_qubits_per_vector']
    pca = PCA(n_components=n_dim)
    embeddings_pca = pca.fit_transform(embeddings_raw)
    print(f"PCA: 384D -> {n_dim}D (variance: {sum(pca.explained_variance_ratio_):.3f})")

    # Scale
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(embeddings_pca)

    # Create training data
    train_data = []
    for c1, c2 in TRAIN_SIMILAR:
        if c1 in all_concepts and c2 in all_concepts:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            train_data.append((vectors[idx1], vectors[idx2], 0.0))  # Similar = 0
    for c1, c2 in TRAIN_DISSIMILAR:
        if c1 in all_concepts and c2 in all_concepts:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            train_data.append((vectors[idx1], vectors[idx2], 1.0))  # Dissimilar = 1

    np.random.shuffle(train_data)
    print(f"Training pairs: {len(train_data)}")

    # Create test data
    test_data = []
    valid_test_pairs = []
    for c1, c2 in all_test_pairs:
        if c1 in all_concepts and c2 in all_concepts:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            sim = cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2])
            test_data.append((vectors[idx1], vectors[idx2], sim))
            valid_test_pairs.append((c1, c2))
    print(f"Test pairs: {len(test_data)}")

    # Setup
    n_total = CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']
    n_params = n_total * 3

    # Check for saved weights
    weights_file = os.path.join(os.path.dirname(__file__), 'results', 'v3_scaled_best_theta.json')

    if CONFIG['use_hardware']:
        # Hardware inference mode
        print("\n[2] CONNECTING TO IBM QUANTUM...")
        service = QiskitRuntimeService(channel="ibm_cloud")
        backend = service.backend(CONFIG['backend_name'])
        print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
        sampler = SamplerV2(mode=backend)

        if not os.path.exists(weights_file):
            print("ERROR: No trained weights! Run in simulation mode first.")
            return

        with open(weights_file, 'r') as f:
            saved = json.load(f)
        theta_opt = np.array(saved['best_theta'])
        print(f"Loaded {len(theta_opt)} parameters")

        print("\n[3] RUNNING HARDWARE INFERENCE...")
        preds, targets, corr, p_val = evaluate_v3_scaled(
            sampler, test_data, theta_opt, n_total, backend
        )

    else:
        # Simulation training mode
        print("\n[2] TRAINING IN SIMULATION...")
        backend = AerSimulator()
        sampler = SamplerV2(mode=backend)

        # Random baseline
        print("\n[BASELINE] Random parameters...")
        theta_random = np.random.uniform(-np.pi, np.pi, n_params)
        _, _, corr_random, _ = evaluate_v3_scaled(sampler, test_data, theta_random, n_total)
        print(f"  Random correlation: {corr_random:.4f}")

        # Train
        print("\n[TRAINING]...")
        theta_opt, losses = train_v3_scaled(sampler, train_data, n_total)

        # Evaluate
        print("\n[EVALUATION]...")
        preds, targets, corr, p_val = evaluate_v3_scaled(sampler, test_data, theta_opt, n_total)

        training_effect = corr - corr_random
        print(f"\n  Random: {corr_random:.4f}")
        print(f"  Trained: {corr:.4f}")
        print(f"  Training effect: {training_effect:+.4f}")

        # Save weights
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)

        with open(weights_file, 'w') as f:
            json.dump({
                'best_theta': theta_opt.tolist(),
                'n_qubits_per_vector': CONFIG['n_qubits_per_vector'],
                'n_training_pairs': len(train_data),
                'simulation_correlation': float(corr),
                'random_correlation': float(corr_random),
                'training_effect': float(training_effect),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        print(f"\n  Weights saved to {weights_file}")

    # Results
    print("\n" + "=" * 70)
    print("V3 SCALING RESULTS")
    print("=" * 70)
    print(f"Test pairs: {len(test_data)}")
    print(f"Correlation: {corr:.4f} (p={p_val:.4f})")

    # Compare to original V3
    print("\n[SCALING COMPARISON]")
    print(f"{'Version':<20} {'Train Pairs':<12} {'Qubits':<8} {'Params':<8} {'Generalization r'}")
    print("-" * 70)
    print(f"{'Original V3':<20} {'12':<12} {'7':<8} {'21':<8} {'0.08 (hardware)'}")
    print(f"{'V3-Scaled':<20} {len(train_data):<12} {n_total:<8} {n_params:<8} {corr:.2f} ({'hardware' if CONFIG['use_hardware'] else 'simulation'})")

    # Save results
    output = {
        "config": CONFIG,
        "mode": "hardware" if CONFIG['use_hardware'] else "simulation",
        "timestamp": datetime.now().isoformat(),
        "scaling": {
            "n_training_pairs": len(train_data),
            "n_qubits": n_total,
            "n_params": n_params,
        },
        "results": {
            "correlation": float(corr),
            "p_value": float(p_val),
            "predictions": preds.tolist(),
            "targets": targets.tolist(),
        },
    }

    mode_str = "hardware" if CONFIG['use_hardware'] else "simulation"
    filename = f"v3_scaling_{mode_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(os.path.dirname(__file__), 'results', filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
