#!/usr/bin/env python3
"""
QUANTUM LEARNING V3: HARDWARE TRANSFER
=======================================

STRATEGY: "Train Locally, Run Globally" (Expert Recommendation)
---------------------------------------------------------------
1. Train in simulation where math is perfect
2. Freeze the best_theta weights
3. Run INFERENCE ONLY on real hardware

EXPECTED RESULTS:
- Simulation: r = 0.71 (from ablation study)
- Hardware (optimistic): r = 0.30 - 0.45
- Hardware (pessimistic): r = 0.0 - 0.10
- SUCCESS THRESHOLD: r > 0.15 (concept survived reality)

WHY V3?
- Strongest signal (+1.22 training effect)
- Sparse encoding (1 feature/qubit) = more noise-resistant
- Ancilla measurement = cleaner output signal
"""

import numpy as np
import json
import os
import subprocess
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
    "experiment_name": "quantum_learning_v3_hardware",
    "n_qubits_per_vector": 3,  # 3 qubits per vector
    "n_ancilla": 1,            # 1 ancilla qubit for output
    "shots": 4096,             # More shots for noise averaging
    "spsa_iterations": 200,    # For simulation training
    "spsa_lr": 0.15,

    # HARDWARE SETTINGS
    "use_hardware": True,     # SET TO True FOR REAL QUANTUM COMPUTER
    "backend_name": "ibm_fez", # 156 qubit Eagle r3 processor
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# V3 ARCHITECTURE: Ancilla-Based Learning
# ============================================================================

def build_ancilla_circuit(v1: np.ndarray, v2: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
    """
    ANCILLA-BASED CLASSIFICATION CIRCUIT

    Layout:
        Qubits 0,1,2: Encode v1
        Qubits 3,4,5: Encode v2
        Qubit 6:      ANCILLA (output) - starts as |0>

    The circuit must LEARN to flip the ancilla for similar pairs.
    With random theta, ancilla output is random.

    Parameters:
        - Layer 1: 7 qubits x 1 = 7 params (local rotations)
        - Layer 2: 7 qubits x 1 = 7 params (after entanglement)
        - Layer 3: 7 qubits x 1 = 7 params (before measurement)
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

    # Ancilla starts as |0> (no encoding)
    qc.barrier()

    # === TRAINABLE LAYER 1: Local rotations ===
    for i in range(n_total):
        qc.ry(float(theta[i]), i)

    qc.barrier()

    # === ENTANGLEMENT: Connect data qubits to ancilla ===
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


def get_ancilla_probability(counts: dict, n_total: int) -> float:
    """
    Extract P(ancilla = 1) from measurement counts.
    Ancilla is the LAST qubit (index n_total-1).
    In Qiskit's little-endian, it's the FIRST bit of the string.
    """
    total_shots = sum(counts.values())
    ancilla_one_count = 0

    for bitstring, count in counts.items():
        bs = bitstring.replace(' ', '').zfill(n_total)
        ancilla_bit = bs[0]  # Leftmost = highest index qubit
        if ancilla_bit == '1':
            ancilla_one_count += count

    return ancilla_one_count / total_shots


def binary_cross_entropy(pred: float, target: float, eps: float = 1e-7) -> float:
    """Binary cross-entropy loss."""
    pred = np.clip(pred, eps, 1 - eps)
    return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))


# ============================================================================
# TRAINING (SIMULATION ONLY)
# ============================================================================

def train_circuit(sampler, train_data, n_total):
    """
    Train circuit with SPSA optimizer.
    ONLY RUN IN SIMULATION MODE!
    """
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
            circuits.append(build_ancilla_circuit(v1, v2, theta_plus))
            circuits.append(build_ancilla_circuit(v1, v2, theta_minus))

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


def evaluate_circuit(sampler, test_data, theta, n_total, backend=None):
    """
    Evaluate circuit on test pairs.
    If backend is provided, transpile for hardware.
    """
    circuits = []
    for v1, v2, _ in test_data:
        qc = build_ancilla_circuit(v1, v2, theta)
        circuits.append(qc)

    # Transpile if running on hardware
    if backend is not None:
        print(f"  Transpiling {len(circuits)} circuits for {backend.name}...")
        circuits = transpile(circuits, backend=backend, optimization_level=1)

        # Report circuit depth
        depths = [c.depth() for c in circuits]
        print(f"  Circuit depths: min={min(depths)}, max={max(depths)}, avg={np.mean(depths):.0f}")

    # Run
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
        targets.append(target_sim)

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
    print("QUANTUM LEARNING V3: HARDWARE TRANSFER")
    print("=" * 70)
    print(f"Mode: {'HARDWARE' if CONFIG['use_hardware'] else 'SIMULATION'}")
    print(f"Data qubits per vector: {CONFIG['n_qubits_per_vector']}")
    print(f"Ancilla qubits: {CONFIG['n_ancilla']}")
    print(f"Total qubits: {CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']}")
    print("=" * 70)

    if CONFIG['use_hardware']:
        print("\n*** HARDWARE MODE ***")
        print("Strategy: 'Train Locally, Run Globally'")
        print("- Using pre-trained weights from simulation")
        print("- Running INFERENCE ONLY (no training)")
        print("- Expected: correlation to DROP due to noise")
        print("- Success threshold: r > 0.15")

    print("=" * 70)

    np.random.seed(42)

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n[1] PREPARING DATA...")

    # Training pairs (same as original V3)
    train_similar = [
        ('dog', 'puppy'), ('cat', 'kitten'), ('horse', 'pony'),
        ('wolf', 'dog'), ('lion', 'tiger'), ('mouse', 'rat'),
    ]

    train_dissimilar = [
        ('dog', 'car'), ('cat', 'mountain'), ('bird', 'computer'),
        ('fish', 'happiness'), ('horse', 'music'), ('lion', 'book'),
    ]

    # Test pairs (held out)
    test_pairs = [
        ('eagle', 'hawk'), ('shark', 'dolphin'), ('cow', 'bull'),
        ('happy', 'car'), ('tree', 'phone'), ('water', 'anger'),
        ('snake', 'lizard'), ('apple', 'rocket'),
    ]

    all_concepts = list(set(
        c for pair in (train_similar + train_dissimilar + test_pairs) for c in pair
    ))

    print(f"Total concepts: {len(all_concepts)}")

    # Get embeddings
    print("Loading sentence-transformers...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_raw = model.encode(all_concepts, show_progress_bar=False)

    # PCA
    n_dim = CONFIG['n_qubits_per_vector']
    print(f"Applying PCA: 384D -> {n_dim}D...")
    pca = PCA(n_components=n_dim)
    embeddings_pca = pca.fit_transform(embeddings_raw)
    print(f"Variance explained: {sum(pca.explained_variance_ratio_):.3f}")

    # Scale to [0.1, pi-0.1] (correct scaling)
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(embeddings_pca)

    # Create training data (similar=0, dissimilar=1)
    def make_train_data(similar_pairs, dissimilar_pairs):
        data = []
        for c1, c2 in similar_pairs:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            data.append((vectors[idx1], vectors[idx2], 0.0))
        for c1, c2 in dissimilar_pairs:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            data.append((vectors[idx1], vectors[idx2], 1.0))
        return data

    # Create test data (continuous similarity for correlation)
    def make_test_data(pairs):
        data = []
        for c1, c2 in pairs:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            sim = cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2])
            data.append((vectors[idx1], vectors[idx2], sim))
        return data

    train_data = make_train_data(train_similar, train_dissimilar)
    test_data = make_test_data(test_pairs)
    np.random.shuffle(train_data)

    print(f"Test similarities: {[f'{d[2]:.2f}' for d in test_data]}")

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

    # =========================================================================
    # LOAD OR TRAIN WEIGHTS
    # =========================================================================

    # Check if we have saved weights
    weights_file = os.path.join(os.path.dirname(__file__), 'results', 'v3_best_theta.json')

    if CONFIG['use_hardware']:
        # HARDWARE MODE: Load pre-trained weights
        print("\n[3] LOADING PRE-TRAINED WEIGHTS...")

        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                saved = json.load(f)
            theta_opt = np.array(saved['best_theta'])
            print(f"  Loaded {len(theta_opt)} parameters from {weights_file}")
            print(f"  Simulation correlation: {saved.get('simulation_correlation', 'N/A')}")
        else:
            print(f"  ERROR: No weights file found at {weights_file}")
            print("  Please run in SIMULATION mode first to train weights!")
            return

        # Verify circuit compiles
        print("\n[4] VERIFYING CIRCUIT...")
        dummy_circuit = build_ancilla_circuit(vectors[0], vectors[1], theta_opt)
        isa_circuit = transpile(dummy_circuit, backend=backend, optimization_level=1)
        print(f"  Hardware circuit depth: {isa_circuit.depth()}")
        print(f"  Gates: {isa_circuit.count_ops()}")

    else:
        # SIMULATION MODE: Train from scratch
        print("\n[3] TRAINING IN SIMULATION...")

        # Random baseline first
        print("\n[BASELINE] Testing random parameters...")
        theta_random = np.random.uniform(-np.pi, np.pi, n_params)
        preds_rand, targets_rand, corr_random, _ = evaluate_circuit(
            sampler, test_data, theta_random, n_total
        )
        print(f"  Random correlation: {corr_random:.4f}")

        # Train
        print("\n[TRAINING] SPSA optimization...")
        theta_opt, losses = train_circuit(sampler, train_data, n_total)

        # Evaluate trained
        print("\n[EVALUATION] Testing trained parameters...")
        preds_trained, targets_trained, corr_trained, p_trained = evaluate_circuit(
            sampler, test_data, theta_opt, n_total
        )

        training_effect = corr_trained - corr_random
        print(f"\n  Results:")
        print(f"    Random correlation:  {corr_random:.4f}")
        print(f"    Trained correlation: {corr_trained:.4f}")
        print(f"    Training effect:     {training_effect:+.4f}")

        # Save weights for hardware transfer
        print("\n[SAVING] Storing weights for hardware transfer...")
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)

        with open(weights_file, 'w') as f:
            json.dump({
                'best_theta': theta_opt.tolist(),
                'simulation_correlation': float(corr_trained),
                'random_correlation': float(corr_random),
                'training_effect': float(training_effect),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Saved to {weights_file}")

        # Also save full results
        output = {
            "config": CONFIG,
            "mode": "simulation",
            "git_commit": get_git_info(),
            "timestamp": datetime.now().isoformat(),
            "results": {
                "random_correlation": float(corr_random),
                "trained_correlation": float(corr_trained),
                "training_effect": float(training_effect),
                "final_loss": float(losses[-1]),
                "best_loss": float(min(losses)),
                "preds_trained": preds_trained.tolist(),
                "targets": targets_trained.tolist(),
            },
            "best_theta": theta_opt.tolist(),
        }

        filename = f"v3_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f"  Full results: {filepath}")

        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        print(f"  Trained correlation: {corr_trained:.4f}")
        print(f"  Weights saved for hardware transfer")
        print("\nTo run on real quantum hardware:")
        print("  1. Set CONFIG['use_hardware'] = True")
        print("  2. Run this script again")
        return

    # =========================================================================
    # HARDWARE INFERENCE
    # =========================================================================
    print("\n[5] RUNNING HARDWARE INFERENCE...")
    print("  This may take several minutes (queue + execution)...")

    # 5A: Random baseline on hardware (to prove LEARNING matters)
    print("\n[5A] RANDOM BASELINE ON HARDWARE...")
    theta_random = np.random.uniform(-np.pi, np.pi, n_params)
    preds_rand_hw, targets_rand_hw, corr_rand_hw, p_rand_hw = evaluate_circuit(
        sampler, test_data, theta_random, n_total, backend=backend
    )
    print(f"  Random (hardware): r = {corr_rand_hw:.4f}")

    # 5B: Trained weights on hardware
    print("\n[5B] TRAINED WEIGHTS ON HARDWARE...")
    preds_hw, targets_hw, corr_hw, p_hw = evaluate_circuit(
        sampler, test_data, theta_opt, n_total, backend=backend
    )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("HARDWARE TRANSFER RESULTS")
    print("=" * 70)

    # Load simulation baseline for comparison
    with open(weights_file, 'r') as f:
        saved = json.load(f)
    sim_corr = saved.get('simulation_correlation', 0)

    hw_learning_effect = corr_hw - corr_rand_hw

    print(f"\n{'Metric':<35} {'Value':<12} {'Verdict'}")
    print("-" * 60)
    print(f"{'Simulation correlation':<35} {sim_corr:.4f}       (baseline)")
    print(f"{'Hardware random correlation':<35} {corr_rand_hw:.4f}       (noise floor)")
    print(f"{'Hardware trained correlation':<35} {corr_hw:.4f}       {'SUCCESS!' if corr_hw > 0.15 else 'NOISE DOMINATED'}")
    print(f"{'Hardware learning effect':<35} {hw_learning_effect:+.4f}       {'LEARNING WORKS!' if hw_learning_effect > 0.1 else 'weak'}")
    print(f"{'Transfer gap (sim→hw)':<35} {sim_corr - corr_hw:.4f}       (expected)")
    print(f"{'P-value':<35} {p_hw:.4f}")

    # Quantum advantage evidence
    print(f"\n{'='*60}")
    print("QUANTUM ADVANTAGE EVIDENCE (Hardware)")
    print(f"{'='*60}")
    print(f"{'Learning Effect (HW)':<35} {hw_learning_effect:+.4f}       (target: >+0.10)")
    print(f"{'Correlation Survived':<35} {corr_hw:.4f}       (target: >0.15)")

    print(f"\n{'Test Pair':<25} {'Target':<10} {'HW Pred':<10}")
    print("-" * 50)
    for i, (v1, v2, target) in enumerate(test_data):
        pair_name = f"{test_pairs[i][0]}-{test_pairs[i][1]}"
        print(f"{pair_name:<25} {target:.3f}      {preds_hw[i]:.3f}")

    # Verdict
    print("\n" + "=" * 70)
    if corr_hw > 0.15:
        print("SUCCESS! Concept survived the noise wall!")
        print(f"  Hardware correlation: {corr_hw:.4f}")
        print(f"  This is a PUBLISHABLE RESULT!")
    elif corr_hw > 0.0:
        print("PARTIAL SUCCESS: Weak signal detected")
        print(f"  Hardware correlation: {corr_hw:.4f}")
        print("  Signal present but heavily degraded by noise")
    else:
        print("NOISE DOMINATED: No correlation survived")
        print(f"  Hardware correlation: {corr_hw:.4f}")
        print("  The noise wall was too strong")
    print("=" * 70)

    # Save hardware results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    output = {
        "config": CONFIG,
        "mode": "hardware",
        "backend": CONFIG['backend_name'],
        "git_commit": get_git_info(),
        "timestamp": datetime.now().isoformat(),
        "results": {
            "simulation_correlation": float(sim_corr),
            "hardware_random_correlation": float(corr_rand_hw),
            "hardware_trained_correlation": float(corr_hw),
            "hardware_learning_effect": float(hw_learning_effect),
            "transfer_gap": float(sim_corr - corr_hw),
            "p_value": float(p_hw),
            "preds_random_hardware": preds_rand_hw.tolist(),
            "preds_trained_hardware": preds_hw.tolist(),
            "targets": targets_hw.tolist(),
            "success_correlation": bool(corr_hw > 0.15),
            "success_learning": bool(hw_learning_effect > 0.1),
        },
        "quantum_advantage_evidence": {
            "learning_effect_hardware": float(hw_learning_effect),
            "correlation_survived": float(corr_hw),
            "learning_validated": bool(hw_learning_effect > 0.1),
            "transfer_validated": bool(corr_hw > 0.15),
        },
    }

    filename = f"v3_hardware_{CONFIG['backend_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
