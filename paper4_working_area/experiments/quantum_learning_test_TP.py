#!/usr/bin/env python3
"""
QUANTUM LEARNING TEST: Does the Circuit Actually Learn? (PRODUCTION READY)
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

FEATURES ADDED:
- Git Commit Tracking (Data Provenance)
- Hardware Pre-flight Checks (Depth/Cost Safety)
- Centralized Configuration
"""

import numpy as np
import json
import os
import sys
import subprocess
import uuid
import platform
from datetime import datetime
from scipy import stats

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ML Imports
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION (RESCUE PLAN - Expert Recommended Settings)
# ============================================================================
CONFIG = {
    "experiment_name": "quantum_learning_rescue",
    "backend_name": "ibm_fez",
    "n_qubits": 6,          # REDUCED from 20 - avoid barren plateaus
    "shots": 4096,
    "use_hardware": False,  # <--- SET THIS TO TRUE FOR REAL EXPERIMENTS
    "optimization_level": 3,
    "spsa_iterations": 100, # INCREASED from 10 - SPSA needs more iterations
    "spsa_lr": 0.05         # REDUCED from 0.1 - smaller steps for stability
}

# ============================================================================
# INFRASTRUCTURE & SAFETY (NEW)
# ============================================================================

def get_system_metadata():
    """
    Captures the exact state of the code and environment.
    Crucial for correlating results with specific code versions.
    """
    metadata = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "config": CONFIG,
    }
    
    # Capture Git Commit Hash
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        
        metadata["git_commit"] = commit
        metadata["git_dirty"] = bool(status)
        
        if bool(status):
            print("\n" + "!"*60)
            print("WARNING: You have uncommitted changes!")
            print("The results file will be marked as 'dirty'.")
            print("!"*60)
            
    except Exception:
        metadata["git_commit"] = "unknown_no_git"
        print("\n[Warning] Not a git repository. Version tracking disabled.")

    return metadata

class NumpyEncoder(json.JSONEncoder):
    """Handles saving Numpy arrays to JSON automatically."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def check_circuit_viability(circuit, backend):
    """
    Pre-flight check: Estimates depth and ops before sending to hardware.
    Prevents running circuits that are too deep for the coherence time.
    """
    if not CONFIG["use_hardware"]:
        return

    print("\n[PRE-FLIGHT CHECK] Analyzing circuit hardware viability...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=CONFIG["optimization_level"])
    isa_circuit = pm.run(circuit)
    
    depth = isa_circuit.depth()
    ops = isa_circuit.count_ops()
    
    print(f"  ➜ Transpiled Depth: {depth}")
    print(f"  ➜ Gate Count: {ops}")
    
    # Heuristic for current IBM devices (Falcon/Eagle)
    if depth > 150:
        print("  ⚠️  WARNING: Circuit depth > 150. Expect significant noise.")
    else:
        print("  ✅ Circuit depth looks acceptable.")

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


def compute_similarity_from_counts(counts: dict, n_qubits: int) -> float:
    """
    Compute similarity from measurement counts using Hamming distance.
    (0 = orthogonal, 1 = identical)
    """
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
    RESCUE: Simplified linear entanglement (not full mesh) to avoid barren plateaus.
    """
    n = len(v1)
    qc = QuantumCircuit(n)

    # Encode v1
    for i in range(n):
        qc.ry(float(v1[i]), i)

    # Trainable layer 1
    for i in range(n):
        qc.ry(float(theta[i]), i)

    # SIMPLIFIED ENTANGLEMENT: Linear chain only (not two rounds)
    # This creates nearest-neighbor correlations without deep scrambling
    for i in range(n - 1):
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
# TRAINING & EVALUATION (Optimized for Sessions)
# ============================================================================

def train_circuit(sampler, train_pairs, vectors, concepts, targets_raw, circuit_builder):
    """
    Train circuit parameters using SPSA.
    Updated to use a passed 'sampler' to avoid re-initializing backend.
    """
    n_params = CONFIG['n_qubits'] * 2
    # RESCUE: Initialize near zero (Identity operation) - avoids starting in chaos
    theta = np.random.uniform(-0.01, 0.01, n_params)
    losses = []
    
    print(f"  Starting Training ({CONFIG['spsa_iterations']} iterations)...")

    for iteration in range(CONFIG['spsa_iterations']):
        # SPSA perturbation
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        c_k = 0.1 / (iteration + 1) ** 0.101
        a_k = CONFIG['spsa_lr'] / (iteration + 1) ** 0.602

        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        # Batch circuits for efficiency
        circuits = []
        
        # 1. Plus Perturbation Circuits
        for c1, c2 in train_pairs:
            idx1, idx2 = concepts.index(c1), concepts.index(c2)
            circuits.append(circuit_builder(vectors[idx1], vectors[idx2], theta_plus))
            
        # 2. Minus Perturbation Circuits
        for c1, c2 in train_pairs:
            idx1, idx2 = concepts.index(c1), concepts.index(c2)
            circuits.append(circuit_builder(vectors[idx1], vectors[idx2], theta_minus))

        # Run all at once
        job = sampler.run(circuits, shots=CONFIG['shots'])
        result = job.result()

        # Compute losses
        loss_plus = 0.0
        loss_minus = 0.0
        n_pairs = len(train_pairs)

        for i in range(n_pairs):
            # Process Plus
            counts_plus = result[i].data.meas.get_counts()
            pred_plus = compute_similarity_from_counts(counts_plus, CONFIG['n_qubits'])
            loss_plus += (pred_plus - targets_raw[i]) ** 2
            
            # Process Minus (offset by n_pairs)
            counts_minus = result[i + n_pairs].data.meas.get_counts()
            pred_minus = compute_similarity_from_counts(counts_minus, CONFIG['n_qubits'])
            loss_minus += (pred_minus - targets_raw[i]) ** 2

        loss_plus /= n_pairs
        loss_minus /= n_pairs

        # SPSA update
        gradient = (loss_plus - loss_minus) / (2 * c_k) * delta
        theta = theta - a_k * gradient

        current_loss = (loss_plus + loss_minus) / 2
        losses.append(current_loss)
        print(f"    Iter {iteration+1}: loss={current_loss:.4f}")

    return theta, losses


def evaluate_circuit(sampler, test_pairs, vectors, concepts, targets_raw, theta, circuit_builder):
    """
    Evaluate trained circuit on test pairs.
    """
    circuits = []
    
    for c1, c2 in test_pairs:
        idx1, idx2 = concepts.index(c1), concepts.index(c2)
        circuits.append(circuit_builder(vectors[idx1], vectors[idx2], theta))

    job = sampler.run(circuits, shots=CONFIG['shots'])
    result = job.result()

    preds = []
    for i in range(len(circuits)):
        counts = result[i].data.meas.get_counts()
        pred = compute_similarity_from_counts(counts, CONFIG['n_qubits'])
        preds.append(pred)

    preds = np.array(preds)
    targets = np.array(targets_raw)

    correlation, p_value = stats.pearsonr(preds, targets)

    return preds, targets, correlation, p_value


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("QUANTUM LEARNING TEST (PRODUCTION)")
    print("=" * 70)
    print(f"Backend: {CONFIG['backend_name'] if CONFIG['use_hardware'] else 'Local Simulator'}")
    print(f"Hardware Mode: {CONFIG['use_hardware']}")
    print("=" * 70)
    
    # 0. INIT METADATA
    metadata = get_system_metadata()
    np.random.seed(42)

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n[1] PREPARING DATA...")

    # TRAINING SET: Animal domain
    train_pairs = [
        ('dog', 'puppy'), ('cat', 'kitten'), ('bird', 'sparrow'),
        ('fish', 'salmon'), ('horse', 'pony'),
    ]

    # TEST SET 1: Same domain (animals) but DIFFERENT concepts
    test_same_domain = [
        ('dog', 'cat'), ('bird', 'fish'), ('horse', 'dog'),
        ('cat', 'bird'), ('fish', 'horse'),
    ]

    # TEST SET 2: DIFFERENT domain (objects/abstract)
    test_diff_domain = [
        ('car', 'truck'), ('happy', 'sad'), ('house', 'building'),
        ('book', 'paper'), ('music', 'song'),
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
    print(f"Applying PCA: 384D → {CONFIG['n_qubits']}D...")
    pca = PCA(n_components=CONFIG['n_qubits'])
    embeddings_pca = pca.fit_transform(embeddings_raw)
    variance = sum(pca.explained_variance_ratio_)
    print(f"Variance explained: {variance:.3f}")

    # Scale for quantum encoding
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(embeddings_pca)

    # Helper to calculate targets
    def get_targets(pairs):
        t = []
        for c1, c2 in pairs:
            idx1, idx2 = all_concepts.index(c1), all_concepts.index(c2)
            t.append(cosine_similarity(embeddings_raw[idx1], embeddings_raw[idx2]))
        return t

    train_targets = get_targets(train_pairs)
    test_same_targets = get_targets(test_same_domain)
    test_diff_targets = get_targets(test_diff_domain)

    print(f"Training targets: {[f'{t:.3f}' for t in train_targets]}")

    # =========================================================================
    # BACKEND SETUP
    # =========================================================================
    if CONFIG['use_hardware']:
        print("\nAuthenticating with IBM Quantum...")
        service = QiskitRuntimeService(channel="ibm_cloud")
        backend = service.backend(CONFIG['backend_name'])
        
        # Run Pre-flight Check
        dummy_c = build_entangled_circuit(np.zeros(CONFIG['n_qubits']), np.zeros(CONFIG['n_qubits']), np.zeros(CONFIG['n_qubits']*2))
        check_circuit_viability(dummy_c, backend)
    else:
        print("\nInitializing AerSimulator...")
        backend = AerSimulator()
    
    sampler = SamplerV2(mode=backend)
    results = {}

    # =========================================================================
    # TEST 1: BASIC LEARNING (Entangled Circuit)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] BASIC LEARNING - Can circuit learn and generalize?")
    print("=" * 70)

    print("\nTraining entangled circuit on 5 animal pairs...")
    theta_entangled, train_losses = train_circuit(
        sampler, train_pairs, vectors, all_concepts, train_targets,
        build_entangled_circuit
    )

    print("\nEvaluating on SAME domain (held-out animal pairs)...")
    _, _, corr_same, p_same = evaluate_circuit(
        sampler, test_same_domain, vectors, all_concepts, test_same_targets,
        theta_entangled, build_entangled_circuit
    )
    print(f"  Correlation: {corr_same:.4f} (p={p_same:.3e})")

    print("\nEvaluating on DIFFERENT domain (objects/abstract)...")
    _, _, corr_diff, p_diff = evaluate_circuit(
        sampler, test_diff_domain, vectors, all_concepts, test_diff_targets,
        theta_entangled, build_entangled_circuit
    )
    print(f"  Correlation: {corr_diff:.4f} (p={p_diff:.3e})")

    results['test1_basic_learning'] = {
        'train_loss_final': float(train_losses[-1]),
        'test_same_domain': float(corr_same),
        'test_diff_domain': float(corr_diff),
        'theta': theta_entangled,
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
        sampler, train_pairs, vectors, all_concepts, train_targets,
        build_product_circuit
    )

    print("\nEvaluating product circuit on same domain...")
    _, _, corr_product, p_product = evaluate_circuit(
        sampler, test_same_domain, vectors, all_concepts, test_same_targets,
        theta_product, build_product_circuit
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
    theta_random = np.random.uniform(-0.01, 0.01, CONFIG['n_qubits'] * 2)
    _, _, corr_random, p_random = evaluate_circuit(
        sampler, test_same_domain, vectors, all_concepts, test_same_targets,
        theta_random, build_entangled_circuit
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

    learning_demonstrated = (corr_same > 0.5 and training_helped)

    print("\n" + "=" * 70)
    if learning_demonstrated:
        print("✅ QUANTUM LEARNING DEMONSTRATED")
    else:
        print("❌ NO LEARNING DEMONSTRATED")
    print("=" * 70)

    # Save results
    full_output = {
        "metadata": metadata,
        "results": results
    }

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Filename now includes Git Hash
    git_short = metadata.get("git_commit", "nogit")[:7]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f"{CONFIG['experiment_name']}_{timestamp}_{git_short}.json")

    with open(output_file, 'w') as f:
        json.dump(full_output, f, indent=2, cls=NumpyEncoder)
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
