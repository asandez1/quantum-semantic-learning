#!/usr/bin/env python3
"""
Cosine Similarity Diagnostic: Corrected Target Calculation
============================================================
Uses COSINE SIMILARITY as targets instead of broken hyperbolic distance.

This should reveal the true correlation between circuit output and semantic similarity.
"""

import numpy as np
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from utils.data_preparation import QManifoldDataPreparation

# Configuration
BACKEND_NAME = "ibm_fez"
N_DATA_QUBITS = 12
SHOTS = 2048


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return dot / (norm1 * norm2)


def safe_corrcoef(preds, targets):
    """Calculate correlation safely."""
    preds = np.array(preds)
    targets = np.array(targets)
    if len(preds) < 2 or np.std(preds) < 1e-12 or np.std(targets) < 1e-12:
        return 0.0
    return np.corrcoef(preds, targets)[0, 1]


def compute_similarity_from_counts(counts: dict, n_qubits: int) -> float:
    """Compute similarity using normalized Hamming weight."""
    total_shots = sum(counts.values())
    weighted_hamming = 0.0
    for bitstring, count in counts.items():
        bs = bitstring.zfill(n_qubits)
        hamming_weight = bs.count('1')
        weighted_hamming += hamming_weight * count
    avg_hamming = weighted_hamming / total_shots
    similarity = 1.0 - (avg_hamming / (n_qubits / 2))
    return max(0.0, min(1.0, similarity))


def build_attention_circuit(v1: np.ndarray, v2: np.ndarray) -> QuantumCircuit:
    """Build the attention circuit."""
    n_qubits = N_DATA_QUBITS
    q_data = QuantumRegister(n_qubits, 'q')
    c_main = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(q_data, c_main)

    # Encode v1
    for i in range(min(len(v1), n_qubits)):
        qc.ry(float(v1[i]), q_data[i])

    # Attention layer 1: local entanglement
    for i in range(0, n_qubits - 1, 2):
        qc.cx(q_data[i], q_data[i + 1])

    # Attention layer 2: cross-group entanglement
    qc.barrier()
    for i in range(n_qubits // 2):
        if i + n_qubits // 2 < n_qubits:
            qc.cx(q_data[i], q_data[i + n_qubits // 2])

    # Encode -v2 (interference)
    for i in range(min(len(v2), n_qubits)):
        qc.ry(float(-v2[i]), q_data[i])

    qc.measure(q_data, c_main)
    return qc


def run_cosine_diagnostic():
    """Run diagnostic with cosine similarity targets."""

    print("=" * 70)
    print("COSINE SIMILARITY DIAGNOSTIC (CORRECTED TARGETS)")
    print("=" * 70)

    # Diverse test pairs across similarity spectrum
    test_pairs = [
        # HIGH similarity (expected cosine > 0.7)
        ('dog', 'puppy'),
        ('car', 'automobile'),
        ('happy', 'joyful'),
        ('big', 'large'),

        # MEDIUM similarity (expected cosine 0.4-0.7)
        ('dog', 'cat'),
        ('car', 'truck'),
        ('happy', 'sad'),
        ('tree', 'plant'),

        # LOW similarity (expected cosine < 0.4)
        ('dog', 'computer'),
        ('car', 'happiness'),
        ('tree', 'mathematics'),
        ('music', 'geology'),
    ]

    # Initialize
    data_prep = QManifoldDataPreparation(target_dim=N_DATA_QUBITS)

    # Setup hardware
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::",
    )
    backend = service.backend(BACKEND_NAME)
    print(f"Backend: {backend.name}")

    # Get all unique concepts
    all_concepts = list(set(c for pair in test_pairs for c in pair))
    print(f"Testing {len(all_concepts)} unique concepts, {len(test_pairs)} pairs")

    # Embed concepts (raw embeddings for cosine similarity)
    embeddings = data_prep.embed_concepts(all_concepts)

    # Also get scaled vectors for circuit encoding
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # Build circuits
    circuits = []
    for c1, c2 in test_pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        v1 = vectors_scaled[idx1]
        v2 = vectors_scaled[idx2]
        circuits.append(build_attention_circuit(v1, v2))

    # Transpile
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    print(f"Transpiling {len(circuits)} circuits...")
    isa_circuits = pm.run(circuits)

    # Submit job
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"Submitted job: {job.job_id()}")
    print("Waiting for job to complete...")

    result = job.result()
    print("Job finished!")

    # Process results
    print("\n" + "=" * 70)
    print("RESULTS: Circuit Prediction vs Cosine Similarity")
    print("=" * 70)
    print(f"{'Pair':<25} | {'Cosine':>8} | {'Pred':>8} | {'Error':>8} | Category")
    print("-" * 70)

    all_targets = []
    all_predictions = []
    results_data = []

    for i, (c1, c2) in enumerate(test_pairs):
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)

        # Cosine similarity from RAW embeddings (correct target)
        target = cosine_similarity(embeddings[idx1], embeddings[idx2])

        # Circuit prediction
        counts = result[i].data.c.get_counts()
        pred = compute_similarity_from_counts(counts, N_DATA_QUBITS)

        error = pred - target

        # Categorize
        if target > 0.7:
            category = "HIGH"
        elif target > 0.4:
            category = "MEDIUM"
        else:
            category = "LOW"

        all_targets.append(target)
        all_predictions.append(pred)
        results_data.append({
            'pair': (c1, c2),
            'target': target,
            'pred': pred,
            'category': category
        })

        print(f"{c1} ↔ {c2:<12} | {target:>8.3f} | {pred:>8.3f} | {error:>+8.3f} | {category}")

    # Compute correlation
    correlation = safe_corrcoef(all_predictions, all_targets)

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # By category
    for cat in ['HIGH', 'MEDIUM', 'LOW']:
        cat_data = [r for r in results_data if r['category'] == cat]
        if cat_data:
            cat_targets = [r['target'] for r in cat_data]
            cat_preds = [r['pred'] for r in cat_data]
            print(f"\n{cat} Similarity Pairs:")
            print(f"  Target range: [{min(cat_targets):.3f}, {max(cat_targets):.3f}]")
            print(f"  Pred range:   [{min(cat_preds):.3f}, {max(cat_preds):.3f}]")
            print(f"  Mean target:  {np.mean(cat_targets):.3f}")
            print(f"  Mean pred:    {np.mean(cat_preds):.3f}")

    print(f"\n{'=' * 70}")
    print(f"OVERALL CORRELATION: {correlation:.4f}")
    print(f"{'=' * 70}")

    if correlation > 0.7:
        print("EXCELLENT: Circuit strongly correlates with semantic similarity!")
    elif correlation > 0.5:
        print("GOOD: Circuit moderately correlates with semantic similarity")
    elif correlation > 0.3:
        print("WEAK: Circuit has weak correlation with semantic similarity")
    else:
        print("POOR: Circuit does not correlate well with semantic similarity")

    # Check monotonicity
    high_preds = [r['pred'] for r in results_data if r['category'] == 'HIGH']
    med_preds = [r['pred'] for r in results_data if r['category'] == 'MEDIUM']
    low_preds = [r['pred'] for r in results_data if r['category'] == 'LOW']

    if np.mean(high_preds) > np.mean(med_preds) > np.mean(low_preds):
        print("Predictions correctly ordered: HIGH > MEDIUM > LOW")
    else:
        print(f"Prediction order: HIGH={np.mean(high_preds):.3f}, MED={np.mean(med_preds):.3f}, LOW={np.mean(low_preds):.3f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'backend': BACKEND_NAME,
        'n_qubits': N_DATA_QUBITS,
        'correlation': float(correlation),
        'results': [
            {'pair': r['pair'], 'target': r['target'], 'pred': r['pred'], 'category': r['category']}
            for r in results_data
        ],
        'summary': {
            'high_mean_target': float(np.mean([r['target'] for r in results_data if r['category'] == 'HIGH'])),
            'high_mean_pred': float(np.mean([r['pred'] for r in results_data if r['category'] == 'HIGH'])),
            'med_mean_target': float(np.mean([r['target'] for r in results_data if r['category'] == 'MEDIUM'])),
            'med_mean_pred': float(np.mean([r['pred'] for r in results_data if r['category'] == 'MEDIUM'])),
            'low_mean_target': float(np.mean([r['target'] for r in results_data if r['category'] == 'LOW'])),
            'low_mean_pred': float(np.mean([r['pred'] for r in results_data if r['category'] == 'LOW'])),
        }
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"cosine_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return correlation, results_data


if __name__ == "__main__":
    print("=" * 70)
    print("COSINE SIMILARITY DIAGNOSTIC")
    print("=" * 70)
    print("Using COSINE SIMILARITY as targets (not hyperbolic)")
    print("This measures true semantic correlation")
    print("=" * 70)

    try:
        correlation, results = run_cosine_diagnostic()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
