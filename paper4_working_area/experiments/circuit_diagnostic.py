#!/usr/bin/env python3
"""
Circuit Diagnostic: Test HIGH, MEDIUM, and LOW similarity pairs
================================================================
Investigate whether the circuit can distinguish between:
- HIGH similarity (semantically identical/very close)
- MEDIUM similarity (related concepts)
- LOW similarity (unrelated concepts - NEGATIVES)

This will reveal if the circuit is compressing outputs to a narrow range.
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


def run_diagnostic():
    """Run diagnostic with HIGH, MEDIUM, and LOW similarity pairs."""

    print("=" * 70)
    print("CIRCUIT DIAGNOSTIC: HIGH/MEDIUM/LOW SIMILARITY")
    print("=" * 70)

    # Define test pairs in three categories
    # HIGH: Very similar concepts (expected target > 0.7)
    # MEDIUM: Related concepts (expected target 0.3-0.6)
    # LOW/NEGATIVE: Unrelated concepts (expected target < 0.2)

    diagnostic_pairs = {
        'HIGH': [
            ('dog', 'puppy'),           # Very similar
            ('car', 'automobile'),      # Synonyms
            ('happy', 'joyful'),        # Synonyms
        ],
        'MEDIUM': [
            ('dog', 'cat'),             # Related (both pets)
            ('car', 'road'),            # Related (transport)
            ('happy', 'emotion'),       # Related (emotion type)
        ],
        'LOW': [
            ('dog', 'mathematics'),     # Unrelated
            ('car', 'philosophy'),      # Unrelated
            ('happy', 'geology'),       # Unrelated
        ]
    }

    # Initialize
    data_prep = QManifoldDataPreparation(target_dim=N_DATA_QUBITS)

    # Setup hardware
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"
    )
    backend = service.backend(BACKEND_NAME)
    print(f"Backend: {backend.name}")

    # Get all unique concepts
    all_concepts = list(set(
        c for pairs in diagnostic_pairs.values()
        for pair in pairs
        for c in pair
    ))
    print(f"Testing {len(all_concepts)} unique concepts")

    # Embed concepts
    embeddings = data_prep.embed_concepts(all_concepts)
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # Build all circuits
    all_test_pairs = []
    circuits = []
    categories = []

    for category, pairs in diagnostic_pairs.items():
        for c1, c2 in pairs:
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]

            circuits.append(build_attention_circuit(v1, v2))
            all_test_pairs.append((c1, c2))
            categories.append(category)

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
    results_by_category = {'HIGH': [], 'MEDIUM': [], 'LOW': []}

    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)

    for i, ((c1, c2), category) in enumerate(zip(all_test_pairs, categories)):
        counts = result[i].data.c.get_counts()
        pred = compute_similarity_from_counts(counts, N_DATA_QUBITS)

        # Compute target
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        v1_pca = vectors_pca[idx1]
        v2_pca = vectors_pca[idx2]
        dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
        target = data_prep.hyperbolic_similarity(dist)

        results_by_category[category].append({
            'pair': (c1, c2),
            'target': target,
            'pred': pred,
            'error': pred - target
        })

    # Print results by category
    for category in ['HIGH', 'MEDIUM', 'LOW']:
        print(f"\n--- {category} SIMILARITY PAIRS ---")
        print(f"{'Pair':<25} | {'Target':>8} | {'Pred':>8} | {'Error':>8}")
        print("-" * 60)

        for r in results_by_category[category]:
            c1, c2 = r['pair']
            print(f"{c1} ↔ {c2:<12} | {r['target']:>8.3f} | {r['pred']:>8.3f} | {r['error']:>+8.3f}")

        targets = [r['target'] for r in results_by_category[category]]
        preds = [r['pred'] for r in results_by_category[category]]
        print(f"\n  Mean Target: {np.mean(targets):.3f}")
        print(f"  Mean Pred:   {np.mean(preds):.3f}")
        print(f"  Pred Range:  [{min(preds):.3f}, {max(preds):.3f}]")

    # Summary analysis
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    high_preds = [r['pred'] for r in results_by_category['HIGH']]
    med_preds = [r['pred'] for r in results_by_category['MEDIUM']]
    low_preds = [r['pred'] for r in results_by_category['LOW']]

    print(f"\nPrediction Means by Category:")
    print(f"  HIGH:   {np.mean(high_preds):.3f}")
    print(f"  MEDIUM: {np.mean(med_preds):.3f}")
    print(f"  LOW:    {np.mean(low_preds):.3f}")

    separation = np.mean(high_preds) - np.mean(low_preds)
    print(f"\nHIGH-LOW Separation: {separation:.3f}")

    if separation > 0.3:
        print("✅ Circuit CAN distinguish between similar and dissimilar pairs!")
    elif separation > 0.1:
        print("⚠️ Circuit has WEAK separation between categories")
    else:
        print("❌ Circuit CANNOT distinguish - outputs are compressed!")

    # Check if predictions follow expected order
    if np.mean(high_preds) > np.mean(med_preds) > np.mean(low_preds):
        print("✅ Predictions follow expected order: HIGH > MEDIUM > LOW")
    else:
        print("❌ Predictions DO NOT follow expected order!")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'backend': BACKEND_NAME,
        'n_qubits': N_DATA_QUBITS,
        'results_by_category': {
            cat: [{'pair': r['pair'], 'target': r['target'], 'pred': r['pred']}
                  for r in results]
            for cat, results in results_by_category.items()
        },
        'summary': {
            'high_mean': float(np.mean(high_preds)),
            'medium_mean': float(np.mean(med_preds)),
            'low_mean': float(np.mean(low_preds)),
            'separation': float(separation)
        }
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"circuit_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results_by_category


if __name__ == "__main__":
    print("=" * 70)
    print("CIRCUIT DIAGNOSTIC EXPERIMENT")
    print("=" * 70)
    print("Testing whether circuit can distinguish:")
    print("  - HIGH similarity (dog/puppy, car/automobile)")
    print("  - MEDIUM similarity (dog/cat, car/road)")
    print("  - LOW similarity (dog/mathematics, car/philosophy)")
    print("=" * 70)

    try:
        run_diagnostic()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
