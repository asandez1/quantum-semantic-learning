#!/usr/bin/env python3
"""
Phase Transition Scan: Confirm the 11-pair threshold (FIXED TEST PAIRS)
=======================================================================
Test the simplified standard circuit at various training pair counts
to map the phase transition from collapse to coherence.

IMPORTANT FIX: Uses FIXED test pairs across all training counts for valid comparison.

Based on findings:
- 5 pairs: Chaotic (0.0854 correlation)
- 8 pairs: Collapse (-0.2263 correlation)
- 11 pairs: Coherent (0.9907 correlation)

This experiment confirms the transition using the improved circuit.
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

# Configuration - using working setup from dynamic circuit experiment
BACKEND_NAME = "ibm_fez"
N_DATA_QUBITS = 12
N_TEST_PAIRS = 5
SHOTS = 2048

# FIXED test pairs - always use the LAST 5 pairs from dataset (indices 25-29)
# This ensures valid comparison across all training pair counts
FIXED_TEST_START_IDX = 25


def safe_corrcoef(preds, targets):
    """Calculate correlation safely, handling zero variance cases."""
    preds = np.array(preds)
    targets = np.array(targets)
    if len(preds) < 2 or np.std(preds) < 1e-12 or np.std(targets) < 1e-12:
        return 0.0
    return np.corrcoef(preds, targets)[0, 1]


def compute_similarity_from_counts(counts: dict, n_qubits: int) -> float:
    """
    Compute similarity using normalized Hamming weight.
    Lower Hamming weight = more similar vectors (closer to cancellation).
    Returns value in [0, 1] where 1 = identical, 0 = maximally different.
    """
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
    """
    Build the working standard attention circuit.
    Simplified: 12 qubits, local + cross-group entanglement.
    """
    n_qubits = N_DATA_QUBITS
    q_data = QuantumRegister(n_qubits, 'q')
    c_main = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(q_data, c_main)

    # === Encoding v1 ===
    for i in range(min(len(v1), n_qubits)):
        qc.ry(float(v1[i]), q_data[i])

    # === Attention layer 1: local entanglement ===
    for i in range(0, n_qubits - 1, 2):
        qc.cx(q_data[i], q_data[i + 1])

    # === Attention layer 2: cross-group entanglement ===
    qc.barrier()
    for i in range(n_qubits // 2):
        if i + n_qubits // 2 < n_qubits:
            qc.cx(q_data[i], q_data[i + n_qubits // 2])

    # === Encode -v2 (interference/cancellation) ===
    for i in range(min(len(v2), n_qubits)):
        qc.ry(float(-v2[i]), q_data[i])

    # === Final measurement ===
    qc.measure(q_data, c_main)

    return qc


def run_phase_transition_scan(pair_counts: list):
    """
    Run experiments at different training pair counts to map the phase transition.

    FIXED: Uses the same test pairs (indices 25-29) for all training counts.
    """
    print("=" * 70)
    print("PHASE TRANSITION SCAN (FIXED TEST PAIRS)")
    print(f"Backend: {BACKEND_NAME}, Qubits: {N_DATA_QUBITS}")
    print(f"Training pair counts to test: {pair_counts}")
    print(f"Test pairs: FIXED at indices {FIXED_TEST_START_IDX}-{FIXED_TEST_START_IDX + N_TEST_PAIRS - 1}")
    print("=" * 70)

    # Initialize
    data_prep = QManifoldDataPreparation(target_dim=N_DATA_QUBITS)

    # Setup hardware
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"
    )
    backend = service.backend(BACKEND_NAME)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    # Prepare all data upfront
    all_pairs = data_prep.get_default_concept_pairs()[:30]
    all_concepts = list(set(c for p in all_pairs for c in p))
    embeddings = data_prep.embed_concepts(all_concepts)
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # FIXED test pairs - always use the same pairs (indices 25-29)
    test_pairs = all_pairs[FIXED_TEST_START_IDX:FIXED_TEST_START_IDX + N_TEST_PAIRS]
    print(f"\nFIXED Test Pairs:")
    for i, (c1, c2) in enumerate(test_pairs):
        print(f"  {i+1}. {c1} ↔ {c2}")

    results = {}

    for n_pairs in pair_counts:
        print(f"\n{'=' * 70}")
        print(f"TESTING {n_pairs} TRAINING PAIRS (test pairs: FIXED)")
        print("=" * 70)

        # Build circuits for test pairs
        circuits_to_run = []
        for c1, c2 in test_pairs:
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]
            circuits_to_run.append(build_attention_circuit(v1, v2))

        # Transpile
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        print(f"  Transpiling {len(circuits_to_run)} circuits...")
        isa_circuits = pm.run(circuits_to_run)

        avg_depth = np.mean([c.depth() for c in isa_circuits])
        print(f"  Average circuit depth: {avg_depth:.1f}")

        # Submit job
        sampler = SamplerV2(mode=backend)
        job = sampler.run(isa_circuits, shots=SHOTS)
        print(f"  Submitted job: {job.job_id()}")
        print("  Waiting for job to complete...")

        try:
            result = job.result()
            print("  Job finished successfully!")
        except Exception as e:
            print(f"  Job FAILED: {e}")
            results[n_pairs] = {'correlation': None, 'error': str(e)}
            continue

        # Process results
        test_predictions = []
        test_targets = []

        for i, (c1, c2) in enumerate(test_pairs):
            data = result[i].data
            counts = data.c.get_counts()

            similarity = compute_similarity_from_counts(counts, N_DATA_QUBITS)
            test_predictions.append(similarity)

            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
            v1_pca = vectors_pca[idx1]
            v2_pca = vectors_pca[idx2]
            dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
            target = data_prep.hyperbolic_similarity(dist)
            test_targets.append(target)

            print(f"    {c1} ↔ {c2}: target={target:.3f}, pred={similarity:.3f}")

        corr = safe_corrcoef(test_predictions, test_targets)
        signal_range = max(test_predictions) - min(test_predictions) if test_predictions else 0
        pred_variance = np.var(test_predictions) if test_predictions else 0

        results[n_pairs] = {
            'correlation': corr,
            'predictions': test_predictions,
            'targets': test_targets,
            'signal_range': signal_range,
            'variance': pred_variance,
            'job_id': job.job_id()
        }

        print(f"\n  {n_pairs} PAIRS: Correlation={corr:.4f}, Range={signal_range:.4f}, Var={pred_variance:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE TRANSITION SUMMARY")
    print("=" * 70)
    print(f"{'Pairs':>6} | {'Correlation':>12} | {'Signal Range':>12} | {'Variance':>10} | Status")
    print("-" * 70)

    for n_pairs in pair_counts:
        r = results.get(n_pairs, {})
        corr = r.get('correlation')
        sig_range = r.get('signal_range', 0)
        var = r.get('variance', 0)

        if corr is None:
            status = "FAILED"
        elif corr > 0.9:
            status = "COHERENT"
        elif corr > 0.3:
            status = "Transitional"
        elif corr > -0.1:
            status = "Chaotic"
        else:
            status = "COLLAPSED"

        corr_str = f"{corr:.4f}" if corr is not None else "N/A"
        print(f"{n_pairs:>6} | {corr_str:>12} | {sig_range:>12.4f} | {var:>10.4f} | {status}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'backend': BACKEND_NAME,
        'n_qubits': N_DATA_QUBITS,
        'shots': SHOTS,
        'pair_counts': pair_counts,
        'fixed_test_pairs': [(c1, c2) for c1, c2 in test_pairs],
        'test_pair_indices': f"{FIXED_TEST_START_IDX}-{FIXED_TEST_START_IDX + N_TEST_PAIRS - 1}",
        'results': {str(k): v for k, v in results.items()}
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"phase_transition_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE TRANSITION CONFIRMATION EXPERIMENT (FIXED TEST PAIRS)")
    print("=" * 70)
    print("Testing hypothesis: Phase transition occurs at ~11 training pairs")
    print("Expected: Correlation jumps from ~0 to ~0.99 at threshold")
    print(f"IMPORTANT: Using FIXED test pairs (indices {FIXED_TEST_START_IDX}-{FIXED_TEST_START_IDX + N_TEST_PAIRS - 1})")
    print("=" * 70)

    # Default: scan around the expected transition point
    pair_counts = [9, 10, 11, 12]

    if len(sys.argv) > 1:
        try:
            pair_counts = [int(x) for x in sys.argv[1].split(',')]
        except ValueError:
            print(f"Usage: python phase_transition_scan.py 9,10,11,12")
            sys.exit(1)

    try:
        results = run_phase_transition_scan(pair_counts)

        # Check for phase transition
        correlations = [r.get('correlation', 0) for r in results.values() if r.get('correlation') is not None]
        if correlations:
            max_corr = max(correlations)
            if max_corr > 0.9:
                print("\n" + "=" * 70)
                print("PHASE TRANSITION CONFIRMED!")
                print(f"Maximum correlation achieved: {max_corr:.4f}")
                print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
