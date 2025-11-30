#!/usr/bin/env python3
"""
Dynamic Circuit Proof-of-Concept for Phase Transition Mitigation
=================================================================
Test whether dynamic circuits can prevent signal collapse below the 11-pair threshold.

Key Innovation: Use mid-circuit measurements to detect collapse and apply corrections.
Target: Make 8-pair training work (currently produces -0.2263 correlation).

v2: Simplified dynamic circuit with single ancilla, uses ibm_fez backend.
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
BACKEND_NAME = "ibm_fez"  # Changed from ibm_torino - better dynamic circuit support
N_DATA_QUBITS = 12  # Reduced from 20 for faster execution and less noise


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
        # Pad bitstring to n_qubits length
        bs = bitstring.zfill(n_qubits)
        hamming_weight = bs.count('1')
        weighted_hamming += hamming_weight * count

    avg_hamming = weighted_hamming / total_shots
    # Normalize: 0 hamming = similarity 1, n_qubits/2 hamming = similarity 0
    similarity = 1.0 - (avg_hamming / (n_qubits / 2))
    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]


def build_dynamic_attention_circuit(v1: np.ndarray, v2: np.ndarray,
                                   use_dynamic: bool = True) -> QuantumCircuit:
    """
    Build attention circuit with optional dynamic error mitigation.

    Simplified v2: Single ancilla qubit, single conditional branch.
    """
    n_qubits = N_DATA_QUBITS
    q_data = QuantumRegister(n_qubits, 'q')
    c_main = ClassicalRegister(n_qubits, 'c')

    if use_dynamic:
        q_ancilla = QuantumRegister(1, 'anc')
        c_syndrome = ClassicalRegister(1, 'syn')
        qc = QuantumCircuit(q_data, q_ancilla, c_main, c_syndrome)
    else:
        qc = QuantumCircuit(q_data, c_main)

    # === Encoding v1 (truncate to n_qubits) ===
    for i in range(min(len(v1), n_qubits)):
        qc.ry(float(v1[i]), q_data[i])

    # === Attention layer 1: local entanglement ===
    for i in range(0, n_qubits - 1, 2):
        qc.cx(q_data[i], q_data[i + 1])

    # === DYNAMIC CORRECTION (simplified: single ancilla, single condition) ===
    if use_dynamic:
        # Prepare ancilla in superposition
        qc.h(q_ancilla[0])

        # Parity check on first half of qubits
        for i in range(n_qubits // 2):
            qc.cx(q_data[i], q_ancilla[0])

        # Mid-circuit measurement
        qc.measure(q_ancilla[0], c_syndrome[0])

        # Conditional correction: if ancilla measured 1, apply phase correction
        with qc.if_test((c_syndrome[0], 1)):
            # Apply corrective rotations to second half
            for i in range(n_qubits // 2, n_qubits):
                qc.rz(np.pi / 4, q_data[i])
                qc.ry(np.pi / 8, q_data[i])

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

def test_dynamic_mitigation(n_pairs: int = 8):
    """
    Test if dynamic circuits can rescue quantum attention below threshold.
    """
    print("=" * 70)
    print(f"DYNAMIC CIRCUIT TEST AT {n_pairs} PAIRS")
    print(f"Backend: {BACKEND_NAME}, Qubits: {N_DATA_QUBITS}")
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

    # Prepare data
    all_pairs = data_prep.get_default_concept_pairs()[:30]
    train_pairs = all_pairs[:n_pairs]
    test_pairs = all_pairs[n_pairs:n_pairs + 5]  # Use 5 test pairs for better statistics

    all_concepts = list(set(c for p in all_pairs for c in p))
    embeddings = data_prep.embed_concepts(all_concepts)
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    results = {}

    for use_dynamic in [False, True]:
        mode = "Dynamic" if use_dynamic else "Standard"
        print(f"\n--- Testing {mode} Circuit ---")

        circuits_to_run = []
        for c1, c2 in test_pairs:
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]
            circuits_to_run.append(build_dynamic_attention_circuit(v1, v2, use_dynamic=use_dynamic))

        # Transpile circuits
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        print(f"  Transpiling {len(circuits_to_run)} circuits...")
        isa_circuits = pm.run(circuits_to_run)

        # Print circuit info
        for i, circ in enumerate(isa_circuits):
            print(f"    Circuit {i}: depth={circ.depth()}, ops={circ.count_ops().get('cx', 0)} CX")

        # Submit job
        sampler = SamplerV2(mode=backend)
        job = sampler.run(isa_circuits, shots=2048)
        print(f"  Submitted job: {job.job_id()}")
        print("  Waiting for job to complete...")

        try:
            result = job.result()
            print("  Job finished successfully!")
        except Exception as e:
            print(f"  Job FAILED: {e}")
            results[mode] = {'correlation': 0, 'predictions': [], 'targets': [], 'error': str(e)}
            continue

        test_predictions = []
        test_targets = []
        for i, (c1, c2) in enumerate(test_pairs):
            data = result[i].data

            # Main measurement counts from the 'c' register
            counts = data.c.get_counts()
            total_shots = sum(counts.values())

            if use_dynamic and hasattr(data, 'syn'):
                # Syndrome measurement counts
                syndrome_counts = data.syn.get_counts()
                syn_0 = syndrome_counts.get(0, 0)
                syn_1 = syndrome_counts.get(1, 0)
                print(f"    Syndrome for '{c1}'-'{c2}': 0={syn_0}, 1={syn_1}")

            # Use Hamming weight based similarity metric
            similarity = compute_similarity_from_counts(counts, N_DATA_QUBITS)
            test_predictions.append(similarity)

            # Target
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
            v1_pca = vectors_pca[idx1]
            v2_pca = vectors_pca[idx2]
            dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
            target = data_prep.hyperbolic_similarity(dist)
            test_targets.append(target)

            print(f"  {c1} ↔ {c2}: target={target:.3f}, pred={similarity:.3f}")

        corr = safe_corrcoef(test_predictions, test_targets)
        results[mode] = {
            'correlation': corr,
            'predictions': test_predictions,
            'targets': test_targets,
            'signal_range': max(test_predictions) - min(test_predictions) if test_predictions else 0
        }
        print(f"\n{mode} Correlation: {corr:.4f}")
        print(f"Signal Range: {results[mode]['signal_range']:.4f}")

    # Compare results
    print("\n" + "=" * 70)
    print("DYNAMIC CIRCUIT IMPACT")
    print("=" * 70)
    standard_corr = results.get('Standard', {}).get('correlation', 0)
    dynamic_corr = results.get('Dynamic', {}).get('correlation', 0)
    improvement = dynamic_corr - standard_corr
    print(f"Standard Circuit: {standard_corr:.4f}")
    print(f"Dynamic Circuit:  {dynamic_corr:.4f}")
    print(f"Improvement:      {improvement:+.4f}")

    if improvement > 0.2:
        print("\n✅ BREAKTHROUGH: Dynamic circuits successfully mitigate collapse!")
    else:
        print("\n- No significant improvement observed.")

    output = {
        'timestamp': datetime.now().isoformat(),
        'n_pairs': n_pairs,
        'backend': BACKEND_NAME,
        'n_qubits': N_DATA_QUBITS,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'error'} for k, v in results.items()},
        'improvement': improvement
    }

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"dynamic_circuit_{n_pairs}pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("DYNAMIC CIRCUIT PROOF OF CONCEPT v2")
    print("=" * 70)
    print(f"Backend: {BACKEND_NAME}")
    print(f"Data qubits: {N_DATA_QUBITS}")
    print("Testing if dynamic error mitigation can lower phase transition threshold")
    print("Estimated time: 2-3 minutes per mode (Standard + Dynamic)")
    print("=" * 70)

    n_pairs = 8
    if len(sys.argv) > 1:
        try:
            n_pairs = int(sys.argv[1])
        except ValueError:
            print(f"Ignoring invalid argument '{sys.argv[1]}'. Using {n_pairs} pairs.")

    try:
        results = test_dynamic_mitigation(n_pairs)

        # Check for breakthrough
        std_corr = results.get('Standard', {}).get('correlation', 0)
        dyn_corr = results.get('Dynamic', {}).get('correlation', 0)

        if dyn_corr > std_corr + 0.2:
            print("\n" + "=" * 70)
            print("BREAKTHROUGH: Dynamic circuits successfully mitigate collapse!")
            print("=" * 70)
        elif 'error' in results.get('Dynamic', {}):
            print("\n" + "=" * 70)
            print("Dynamic circuit failed - see error above")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()