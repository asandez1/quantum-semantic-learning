#!/usr/bin/env python3
"""
Q-Manifold Validation Benchmark (Paper 4)

Tests generalization of Phase 1B optimized parameters on full 150-pair dataset.
This is an INFERENCE-ONLY run (no training) - uses remaining quantum budget efficiently.

Estimated time: ~2 minutes for 150 pairs
"""

import sys
import os
import json
import numpy as np
import scipy.stats
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_preparation import QManifoldDataPreparation
from utils.quantum_circuit import QManifoldCircuit, FidelityMeasurement
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# === CONFIGURATION ===
PHASE_1B_RESULTS = "../results/phase1B_hardware_ibm_fez_20251122_155224.json"
BACKEND_NAME = "ibm_fez"
API_KEY = "Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL"
CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"
SHOTS = 4096
# =====================


def run_validation_benchmark():
    """
    Runs inference validation on 150-pair benchmark using Phase 1B optimized parameters.

    This tests GENERALIZATION: Can the circuit trained on 30 pairs predict relationships
    of 150 pairs?
    """
    print("=" * 70)
    print("Q-MANIFOLD VALIDATION BENCHMARK")
    print("=" * 70)

    # 1. Load Phase 1B Optimized Parameters
    print(f"\n[Step 1/6] Loading optimized parameters from Phase 1B...")
    with open(PHASE_1B_RESULTS, 'r') as f:
        phase1b_data = json.load(f)
        theta_opt = np.array(phase1b_data['theta_optimized'])

    print(f"  ✓ Loaded {len(theta_opt)} parameters")
    print(f"  ✓ Phase 1B final loss: {phase1b_data['final_loss']:.6f}")

    # 2. Prepare Full 150-Pair Benchmark
    print(f"\n[Step 2/6] Preparing full benchmark dataset...")
    data_prep = QManifoldDataPreparation(target_dim=20)

    # Get all available concept pairs
    all_pairs = data_prep.get_default_concept_pairs()

    # Take first 150 pairs (or all if less than 150)
    num_pairs = min(150, len(all_pairs))
    benchmark_pairs = all_pairs[:num_pairs]

    print(f"  ✓ Preparing {num_pairs} concept pairs...")

    # Prepare data (this fits PCA, computes hyperbolic distances)
    benchmark_data = data_prep.prepare_training_batch(benchmark_pairs, batch_size=num_pairs)

    vectors_20d = benchmark_data['vectors_20d']
    pair_indices = benchmark_data['pair_indices']
    target_similarities = benchmark_data['target_similarities']

    print(f"  ✓ Data prepared: {len(pair_indices)} pairs")
    print(f"  ✓ Target similarity range: [{min(target_similarities):.3f}, {max(target_similarities):.3f}]")

    # 3. Build Circuit
    print(f"\n[Step 3/6] Building quantum circuit...")
    circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2, entanglement='circular')
    print(f"  ✓ Circuit built: {circuit.get_parameter_count()} parameters")

    # 4. Connect to IBM Quantum and Transpile
    print(f"\n[Step 4/6] Connecting to IBM Cloud Quantum ({BACKEND_NAME})...")
    service = QiskitRuntimeService(channel="ibm_cloud", token=API_KEY, instance=CRN)
    backend = service.backend(BACKEND_NAME)

    print(f"  ✓ Backend: {backend.name}")
    print(f"  ✓ Qubits: {backend.num_qubits}")

    print(f"\n  Transpiling to ISA...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit.qc_sampler)

    print(f"  ✓ ISA circuit depth: {isa_circuit.depth()}")
    print(f"  ✓ ISA circuit width: {isa_circuit.num_qubits}")

    # 5. Prepare Parameter Bindings for Inference
    print(f"\n[Step 5/6] Preparing parameter bindings for {num_pairs} pairs...")

    # Build parameter bindings: [x_vec, theta, y_vec]
    # One evaluation per pair (no SPSA perturbations needed for inference)
    parameter_values = []

    for (idx1, idx2) in pair_indices:
        x_vec = vectors_20d[idx1]
        y_vec = vectors_20d[idx2]

        # Bind optimized theta
        params = np.concatenate([x_vec, theta_opt, y_vec])
        parameter_values.append(params)

    parameter_values = np.array(parameter_values)  # Shape: (num_pairs, num_params)

    print(f"  ✓ Parameter bindings prepared: {parameter_values.shape}")

    # 6. Execute Inference on Quantum Hardware
    print(f"\n[Step 6/6] Submitting inference job to quantum hardware...")
    print(f"  ⏱ Estimated time: ~2 minutes")

    # Create PUB (Primitive Unified Bloc)
    pub = (isa_circuit, parameter_values)

    # Execute
    sampler = SamplerV2(mode=backend)
    job = sampler.run([pub], shots=SHOTS)

    print(f"  ⏳ Job submitted, waiting for results...")
    result = job.result()

    print(f"  ✓ Quantum execution complete!")

    # 7. Analyze Results
    print(f"\n{'=' * 70}")
    print("ANALYSIS: GENERALIZATION TEST")
    print("=" * 70)

    pub_result = result[0]

    # Extract fidelities from quantum measurements
    fidelities = []
    for i in range(len(pair_indices)):
        counts = pub_result.data.meas.get_counts(i)
        fidelity = FidelityMeasurement.fidelity_from_counts(counts, circuit.n_qubits)
        fidelities.append(fidelity)

    fidelities = np.array(fidelities)

    # Compute correlation with target similarities
    correlation, p_value = scipy.stats.pearsonr(fidelities, target_similarities)

    # Compute loss
    losses = (fidelities - target_similarities) ** 2
    mean_loss = np.mean(losses)

    print(f"\nResults on {num_pairs} pairs:")
    print(f"  Fidelity range:       [{fidelities.min():.3f}, {fidelities.max():.3f}]")
    print(f"  Target range:         [{min(target_similarities):.3f}, {max(target_similarities):.3f}]")
    print(f"")
    print(f"  Fidelity-Target Correlation:  {correlation:.6f}")
    print(f"  P-value:                       {p_value:.2e}")
    print(f"  Mean Squared Loss:             {mean_loss:.6f}")
    print(f"")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)

    if correlation > 0.90:
        print("  🟢 EXCELLENT: Correlation > 0.90 - BREAKTHROUGH!")
        print("     The quantum circuit generalizes perfectly to unseen pairs.")
    elif correlation > 0.70:
        print("  🟡 GOOD: Correlation > 0.70 - Strong generalization")
        print("     The circuit learned meaningful geometric structure.")
    elif correlation > 0.50:
        print("  🟠 MODERATE: Correlation > 0.50 - Partial generalization")
        print("     Some geometric structure preserved, but imperfect.")
    elif correlation > 0.30:
        print("  🔴 WEAK: Correlation > 0.30 - Limited generalization")
        print("     Circuit may have overfit to training pairs.")
    else:
        print("  ⛔ POOR: Correlation < 0.30 - No generalization")
        print("     Circuit did not learn transferable geometry.")

    # Comparison to baselines
    print(f"\nComparison to Baselines:")
    print(f"  Paper 1 (Classical):      0.927 correlation")
    print(f"  Paper 2 (Best Quantum):   0.76 correlation")
    print(f"  Paper 3 (Hardware):       0.165 overall")
    print(f"  Q-Manifold (Training):    Loss {phase1b_data['final_loss']:.3f} on 30 pairs")
    print(f"  Q-Manifold (Validation):  Correlation {correlation:.3f} on {num_pairs} pairs")

    # 8. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'validation_benchmark',
        'timestamp': timestamp,
        'backend': BACKEND_NAME,
        'num_pairs': num_pairs,
        'shots': SHOTS,
        'phase1b_source': PHASE_1B_RESULTS,
        'phase1b_final_loss': phase1b_data['final_loss'],
        'results': {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'mean_loss': float(mean_loss),
            'fidelity_range': [float(fidelities.min()), float(fidelities.max())],
            'target_range': [float(min(target_similarities)), float(max(target_similarities))]
        },
        'fidelities': fidelities.tolist(),
        'target_similarities': target_similarities
    }

    results_file = f'../results/validation_benchmark_{num_pairs}pairs_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    print(f"\n{'=' * 70}")
    print("VALIDATION BENCHMARK COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_validation_benchmark()
