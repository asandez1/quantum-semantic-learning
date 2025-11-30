#!/usr/bin/env python3
"""
Q-Manifold Probe Experiment (Paper 4)

Strategic 3-phase execution designed for 10-minute quantum budget:
- Phase 0: Simulator validation (free, verify convergence)
- Phase 1A: Hardware proof-of-concept (3 iterations, ~2 min)
- Phase 1B: Hardware convergence test (7 iterations, ~5 min)
- Phase 1C: Reserve buffer (~2 min)

Usage:
    python run_qmanifold_probe.py --phase 0  # Simulator only
    python run_qmanifold_probe.py --phase 1A # Hardware probe
    python run_qmanifold_probe.py --phase 1B # Hardware full
    python run_qmanifold_probe.py --phase all # All phases
"""

import sys
import os
import argparse
import numpy as np
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_preparation import QManifoldDataPreparation
from utils.quantum_circuit import QManifoldCircuit
from utils.batch_optimizer import BatchSPSAOptimizer, SPSAConfig

# Qiskit imports
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, EstimatorV2


def phase_0_simulator(data_prep, num_pairs=16, iterations=10, use_estimator=True):
    """
    Phase 0: Simulator Validation

    Goals:
    - Verify circuit constructs correctly
    - Confirm SPSA converges
    - Establish baseline loss
    - Validate data pipeline

    Budget: Free (local simulation)
    Time: ~2-3 minutes
    """
    print("\n" + "=" * 70)
    print("PHASE 0: SIMULATOR VALIDATION")
    print("=" * 70)

    # Prepare data
    print("\n[Phase 0] Preparing data...")
    pairs = data_prep.get_default_concept_pairs()[:num_pairs]
    data = data_prep.prepare_training_batch(pairs, batch_size=8)

    # Build circuit
    print("\n[Phase 0] Building quantum circuit...")
    circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2, entanglement='circular')

    # Setup simulator
    print("\n[Phase 0] Setting up Aer simulator...")
    simulator = AerSimulator()
    pm = generate_preset_pass_manager(
        backend=simulator,
        optimization_level=3
    )

    # Transpile circuit (one version for simulation)
    if use_estimator:
        isa_circuit = pm.run(circuit.qc_estimator)
        observable = circuit.get_observable_fidelity_proxy()
    else:
        isa_circuit = pm.run(circuit.qc_sampler)
        observable = None

    print(f"[Phase 0] Transpiled circuit depth: {isa_circuit.depth()}")

    # Initialize parameters
    theta_init = np.random.rand(circuit.get_parameter_count()) * 0.1

    # Configure optimizer
    config = SPSAConfig(
        max_iterations=iterations,
        batch_size=8,
        learning_rate=0.1,
        perturbation_size=0.1,
        shots=2048,
        use_estimator=use_estimator
    )

    optimizer = BatchSPSAOptimizer(config, circuit, verbose=True)

    # Run optimization
    print("\n[Phase 0] Starting optimization...")

    if use_estimator:
        estimator = EstimatorV2(mode=simulator)
        theta_opt, history = optimizer.optimize_with_estimator(
            theta_init, data, estimator, isa_circuit, observable
        )
    else:
        sampler = SamplerV2(mode=simulator)
        theta_opt, history = optimizer.optimize_with_sampler(
            theta_init, data, sampler, isa_circuit
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'phase': '0_simulator',
        'timestamp': timestamp,
        'config': {
            'num_pairs': num_pairs,
            'iterations': iterations,
            'use_estimator': use_estimator,
            'batch_size': config.batch_size,
            'shots': config.shots
        },
        'initial_loss': history[0]['loss'],
        'final_loss': history[-1]['loss'],
        'improvement': history[0]['loss'] - history[-1]['loss'],
        'history': history,
        'theta_optimized': theta_opt.tolist()
    }

    results_file = f'../results/phase0_simulator_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Phase 0] RESULTS:")
    print(f"  Initial loss: {results['initial_loss']:.6f}")
    print(f"  Final loss: {results['final_loss']:.6f}")
    print(f"  Improvement: {results['improvement']:.6f}")
    print(f"  Saved to: {results_file}")

    # Analysis
    if results['improvement'] > 0:
        print("\n✓ PHASE 0 SUCCESS: Optimizer converges!")
    else:
        print("\n⚠ PHASE 0 WARNING: No convergence detected")

    return results


def phase_1A_hardware_probe(data_prep, theta_init=None, backend_name='ibm_kyiv', skip_confirm=False):
    """
    Phase 1A: Hardware Proof-of-Concept

    Goals:
    - Verify hardware execution works
    - Confirm transpilation successful
    - Check if hardware advantage exists (vs simulator)

    Budget: ~2 minutes (3 iterations × 40s)
    Batch: 8 pairs
    """
    print("\n" + "=" * 70)
    print("PHASE 1A: HARDWARE PROOF-OF-CONCEPT")
    print("=" * 70)

    if not skip_confirm:
        print("\n⚠ WARNING: This will use ~2 minutes of quantum time!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return None
    else:
        print("⚠ Using ~2 minutes of quantum time (auto-confirmed)")

    # Prepare data (smaller set)
    print("\n[Phase 1A] Preparing data...")
    pairs = data_prep.get_default_concept_pairs()[:16]
    data = data_prep.prepare_training_batch(pairs, batch_size=8)

    # Build circuit
    print("\n[Phase 1A] Building quantum circuit...")
    circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2, entanglement='circular')

    # Connect to IBM Quantum with explicit credentials
    print(f"\n[Phase 1A] Connecting to IBM Cloud Quantum ({backend_name})...")
    API_KEY = "Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL"
    CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"
    service = QiskitRuntimeService(channel="ibm_cloud", token=API_KEY, instance=CRN)
    backend = service.backend(backend_name)

    print(f"[Phase 1A] Backend: {backend.name}")
    print(f"[Phase 1A] Qubits: {backend.num_qubits}")

    # Transpile for hardware (constrain to 20 qubits)
    print("\n[Phase 1A] Transpiling to ISA...")

    # Specify initial qubit layout to constrain to 20 qubits
    from qiskit.transpiler import Layout
    initial_layout = Layout.from_intlist(list(range(20)), *circuit.qc_estimator.qregs)

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        initial_layout=initial_layout
    )

    isa_circuit = pm.run(circuit.qc_estimator)

    # Create observable matching the transpiled circuit qubits
    # The observable needs to match the physical qubits used
    observable = circuit.get_observable_fidelity_proxy()

    print(f"[Phase 1A] ISA circuit depth: {isa_circuit.depth()}")
    print(f"[Phase 1A] ISA circuit width: {isa_circuit.num_qubits}")

    # Initialize parameters
    if theta_init is None:
        theta_init = np.random.rand(circuit.get_parameter_count()) * 0.1

    # Configure optimizer (aggressive settings for probe)
    # Use SAMPLER for hardware to avoid observable mismatch issues
    config = SPSAConfig(
        max_iterations=3,  # Only 3 iterations for probe
        batch_size=8,
        learning_rate=0.15,  # Slightly higher for faster convergence
        perturbation_size=0.1,
        shots=2048,  # Balance precision vs speed
        use_estimator=False  # Use Sampler for hardware
    )

    optimizer = BatchSPSAOptimizer(config, circuit, verbose=True)

    # Run optimization
    print("\n[Phase 1A] Starting hardware optimization...")
    print("[Phase 1A] Estimated time: ~2 minutes")

    # Transpile the sampler circuit instead
    isa_circuit_sampler = pm.run(circuit.qc_sampler)

    sampler = SamplerV2(mode=backend)
    theta_opt, history = optimizer.optimize_with_sampler(
        theta_init, data, sampler, isa_circuit_sampler
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'phase': '1A_hardware_probe',
        'timestamp': timestamp,
        'backend': backend.name,
        'config': {
            'num_pairs': len(pairs),
            'iterations': 3,
            'batch_size': config.batch_size,
            'shots': config.shots
        },
        'initial_loss': history[0]['loss'],
        'final_loss': history[-1]['loss'],
        'improvement': history[0]['loss'] - history[-1]['loss'],
        'history': history,
        'theta_optimized': theta_opt.tolist()
    }

    results_file = f'../results/phase1A_hardware_{backend.name}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Phase 1A] RESULTS:")
    print(f"  Backend: {backend.name}")
    print(f"  Initial loss: {results['initial_loss']:.6f}")
    print(f"  Final loss: {results['final_loss']:.6f}")
    print(f"  Improvement: {results['improvement']:.6f}")
    print(f"  Saved to: {results_file}")

    total_time = sum(h['time'] for h in history)
    print(f"\n[Phase 1A] Time used: {total_time / 60:.2f} minutes")
    print(f"[Phase 1A] Remaining budget: ~{10 - total_time / 60:.2f} minutes")

    return results, theta_opt


def phase_1B_hardware_convergence(data_prep, theta_init, backend_name='ibm_kyiv', skip_confirm=False):
    """
    Phase 1B: Hardware Convergence Test

    Goals:
    - Demonstrate full convergence on hardware
    - Measure final correlation with hyperbolic distances
    - Generate publication-quality results

    Budget: ~5 minutes (7 iterations × 40s)
    Batch: 12 pairs (more diversity)
    """
    print("\n" + "=" * 70)
    print("PHASE 1B: HARDWARE CONVERGENCE TEST")
    print("=" * 70)

    if not skip_confirm:
        print("\n⚠ WARNING: This will use ~5 minutes of quantum time!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return None
    else:
        print("⚠ Using ~5 minutes of quantum time (auto-confirmed)")

    # Prepare data (larger set)
    print("\n[Phase 1B] Preparing data...")
    pairs = data_prep.get_default_concept_pairs()[:30]
    data = data_prep.prepare_training_batch(pairs, batch_size=12)

    # Build circuit
    circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2, entanglement='circular')

    # Connect to IBM Quantum with explicit credentials
    print(f"\n[Phase 1B] Connecting to IBM Cloud Quantum ({backend_name})...")
    API_KEY = "Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL"
    CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"
    service = QiskitRuntimeService(channel="ibm_cloud", token=API_KEY, instance=CRN)
    backend = service.backend(backend_name)

    # Transpile (constrain to 20 qubits)
    from qiskit.transpiler import Layout
    initial_layout = Layout.from_intlist(list(range(20)), *circuit.qc_estimator.qregs)

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        initial_layout=initial_layout
    )
    isa_circuit = pm.run(circuit.qc_estimator)
    observable = circuit.get_observable_fidelity_proxy()

    # Configure optimizer
    config = SPSAConfig(
        max_iterations=7,
        batch_size=12,
        learning_rate=0.1,
        perturbation_size=0.1,
        shots=4096,  # Higher precision
        use_estimator=False  # Use Sampler for hardware
    )

    optimizer = BatchSPSAOptimizer(config, circuit, verbose=True)

    # Run optimization
    print("\n[Phase 1B] Starting hardware optimization...")
    print("[Phase 1B] Estimated time: ~5 minutes")

    # Transpile sampler circuit
    isa_circuit_sampler = pm.run(circuit.qc_sampler)

    sampler = SamplerV2(mode=backend)
    theta_opt, history = optimizer.optimize_with_sampler(
        theta_init, data, sampler, isa_circuit_sampler
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'phase': '1B_hardware_convergence',
        'timestamp': timestamp,
        'backend': backend.name,
        'config': {
            'num_pairs': len(pairs),
            'iterations': 7,
            'batch_size': config.batch_size,
            'shots': config.shots
        },
        'initial_loss': history[0]['loss'],
        'final_loss': history[-1]['loss'],
        'improvement': history[0]['loss'] - history[-1]['loss'],
        'history': history,
        'theta_optimized': theta_opt.tolist()
    }

    results_file = f'../results/phase1B_hardware_{backend.name}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Phase 1B] RESULTS:")
    print(f"  Backend: {backend.name}")
    print(f"  Initial loss: {results['initial_loss']:.6f}")
    print(f"  Final loss: {results['final_loss']:.6f}")
    print(f"  Improvement: {results['improvement']:.6f}")
    print(f"  Convergence rate: {results['improvement'] / 7:.6f} per iteration")

    total_time = sum(h['time'] for h in history)
    print(f"\n[Phase 1B] Time used: {total_time / 60:.2f} minutes")

    return results


def main():
    parser = argparse.ArgumentParser(description='Q-Manifold Probe Experiment')
    parser.add_argument(
        '--phase',
        choices=['0', '1A', '1B', 'all'],
        default='0',
        help='Which phase to run (default: 0 for simulator)'
    )
    parser.add_argument(
        '--backend',
        default='ibm_fez',
        help='IBM Cloud Quantum backend name (ibm_fez, ibm_torino, or ibm_marrakesh)'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompts'
    )

    args = parser.parse_args()

    # Initialize data preparation
    print("Initializing Q-Manifold experiment...")
    data_prep = QManifoldDataPreparation(target_dim=20)

    # Run requested phase(s)
    if args.phase == '0' or args.phase == 'all':
        phase_0_simulator(data_prep, num_pairs=16, iterations=10)

    if args.phase == '1A' or args.phase == 'all':
        result_1A, theta_1A = phase_1A_hardware_probe(data_prep, backend_name=args.backend, skip_confirm=args.yes)

        if args.phase == 'all' and result_1A is not None:
            # Use optimized theta from 1A as initialization for 1B
            phase_1B_hardware_convergence(data_prep, theta_1A, backend_name=args.backend, skip_confirm=args.yes)

    if args.phase == '1B':
        # Need to load theta from previous run
        print("\n[Error] Phase 1B requires theta from Phase 1A.")
        print("Either run Phase 1A first, or use --phase all")
        return

    print("\n" + "=" * 70)
    print("Q-MANIFOLD EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
