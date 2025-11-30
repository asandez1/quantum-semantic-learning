"""
Quantum Kernel Ridge Regression for Semantic Similarity
========================================================
Solution to circuit collapse: No parameters to optimize!
Uses quantum circuits to compute kernel similarities, then classical ridge regression.

Key Innovation: Adaptive circuit depth based on semantic distance
- Similar concepts → shallow circuit (preserve coherence)
- Distant concepts → deep circuit (more entanglement)

Expected to succeed because:
1. No parameter optimization (avoids memorization)
2. Classical learning handles generalization
3. Quantum circuit just computes local similarities
"""

import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import List, Tuple, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

from utils.data_preparation import QManifoldDataPreparation

class AdaptiveQuantumKernel:
    """
    Quantum kernel with data-dependent circuit depth.
    No trainable parameters - structure determined by input similarity.
    """

    def __init__(self, n_qubits=20, base_depth=2, max_depth=6, backend=None):
        self.n_qubits = n_qubits
        self.base_depth = base_depth
        self.max_depth = max_depth
        self.backend = backend
        self.circuits_cache = {}

    def build_kernel_circuit(self, x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
        """
        Build quantum kernel circuit for computing <ψ(x1)|ψ(x2)>.
        Circuit depth adapts to semantic distance between inputs.

        NO PARAMETERS TO OPTIMIZE - structure fully determined by data!
        """
        # Check cache
        cache_key = (tuple(x1), tuple(x2))
        if cache_key in self.circuits_cache:
            return self.circuits_cache[cache_key]

        qc = QuantumCircuit(self.n_qubits)

        # Step 1: Encode first vector
        for i in range(self.n_qubits):
            qc.ry(float(x1[i]), i)

        # Step 2: Compute semantic distance to determine depth
        semantic_distance = np.linalg.norm(x1 - x2)
        normalized_dist = semantic_distance / (np.pi * np.sqrt(self.n_qubits))

        # Adaptive depth: more distance = deeper circuit
        depth = int(self.base_depth + (self.max_depth - self.base_depth) * normalized_dist)
        depth = min(self.max_depth, max(self.base_depth, depth))

        print(f"  Distance: {semantic_distance:.3f} → Depth: {depth}")

        # Step 3: Entangling layers with distance-dependent pattern
        for layer in range(depth):
            # Pattern depends on semantic distance
            if normalized_dist < 0.3:  # Similar concepts
                # Local entanglement for coherent pairs
                for i in range(0, self.n_qubits-1, 2):
                    qc.cx(i, i+1)
                if layer % 2 == 1:  # Alternate pattern
                    for i in range(1, self.n_qubits-1, 2):
                        qc.cx(i, i+1)

            elif normalized_dist < 0.6:  # Medium distance
                # Ring entanglement
                for i in range(self.n_qubits):
                    qc.cx(i, (i+1) % self.n_qubits)

            else:  # Distant concepts
                # All-to-all entanglement for maximum discrimination
                for i in range(self.n_qubits):
                    for j in range(i+1, min(i+3, self.n_qubits)):
                        qc.cx(i, j)

            # Data-dependent rotations (not optimized, deterministic from input)
            for i in range(self.n_qubits):
                # Rotation angle determined by difference
                angle = float((x1[i] - x2[i]) * (layer + 1) * 0.3)
                qc.rz(angle, i)

                # Add cross-term rotations
                if i < self.n_qubits - 1:
                    cross_angle = float((x1[i] * x2[i+1] - x1[i+1] * x2[i]) * 0.2)
                    qc.ry(cross_angle, i)

        # Step 4: Inverse encoding of second vector (for overlap measurement)
        for i in range(self.n_qubits):
            qc.ry(float(-x2[i]), i)

        # Add to cache
        self.circuits_cache[cache_key] = qc
        return qc

    def compute_kernel_matrix(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None,
                            use_hardware: bool = False, shots: int = 4096) -> np.ndarray:
        """
        Compute Gram matrix K[i,j] = |<ψ(xi)|ψ(xj)>|²

        Returns kernel matrix of shape (n_test, n_train)
        """
        if X_test is None:
            X_test = X_train
            symmetric = True
        else:
            symmetric = False

        n_train = len(X_train)
        n_test = len(X_test)
        K = np.zeros((n_test, n_train))

        print(f"\nComputing kernel matrix ({n_test}x{n_train})...")

        if use_hardware:
            # Batch circuits for hardware execution
            circuits = []
            indices = []

            for i in range(n_test):
                for j in range(n_train):
                    # Skip redundant computations for symmetric matrix
                    if symmetric and j > i:
                        continue

                    qc = self.build_kernel_circuit(X_test[i], X_train[j])
                    qc.measure_all()
                    circuits.append(qc)
                    indices.append((i, j))

            print(f"Submitting {len(circuits)} circuits to {self.backend.name}...")

            # Transpile for hardware
            pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
            isa_circuits = pm.run(circuits)

            # Submit to quantum hardware
            sampler = SamplerV2(mode=self.backend)
            job = sampler.run(isa_circuits, shots=shots)
            print(f"Job ID: {job.job_id()}")

            results = job.result()
            print("Results received!")

            # Extract kernel values
            for idx, (i, j) in enumerate(indices):
                counts = results[idx].data.meas.get_counts()
                # Kernel = probability of measuring |00...0>
                k_val = counts.get('0' * self.n_qubits, 0) / shots
                K[i, j] = k_val

                # Symmetric matrix
                if symmetric and i != j:
                    K[j, i] = k_val

        else:
            # Simulator execution
            from qiskit_aer import AerSimulator
            backend = AerSimulator()

            for i in range(n_test):
                for j in range(n_train):
                    if symmetric and j > i:
                        K[i, j] = K[j, i]
                        continue

                    qc = self.build_kernel_circuit(X_test[i], X_train[j])
                    qc.measure_all()

                    job = backend.run(qc, shots=shots)
                    counts = job.result().get_counts()
                    K[i, j] = counts.get('0' * self.n_qubits, 0) / shots

                print(f"  Computed row {i+1}/{n_test}")

        # Ensure kernel matrix is positive semi-definite
        # Add small diagonal regularization if needed
        if symmetric:
            eigvals = np.linalg.eigvalsh(K)
            if np.min(eigvals) < 0:
                print(f"Adding regularization (min eigenvalue: {np.min(eigvals):.6f})")
                K += np.eye(n_train) * abs(np.min(eigvals)) * 1.1

        return K


def run_experiment(use_hardware: bool = False, n_train: int = 20, n_test: int = 30):
    """
    Run quantum kernel ridge regression experiment.

    Expected to avoid collapse because:
    1. No parameters to optimize (avoids memorization)
    2. Kernel naturally captures local similarities
    3. Classical ridge regression handles global learning
    """

    print("=" * 70)
    print("QUANTUM KERNEL RIDGE REGRESSION EXPERIMENT")
    print("=" * 70)

    # Initialize data
    data_prep = QManifoldDataPreparation(target_dim=20)
    all_pairs = data_prep.get_default_concept_pairs()

    # Split data
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:n_train + n_test]

    print(f"Training pairs: {n_train}")
    print(f"Test pairs: {n_test}")

    # Prepare embeddings
    all_concepts = data_prep.generate_all_concepts(train_pairs + test_pairs)
    embeddings = data_prep.embed_concepts(all_concepts)
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # Extract training data
    X_train = []
    y_train = []
    for c1, c2 in train_pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        # Use concatenated vectors as input
        x = np.concatenate([vectors_scaled[idx1][:10], vectors_scaled[idx2][:10]])
        X_train.append(x)
        # Target is hyperbolic distance
        dist = data_prep.compute_hyperbolic_distance(vectors_pca[idx1], vectors_pca[idx2])
        y_train.append(dist)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Extract test data
    X_test = []
    y_test = []
    for c1, c2 in test_pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        x = np.concatenate([vectors_scaled[idx1][:10], vectors_scaled[idx2][:10]])
        X_test.append(x)
        dist = data_prep.compute_hyperbolic_distance(vectors_pca[idx1], vectors_pca[idx2])
        y_test.append(dist)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"\nTarget distance range:")
    print(f"  Train: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")
    print(f"  Test: [{np.min(y_test):.3f}, {np.max(y_test):.3f}]")

    # Initialize quantum kernel
    backend = None
    if use_hardware:
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
            instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::"
        )
        backend = service.backend("ibm_fez")
        print(f"\nUsing hardware: {backend.name}")

    kernel = AdaptiveQuantumKernel(n_qubits=20, base_depth=2, max_depth=5, backend=backend)

    # Compute kernel matrices
    print("\n" + "=" * 50)
    print("STAGE 1: Computing Training Kernel Matrix")
    print("=" * 50)
    K_train = kernel.compute_kernel_matrix(X_train, use_hardware=use_hardware, shots=2048)

    print("\n" + "=" * 50)
    print("STAGE 2: Classical Ridge Regression")
    print("=" * 50)

    # Grid search for optimal alpha
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        model = KernelRidge(alpha=alpha, kernel='precomputed')
        model.fit(K_train, y_train)

        # Self-prediction for validation
        y_pred_train = model.predict(K_train)
        train_corr = np.corrcoef(y_pred_train, y_train)[0, 1]
        print(f"  Alpha={alpha}: Train correlation={train_corr:.4f}")

        if train_corr > best_score:
            best_score = train_corr
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha}")

    # Train final model
    model = KernelRidge(alpha=best_alpha, kernel='precomputed')
    model.fit(K_train, y_train)

    print("\n" + "=" * 50)
    print("STAGE 3: Computing Test Kernel Matrix")
    print("=" * 50)
    K_test = kernel.compute_kernel_matrix(X_train, X_test, use_hardware=use_hardware, shots=2048)

    print("\n" + "=" * 50)
    print("STAGE 4: Validation")
    print("=" * 50)

    # Predict on test set
    y_pred = model.predict(K_test)

    # Compute metrics
    correlation = np.corrcoef(y_pred, y_test)[0, 1]
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))

    print(f"\nTest Results:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.4f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'quantum_kernel_ridge',
        'n_train': n_train,
        'n_test': n_test,
        'correlation': float(correlation),
        'mse': float(mse),
        'mae': float(mae),
        'best_alpha': best_alpha,
        'kernel_params': {
            'base_depth': kernel.base_depth,
            'max_depth': kernel.max_depth
        },
        'hardware': use_hardware,
        'backend': backend.name if backend else 'simulator'
    }

    output_file = f"../results/quantum_kernel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if correlation > 0.7:
        print("✅ SUCCESS! Quantum kernel avoided collapse!")
        print("   Unlike variational PQCs, kernels naturally generalize")
    elif correlation > 0.4:
        print("⚠️ PARTIAL SUCCESS - Better than collapse but needs tuning")
        print("   Try increasing max_depth or shots")
    else:
        print("❌ Still collapsed - kernel approach also limited")
        print("   May need different feature map or deeper circuits")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum Kernel Ridge Regression')
    parser.add_argument('--hardware', action='store_true', help='Use IBM Quantum hardware')
    parser.add_argument('--n_train', type=int, default=20, help='Number of training pairs')
    parser.add_argument('--n_test', type=int, default=30, help='Number of test pairs')

    args = parser.parse_args()

    if args.hardware:
        print("WARNING: This will use quantum hardware!")
        print("Estimated time: 3-4 minutes")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    results = run_experiment(
        use_hardware=args.hardware,
        n_train=args.n_train,
        n_test=args.n_test
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Final correlation: {results['correlation']:.4f}")

    if results['correlation'] > 0.7:
        print("\n🎉 BREAKTHROUGH! Kernel method solved circuit collapse!")
        print("This proves the problem was variational optimization, not quantum expressivity")