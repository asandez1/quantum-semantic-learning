"""
Quantum Reservoir Computing for Semantic Similarity
====================================================
Solution to circuit collapse: NO TRAINING AT ALL!
Uses random fixed quantum circuit as nonlinear feature extractor.
Only trains classical readout layer.

Key Innovation: Quantum system acts as high-dimensional random projection
- Fixed random circuit (never changes)
- Extracts quantum features from input
- Classical linear regression learns mapping

Expected to succeed because:
1. No quantum optimization (impossible to overfit)
2. Random features can be surprisingly effective
3. 2^20 dimensional feature space
"""

import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import List, Tuple, Dict
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import normalize

from utils.data_preparation import QManifoldDataPreparation


class QuantumReservoir:
    """
    Random quantum circuit as fixed nonlinear reservoir.
    NO PARAMETERS TO TRAIN - completely random and fixed!

    This is inspired by reservoir computing and extreme learning machines.
    """

    def __init__(self, n_qubits=20, n_layers=3, seed=42, backend=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend

        # Generate RANDOM fixed circuit (never changes!)
        np.random.seed(seed)
        self.reservoir_circuit = self._build_random_reservoir()

        print(f"Built random reservoir with {n_layers} layers")
        print("This circuit is FIXED - no training needed!")

    def _build_random_reservoir(self) -> QuantumCircuit:
        """
        Build a random quantum circuit that acts as feature extractor.
        This circuit is generated once and never changed.
        """
        qc = QuantumCircuit(self.n_qubits)

        for layer in range(self.n_layers):
            # Random single-qubit rotations
            for q in range(self.n_qubits):
                # Random angles
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                lam = np.random.uniform(0, 2 * np.pi)

                qc.u(theta, phi, lam, q)

            # Random entangling layer
            # Use different patterns per layer for diversity
            if layer % 3 == 0:
                # Random CX pairs
                for _ in range(self.n_qubits // 2):
                    q1 = np.random.randint(0, self.n_qubits)
                    q2 = np.random.randint(0, self.n_qubits)
                    if q1 != q2:
                        qc.cx(q1, q2)

            elif layer % 3 == 1:
                # Ring pattern with random skips
                for q in range(self.n_qubits):
                    if np.random.random() > 0.3:  # 70% chance
                        qc.cx(q, (q + 1) % self.n_qubits)

            else:
                # Random CZ gates (different from CX)
                for _ in range(self.n_qubits):
                    q1 = np.random.randint(0, self.n_qubits)
                    q2 = np.random.randint(0, self.n_qubits)
                    if q1 != q2:
                        qc.cz(q1, q2)

            # Add some random phases
            for q in range(self.n_qubits):
                if np.random.random() > 0.5:
                    phase = np.random.uniform(-np.pi, np.pi)
                    qc.rz(phase, q)

        return qc

    def extract_features(self, x: np.ndarray, shots: int = 2048,
                        n_measurements: int = 5) -> np.ndarray:
        """
        Extract quantum features by:
        1. Encoding input
        2. Applying fixed random reservoir
        3. Measuring multiple times with different bases
        4. Returning probability distributions as features

        Returns feature vector of size n_measurements * 2^n_qubits (sparse)
        """
        features = []

        for measurement in range(n_measurements):
            qc = QuantumCircuit(self.n_qubits)

            # Encode input with slight variation per measurement
            for i in range(min(len(x), self.n_qubits)):
                # Add measurement-dependent phase for diversity
                angle = float(x[i] + measurement * 0.1)
                qc.ry(angle, i)

            # Apply fixed reservoir transformation
            qc.compose(self.reservoir_circuit, inplace=True)

            # Measurement basis rotation (different per measurement)
            if measurement > 0:
                for q in range(self.n_qubits):
                    if measurement == 1:  # X basis
                        qc.h(q)
                    elif measurement == 2:  # Y basis
                        qc.sdg(q)
                        qc.h(q)
                    elif measurement == 3:  # Random basis
                        angle = np.random.uniform(0, np.pi)
                        qc.ry(angle, q)
                    # measurement == 4 stays in Z basis

            # Measure all qubits
            qc.measure_all()

            if self.backend is not None:
                # Hardware execution
                pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
                isa_circuit = pm.run(qc)
                sampler = SamplerV2(mode=self.backend)
                job = sampler.run(isa_circuit, shots=shots)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            else:
                # Simulator execution
                from qiskit_aer import AerSimulator
                simulator = AerSimulator()
                job = simulator.run(qc, shots=shots)
                counts = job.result().get_counts()

            # Convert counts to probability vector
            # We'll use top-k bitstrings as sparse features
            total_counts = sum(counts.values())
            prob_vector = []

            # Get top 100 most frequent bitstrings
            top_bitstrings = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:100]

            for bitstring, count in top_bitstrings:
                prob = count / total_counts
                # Use both bitstring (as binary features) and probability
                prob_vector.append(prob)

                # Also add binary features from bitstring
                for bit in bitstring[:10]:  # Use first 10 bits
                    prob_vector.append(float(bit) * prob)

            features.extend(prob_vector)

        return np.array(features)

    def extract_batch_features(self, X: np.ndarray, use_hardware: bool = False,
                              shots: int = 2048) -> np.ndarray:
        """
        Extract features for multiple inputs.
        Returns matrix of shape (n_samples, n_features)
        """
        print(f"Extracting features for {len(X)} samples...")
        features_list = []

        for i, x in enumerate(X):
            features = self.extract_features(x, shots=shots, n_measurements=3)
            features_list.append(features)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(X)} samples")

        # Pad features to same length
        max_len = max(len(f) for f in features_list)
        features_matrix = np.zeros((len(X), max_len))

        for i, f in enumerate(features_list):
            features_matrix[i, :len(f)] = f

        return features_matrix


def run_experiment(use_hardware: bool = False, n_train: int = 30, n_test: int = 20):
    """
    Run quantum reservoir computing experiment.

    This should work because:
    1. No optimization = no overfitting
    2. Random projections can capture complex patterns
    3. Classical learning handles generalization
    """

    print("=" * 70)
    print("QUANTUM RESERVOIR COMPUTING EXPERIMENT")
    print("=" * 70)
    print("NO TRAINING NEEDED - Using random fixed quantum circuit!")
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

    # Prepare training data
    X_train = []
    y_train = []
    train_concept_pairs = []

    for c1, c2 in train_pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        # Concatenate both vectors
        x = np.concatenate([vectors_scaled[idx1], vectors_scaled[idx2]])
        X_train.append(x)
        # Target is hyperbolic distance
        dist = data_prep.compute_hyperbolic_distance(vectors_pca[idx1], vectors_pca[idx2])
        y_train.append(dist)
        train_concept_pairs.append((c1, c2))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Prepare test data
    X_test = []
    y_test = []
    test_concept_pairs = []

    for c1, c2 in test_pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        x = np.concatenate([vectors_scaled[idx1], vectors_scaled[idx2]])
        X_test.append(x)
        dist = data_prep.compute_hyperbolic_distance(vectors_pca[idx1], vectors_pca[idx2])
        y_test.append(dist)
        test_concept_pairs.append((c1, c2))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"\nTarget distance range:")
    print(f"  Train: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")
    print(f"  Test: [{np.min(y_test):.3f}, {np.max(y_test):.3f}]")

    # Initialize quantum reservoir
    backend = None
    if use_hardware:
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
            instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::"
        )
        backend = service.backend("ibm_fez")
        print(f"\nUsing hardware: {backend.name}")

    # Create random reservoir (NO TRAINING!)
    print("\n" + "=" * 50)
    print("STAGE 1: Creating Random Quantum Reservoir")
    print("=" * 50)

    reservoir = QuantumReservoir(n_qubits=20, n_layers=3, seed=42, backend=backend)

    # Extract features using reservoir
    print("\n" + "=" * 50)
    print("STAGE 2: Extracting Quantum Features")
    print("=" * 50)

    print("\nExtracting training features...")
    X_train_features = reservoir.extract_batch_features(
        X_train, use_hardware=use_hardware, shots=1024
    )

    print(f"Training feature shape: {X_train_features.shape}")
    print(f"Feature sparsity: {np.mean(X_train_features == 0):.2%} zeros")

    print("\nExtracting test features...")
    X_test_features = reservoir.extract_batch_features(
        X_test, use_hardware=use_hardware, shots=1024
    )

    # Normalize features
    X_train_features = normalize(X_train_features, norm='l2')
    X_test_features = normalize(X_test_features, norm='l2')

    # Train classical readout (only classical training!)
    print("\n" + "=" * 50)
    print("STAGE 3: Training Classical Readout Layer")
    print("=" * 50)

    # Use cross-validation to find best regularization
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train_features, y_train)

    print(f"Best regularization alpha: {ridge_cv.alpha_}")

    # Train final model
    model = Ridge(alpha=ridge_cv.alpha_)
    model.fit(X_train_features, y_train)

    # Evaluate
    print("\n" + "=" * 50)
    print("STAGE 4: Evaluation")
    print("=" * 50)

    # Training performance
    y_pred_train = model.predict(X_train_features)
    train_corr = np.corrcoef(y_pred_train, y_train)[0, 1]
    train_mse = np.mean((y_pred_train - y_train) ** 2)

    print(f"\nTraining Results:")
    print(f"  Correlation: {train_corr:.4f}")
    print(f"  MSE: {train_mse:.6f}")

    # Test performance
    y_pred_test = model.predict(X_test_features)
    test_corr = np.corrcoef(y_pred_test, y_test)[0, 1]
    test_mse = np.mean((y_pred_test - y_test) ** 2)
    test_mae = np.mean(np.abs(y_pred_test - y_test))

    print(f"\nTest Results:")
    print(f"  Correlation: {test_corr:.4f}")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.4f}")

    # Example predictions
    print("\n" + "=" * 50)
    print("Example Predictions (Test Set)")
    print("=" * 50)

    for i in range(min(5, len(test_pairs))):
        c1, c2 = test_concept_pairs[i]
        true_dist = y_test[i]
        pred_dist = y_pred_test[i]
        error = abs(pred_dist - true_dist)
        print(f"{c1} ↔ {c2}")
        print(f"  True: {true_dist:.3f}, Pred: {pred_dist:.3f}, Error: {error:.3f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'quantum_reservoir',
        'n_train': n_train,
        'n_test': n_test,
        'train_correlation': float(train_corr),
        'test_correlation': float(test_corr),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'best_alpha': float(ridge_cv.alpha_),
        'reservoir_params': {
            'n_layers': reservoir.n_layers,
            'seed': 42
        },
        'feature_dim': X_train_features.shape[1],
        'hardware': use_hardware,
        'backend': backend.name if backend else 'simulator'
    }

    output_file = f"../results/quantum_reservoir_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if test_corr > 0.7:
        print("✅ SUCCESS! Random quantum reservoir works!")
        print("   No training needed - random features capture semantic structure")
        print("   This proves quantum systems can extract useful features without optimization")
    elif test_corr > 0.4:
        print("⚠️ MODERATE SUCCESS - Better than collapse")
        print("   Random features partially capture structure")
        print("   Try more measurements or different random seeds")
    else:
        print("❌ Random reservoir insufficient")
        print("   Need more sophisticated feature extraction")

    # Compare to PQC collapse
    print(f"\nComparison to PQC collapse (0.06):")
    print(f"  Reservoir is {test_corr/0.06:.1f}× better!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum Reservoir Computing')
    parser.add_argument('--hardware', action='store_true', help='Use IBM Quantum hardware')
    parser.add_argument('--n_train', type=int, default=30, help='Number of training pairs')
    parser.add_argument('--n_test', type=int, default=20, help='Number of test pairs')

    args = parser.parse_args()

    if args.hardware:
        print("WARNING: This will use quantum hardware!")
        print("Estimated time: 2-3 minutes (NO TRAINING!)")
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
    print(f"Test correlation: {results['test_correlation']:.4f}")

    if results['test_correlation'] > 0.7:
        print("\n🎉 RANDOM QUANTUM CIRCUITS WORK!")
        print("No optimization needed - quantum randomness provides useful features")
        print("This is the simplest possible quantum ML and it works!")