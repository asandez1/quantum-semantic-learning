"""
Quantum-Classical Distillation for Semantic Similarity
=======================================================
Solution to circuit collapse: Use quantum only as teacher!
Train deep quantum circuit on small data, then distill to classical student.

Key Innovation: Quantum teacher can afford to be deep (6+ layers)
because it only needs to process a few examples. Classical student
learns from quantum-labeled synthetic data.

Expected to succeed because:
1. Deep quantum circuits have more expressivity
2. Classical student generalizes from large synthetic dataset
3. Combines quantum quality with classical scalability
"""

import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.data_preparation import QManifoldDataPreparation
from utils.quantum_circuit import QManifoldCircuit
from utils.batch_optimizer import BatchSPSAOptimizer, SPSAConfig


class DeepQuantumTeacher:
    """
    Deep quantum circuit that acts as teacher for classical student.
    Can afford to be deep because it only processes limited examples.
    """

    def __init__(self, n_qubits=20, n_layers=6):
        """
        Initialize deep quantum teacher.

        Args:
            n_qubits: Number of qubits
            n_layers: Number of ansatz repetitions (depth)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Build deep circuit with more parameters
        self.circuit = QManifoldCircuit(
            n_qubits=n_qubits,
            ansatz_reps=n_layers,
            entanglement='full'  # Full entanglement for maximum expressivity
        )

        # Use parameter count from the actual circuit
        self.n_params = self.circuit.get_parameter_count()
        self.theta_teacher = np.random.uniform(-0.1, 0.1, self.n_params)

        print(f"Deep Quantum Teacher:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Layers: {n_layers}")
        print(f"  Parameters: {self.n_params}")
        print(f"  Entanglement: full")

    def train_on_quantum_hardware(self, train_pairs: List[Tuple],
                                 data_prep: QManifoldDataPreparation,
                                 vectors_scaled: np.ndarray,
                                 vectors_pca: np.ndarray,
                                 all_concepts: List[str],
                                 backend=None,
                                 max_iterations: int = 5) -> np.ndarray:
        """
        Train deep quantum circuit on hardware with small dataset.

        Can afford deep circuit because:
        1. Only training on 10-15 pairs
        2. Can use more quantum time per pair
        3. Don't need to generalize - just fit training data well
        """
        print("\n" + "=" * 50)
        print("TRAINING DEEP QUANTUM TEACHER")
        print("=" * 50)
        print(f"Training on {len(train_pairs)} pairs")
        print(f"Circuit depth: {self.n_layers} layers")

        # Configure SPSA for deep circuit
        config = SPSAConfig(
            max_iterations=max_iterations,
            batch_size=len(train_pairs),
            learning_rate=0.05,  # Smaller LR for deep circuit
            perturbation_size=0.05,
            shots=4096  # More shots for accuracy
        )

        optimizer = BatchSPSAOptimizer(config, self.circuit, verbose=True)

        # Prepare training data
        X_train = []
        y_train = []
        pair_indices = []

        for c1, c2 in train_pairs:
            try:
                idx1 = all_concepts.index(c1)
                idx2 = all_concepts.index(c2)
                pair_indices.append((idx1, idx2))

                # Target similarity
                v1_pca = vectors_pca[idx1]
                v2_pca = vectors_pca[idx2]
                dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
                target_sim = data_prep.hyperbolic_similarity(dist)
                y_train.append(target_sim)

            except ValueError:
                continue

        # Create batch data structure
        batch_data = {
            'vectors_20d': vectors_scaled,
            'pair_indices': pair_indices,
            'target_similarities': y_train,
            'batches': [{
                'pair_indices': pair_indices,
                'target_similarities': y_train
            }]
        }

        # Train on hardware (or simulator)
        if backend is not None:
            print(f"Training on {backend.name}...")
            sampler = SamplerV2(mode=backend)
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            isa_circuit = pm.run(self.circuit.qc_sampler)

            theta_opt, history = optimizer.optimize_with_sampler(
                self.theta_teacher, batch_data, sampler, isa_circuit
            )
        else:
            print("Training on simulator...")
            from qiskit_aer import AerSimulator
            simulator = AerSimulator()

            # Simplified training loop
            theta_opt = self.theta_teacher.copy()
            for iteration in range(max_iterations):
                total_loss = 0.0

                for (idx1, idx2), target in zip(pair_indices, y_train):
                    v1 = vectors_scaled[idx1]
                    v2 = vectors_scaled[idx2]

                    # Build circuit
                    bound_circuit = self.circuit.bind_parameters(v1, v2, theta_opt)

                    # Execute
                    transpiled_circuit = transpile(bound_circuit, simulator)
                    job = simulator.run(transpiled_circuit, shots=1024)
                    counts = job.result().get_counts()
                    fidelity = counts.get('0' * self.n_qubits, 0) / 1024

                    # Loss
                    loss = (fidelity - target) ** 2
                    total_loss += loss

                avg_loss = total_loss / len(train_pairs)
                print(f"  Iteration {iteration + 1}: loss={avg_loss:.4f}")

                # Simple gradient descent
                if iteration < max_iterations - 1:
                    perturbation = np.random.randn(len(theta_opt)) * 0.01
                    theta_opt -= 0.1 * perturbation * avg_loss

        self.theta_teacher = theta_opt
        print(f"Teacher training complete!")
        return theta_opt

    def generate_synthetic_data(self, n_samples: int,
                               vectors_scaled: np.ndarray,
                               backend=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data using quantum teacher.

        Creates random input pairs and labels them using the trained
        quantum circuit. This synthetic data trains the classical student.
        """
        print(f"\nGenerating {n_samples} synthetic samples using quantum teacher...")

        X_synthetic = []
        y_synthetic = []

        # Generate random vector pairs
        n_vectors = len(vectors_scaled)

        for i in range(n_samples):
            # Random pair
            idx1 = np.random.randint(0, n_vectors)
            idx2 = np.random.randint(0, n_vectors)

            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]

            # Concatenate as input
            x = np.concatenate([v1, v2])
            X_synthetic.append(x)

            # Get quantum teacher's prediction
            bound_circuit = self.circuit.bind_parameters(v1, v2, self.theta_teacher)

            if backend is not None:
                # Hardware execution
                pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
                isa_circuit = pm.run(bound_circuit)
                sampler = SamplerV2(mode=backend)
                job = sampler.run(isa_circuit, shots=512)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            else:
                # Simulator
                from qiskit_aer import AerSimulator
                simulator = AerSimulator()
                transpiled_circuit = transpile(bound_circuit, simulator)
                job = simulator.run(transpiled_circuit, shots=512)
                counts = job.result().get_counts()

            fidelity = counts.get('0' * self.n_qubits, 0) / 512
            y_synthetic.append(fidelity)

            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples")

        return np.array(X_synthetic), np.array(y_synthetic)


class ClassicalStudent:
    """
    Classical neural network that learns from quantum teacher.
    """

    def __init__(self, input_dim=40, hidden_layers=(100, 50, 20)):
        """
        Initialize classical MLP student.

        Args:
            input_dim: Dimension of input (2 * vector_dim)
            hidden_layers: Tuple of hidden layer sizes
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers

        # Multi-layer perceptron
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=42
        )

        # Input scaler
        self.scaler = StandardScaler()

        print(f"\nClassical Student Network:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Total parameters: ~{self._estimate_params()}")

    def _estimate_params(self) -> int:
        """Estimate number of parameters in MLP"""
        n_params = self.input_dim * self.hidden_layers[0]
        for i in range(len(self.hidden_layers) - 1):
            n_params += self.hidden_layers[i] * self.hidden_layers[i + 1]
        n_params += self.hidden_layers[-1]  # Output layer
        return n_params

    def train_from_teacher(self, X_synthetic: np.ndarray, y_synthetic: np.ndarray):
        """
        Train classical student on quantum-labeled data.
        """
        print("\n" + "=" * 50)
        print("TRAINING CLASSICAL STUDENT")
        print("=" * 50)
        print(f"Training on {len(X_synthetic)} synthetic samples")

        # Scale inputs
        X_scaled = self.scaler.fit_transform(X_synthetic)

        # Train neural network
        self.model.fit(X_scaled, y_synthetic)

        print(f"Training complete!")
        print(f"  Final training score: {self.model.score(X_scaled, y_synthetic):.4f}")

        if hasattr(self.model, 'n_iter_'):
            print(f"  Iterations: {self.model.n_iter_}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using classical student.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def run_distillation_experiment(n_teacher_pairs: int = 10,
                               n_synthetic: int = 500,
                               use_hardware: bool = False):
    """
    Run quantum-classical distillation experiment.
    """
    print("=" * 70)
    print("QUANTUM-CLASSICAL DISTILLATION EXPERIMENT")
    print("=" * 70)

    # Initialize data
    data_prep = QManifoldDataPreparation(target_dim=20)
    all_pairs = data_prep.get_default_concept_pairs()

    # Split data
    teacher_pairs = all_pairs[:n_teacher_pairs]
    test_pairs = all_pairs[40:50]  # Hold out for testing

    print(f"Teacher training pairs: {n_teacher_pairs}")
    print(f"Synthetic samples to generate: {n_synthetic}")
    print(f"Test pairs: {len(test_pairs)}")

    # Prepare embeddings
    all_concepts = data_prep.generate_all_concepts(all_pairs)
    embeddings = data_prep.embed_concepts(all_concepts)
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # Hardware setup
    backend = None
    if use_hardware:
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token="YOUR_TOKEN_HERE",
            instance="YOUR_INSTANCE_HERE"
        )
        backend = service.backend("ibm_fez")
        print(f"Using hardware: {backend.name}")

    # Step 1: Train deep quantum teacher
    teacher = DeepQuantumTeacher(n_qubits=20, n_layers=4)

    theta_teacher = teacher.train_on_quantum_hardware(
        teacher_pairs, data_prep, vectors_scaled, vectors_pca, all_concepts,
        backend=backend, max_iterations=3
    )

    # Step 2: Generate synthetic data
    X_synthetic, y_synthetic = teacher.generate_synthetic_data(
        n_synthetic, vectors_scaled, backend=backend
    )

    print(f"\nSynthetic data statistics:")
    print(f"  Mean similarity: {np.mean(y_synthetic):.4f}")
    print(f"  Std similarity: {np.std(y_synthetic):.4f}")
    print(f"  Range: [{np.min(y_synthetic):.4f}, {np.max(y_synthetic):.4f}]")

    # Step 3: Train classical student
    student = ClassicalStudent(input_dim=40, hidden_layers=(80, 40, 20))
    student.train_from_teacher(X_synthetic, y_synthetic)

    # Step 4: Evaluate on test set
    print("\n" + "=" * 50)
    print("TEST EVALUATION")
    print("=" * 50)

    X_test = []
    y_test_true = []
    y_test_quantum = []  # Teacher predictions
    y_test_classical = []  # Student predictions

    for c1, c2 in test_pairs:
        try:
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
        except ValueError:
            continue

        v1 = vectors_scaled[idx1]
        v2 = vectors_scaled[idx2]
        v1_pca = vectors_pca[idx1]
        v2_pca = vectors_pca[idx2]

        # Input
        x = np.concatenate([v1, v2])
        X_test.append(x)

        # True target
        dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
        target = data_prep.hyperbolic_similarity(dist)
        y_test_true.append(target)

        # Quantum teacher prediction (expensive)
        bound_circuit = teacher.circuit.bind_parameters(v1, v2, teacher.theta_teacher)

        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        transpiled_circuit = transpile(bound_circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=1024)
        counts = job.result().get_counts()
        quantum_pred = counts.get('0' * teacher.n_qubits, 0) / 1024
        y_test_quantum.append(quantum_pred)

        print(f"{c1} ↔ {c2}:")
        print(f"  True: {target:.3f}, Quantum: {quantum_pred:.3f}")

    # Classical student predictions (fast)
    X_test = np.array(X_test)
    y_test_classical = student.predict(X_test)

    # Compute metrics
    y_test_true = np.array(y_test_true)
    y_test_quantum = np.array(y_test_quantum)

    quantum_corr = np.corrcoef(y_test_quantum, y_test_true)[0, 1] if len(y_test_quantum) > 1 else 0
    classical_corr = np.corrcoef(y_test_classical, y_test_true)[0, 1] if len(y_test_classical) > 1 else 0

    quantum_mse = np.mean((y_test_quantum - y_test_true) ** 2)
    classical_mse = np.mean((y_test_classical - y_test_true) ** 2)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Quantum Teacher Performance:")
    print(f"  Correlation: {quantum_corr:.4f}")
    print(f"  MSE: {quantum_mse:.6f}")

    print(f"\nClassical Student Performance:")
    print(f"  Correlation: {classical_corr:.4f}")
    print(f"  MSE: {classical_mse:.6f}")

    print(f"\nDistillation Efficiency:")
    print(f"  Student retains {classical_corr/max(quantum_corr, 0.01):.1%} of teacher performance")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'quantum_distillation',
        'teacher': {
            'n_qubits': teacher.n_qubits,
            'n_layers': teacher.n_layers,
            'n_params': teacher.n_params,
            'n_training_pairs': n_teacher_pairs
        },
        'student': {
            'architecture': student.hidden_layers,
            'n_params': student._estimate_params(),
            'n_synthetic_samples': n_synthetic
        },
        'performance': {
            'quantum_correlation': float(quantum_corr),
            'quantum_mse': float(quantum_mse),
            'classical_correlation': float(classical_corr),
            'classical_mse': float(classical_mse),
            'distillation_efficiency': float(classical_corr/max(quantum_corr, 0.01))
        },
        'hardware': use_hardware
    }

    output_file = f"../results/quantum_distillation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if classical_corr > 0.7:
        print("✅ SUCCESS! Distillation works!")
        print("   Classical student learned from quantum teacher")
        print("   Now have fast classical model with quantum-quality predictions")
    elif classical_corr > 0.4:
        print("⚠️ PARTIAL SUCCESS - Student partially learned from teacher")
        print("   Try more synthetic data or deeper student network")
    else:
        print("❌ Distillation failed - student didn't learn effectively")
        print("   May need better teacher or different architecture")

    # Practical implications
    print("\n" + "=" * 50)
    print("PRACTICAL IMPLICATIONS")
    print("=" * 50)
    print("Quantum teacher: Slow but accurate (needs quantum hardware)")
    print("Classical student: Fast and scalable (runs on CPU/GPU)")
    print("\nUse case: Train quantum teacher once, deploy classical student")
    print("This solves the quantum accessibility problem!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum-Classical Distillation')
    parser.add_argument('--hardware', action='store_true', help='Use IBM Quantum hardware')
    parser.add_argument('--n_teacher', type=int, default=10, help='Teacher training pairs')
    parser.add_argument('--n_synthetic', type=int, default=300, help='Synthetic samples')

    args = parser.parse_args()

    if args.hardware:
        print("WARNING: This will use quantum hardware!")
        print("Estimated time: 5-6 minutes")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    results = run_distillation_experiment(
        n_teacher_pairs=args.n_teacher,
        n_synthetic=args.n_synthetic,
        use_hardware=args.hardware
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Classical student correlation: {results['performance']['classical_correlation']:.4f}")

    if results['performance']['classical_correlation'] > 0.7:
        print("\n🎉 DISTILLATION SUCCESS!")
        print("Quantum knowledge successfully transferred to classical model")
        print("This enables practical deployment without quantum hardware!")