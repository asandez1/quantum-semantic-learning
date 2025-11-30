"""
Experiment 3: Hyperbolic Contrastive Learning

Objective: Achieve >0.90 preservation by aligning quantum fidelity with hyperbolic geometry
Target: Pass all four Semantic Preservation Benchmark tests

This is the HIGHEST PRIORITY experiment as it directly targets the 0.90 threshold
by solving the fundamental Hilbert space mismatch problem.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
    Session
)
from qiskit_algorithms.optimizers import SPSA, COBYLA
from typing import List, Dict, Tuple, Callable
import json
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for hyperbolic training"""
    n_qubits: int = 12
    ansatz_reps: int = 2
    optimizer: str = 'SPSA'  # or 'COBYLA'
    max_iterations: int = 50
    shots: int = 4096
    learning_rate: float = 0.1
    hyperbolic_scale: float = 1.0


class HyperbolicContrastiveLearner:
    """
    Implements Siamese Quantum Neural Network with hyperbolic geometry alignment.

    Key Innovation: Trains the quantum circuit to minimize the difference between
    quantum fidelity measurements and classical hyperbolic distances, thereby
    solving the Euclidean-Hilbert space mismatch.
    """

    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: Training configuration parameters
        """
        self.config = config
        self.ansatz = self._build_ansatz()
        self.training_history = []

    def _build_ansatz(self) -> QuantumCircuit:
        """
        Builds the variational quantum circuit (ansatz).

        Uses RealAmplitudes with pairwise entanglement for trainability.

        Returns:
            Parameterized quantum circuit
        """
        # RealAmplitudes is a standard ansatz that's hardware-efficient
        ansatz = RealAmplitudes(
            self.config.n_qubits,
            reps=self.config.ansatz_reps,
            entanglement='pairwise',
            insert_barriers=True
        )

        # Alternative: EfficientSU2 (more expressive but deeper)
        # ansatz = EfficientSU2(
        #     self.config.n_qubits,
        #     reps=self.config.ansatz_reps,
        #     entanglement='linear'
        # )

        return ansatz

    def encode_embedding(
        self,
        embedding: np.ndarray,
        method: str = 'amplitude'
    ) -> QuantumCircuit:
        """
        Encodes a semantic embedding as initial quantum state.

        Args:
            embedding: Classical embedding vector (384-dim typically)
            method: Encoding method ('amplitude', 'angle', or 'hybrid')

        Returns:
            Quantum circuit with encoded state
        """
        n_qubits = self.config.n_qubits
        qc = QuantumCircuit(n_qubits)

        if method == 'amplitude':
            # Normalize and truncate to 2^n dimensions
            state_size = 2**n_qubits
            state_vector = embedding[:state_size]

            # Pad if necessary
            if len(state_vector) < state_size:
                state_vector = np.pad(state_vector, (0, state_size - len(state_vector)))

            # Initialize quantum state (let Qiskit handle normalization)
            qc.initialize(state_vector, range(n_qubits), normalize=True)

        elif method == 'angle':
            # Encode into rotation angles
            angles = embedding[:n_qubits]
            angles = (angles / np.max(np.abs(angles) + 1e-10)) * np.pi

            for i, angle in enumerate(angles):
                qc.ry(angle, i)

        elif method == 'hybrid':
            # Combine both methods
            # First half: amplitude encoding on subset
            # Second half: angle encoding
            half = n_qubits // 2

            # Amplitude on first half
            state_size = 2**half
            norm_emb = embedding[:state_size]
            norm_emb = norm_emb / (np.linalg.norm(norm_emb) + 1e-10)
            qc.initialize(norm_emb, range(half))

            # Angle on second half
            angles = embedding[state_size:state_size + half]
            angles = (angles / np.max(np.abs(angles) + 1e-10)) * np.pi
            for i, angle in enumerate(angles):
                qc.ry(angle, half + i)

        return qc

    @staticmethod
    def hyperbolic_distance(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> float:
        """
        Computes hyperbolic distance in Poincaré disk model.

        Args:
            u, v: Points in Euclidean space
            c: Curvature parameter (default: 1.0)

        Returns:
            Hyperbolic distance
        """
        # Project to Poincaré disk if not already
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)

        # Normalize to unit ball if outside
        if u_norm >= 1.0:
            u = u / (u_norm + 1e-3)
        if v_norm >= 1.0:
            v = v / (v_norm + 1e-3)

        # Hyperbolic distance formula
        diff_squared = np.sum((u - v)**2)
        denom = (1 - np.sum(u**2)) * (1 - np.sum(v**2))

        # Avoid numerical issues
        if denom <= 1e-10:
            return 0.0

        # arccosh(1 + 2 * delta / ((1-||u||²)(1-||v||²)))
        delta = 2.0 * diff_squared / (denom + 1e-10)
        distance = np.arccosh(1.0 + delta)

        return distance * np.sqrt(c)

    def hyperbolic_similarity(self, distance: float) -> float:
        """
        Converts hyperbolic distance to similarity score [0, 1].

        Args:
            distance: Hyperbolic distance

        Returns:
            Similarity score (higher = more similar)
        """
        # Use exponential decay: sim = exp(-distance)
        # Scale to [0, 1] range
        similarity = np.exp(-distance / self.config.hyperbolic_scale)
        return similarity

    def build_contrastive_cost_function(
        self,
        concept_pair: Tuple[np.ndarray, np.ndarray],
        target_distance: float,
        estimator: Estimator,
        use_simulator: bool = True,
        backend=None
    ) -> Callable:
        """
        Builds the cost function for contrastive learning.

        Cost = |Quantum_Fidelity - Hyperbolic_Similarity|²

        Args:
            concept_pair: (source_embedding, target_embedding)
            target_distance: Classical hyperbolic distance between concepts
            estimator: Qiskit Estimator primitive
            use_simulator: Whether to use simulator or hardware
            backend: IBM backend instance (required if use_simulator=False)

        Returns:
            Cost function callable
        """
        source_emb, target_emb = concept_pair
        target_similarity = self.hyperbolic_similarity(target_distance)

        # Build the full circuit: encoding + ansatz
        # Use angle encoding for hardware (amplitude needs 'initialize' which isn't supported)
        encoding_method = 'angle' if not use_simulator else 'amplitude'
        encoding_circuit = self.encode_embedding(source_emb, method=encoding_method)
        full_circuit = encoding_circuit.compose(self.ansatz)

        # Observable: Global Z magnetization (proxy for state overlap)
        observable = SparsePauliOp("Z" * self.config.n_qubits)

        # Transpile to ISA if running on hardware
        if not use_simulator and backend is not None:
            # Optimization level 1 is usually sufficient and faster for training loops
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            full_circuit = pm.run(full_circuit)

            # Map observable to physical qubits if layout changed
            if full_circuit.layout is not None:
                observable = observable.apply_layout(full_circuit.layout)

        def cost_function(params: np.ndarray) -> float:
            """
            Evaluates the cost for given parameters.

            Args:
                params: Variational parameters

            Returns:
                Cost value
            """
            # Check if using V1 (Aer) or V2 (IBM Runtime) estimator
            if use_simulator:
                # V1 API for AerEstimator
                bound_circuit = full_circuit.assign_parameters(params)
                job = estimator.run(bound_circuit, observable)
                result = job.result()
                expectation = result.values[0]
            else:
                # V2 API for IBM Runtime EstimatorV2 (uses PUBs)
                # PUB format: (circuit, observables, parameter_values)
                pub = (full_circuit, [observable], [params])
                job = estimator.run([pub])
                result = job.result()
                # V2 result format
                expectation = result[0].data.evs[0]

            # Map expectation [-1, 1] to fidelity-like metric [0, 1]
            quantum_fidelity = (expectation + 1.0) / 2.0

            # Compute loss: MSE between quantum fidelity and target similarity
            loss = (quantum_fidelity - target_similarity)**2

            # Track history
            self.training_history.append({
                'params': params.tolist(),
                'quantum_fidelity': float(quantum_fidelity),
                'target_similarity': float(target_similarity),
                'loss': float(loss)
            })

            return loss

        return cost_function

    def train_on_concept_pair(
        self,
        source_emb: np.ndarray,
        target_emb: np.ndarray,
        backend_name: str = None,
        use_simulator: bool = True
    ) -> Dict:
        """
        Trains the quantum circuit on a single concept pair.

        Args:
            source_emb: Source concept embedding
            target_emb: Target concept embedding
            backend_name: IBM backend name (if not using simulator)
            use_simulator: Whether to use simulator or hardware

        Returns:
            Training results dictionary
        """
        # Compute classical hyperbolic distance
        target_distance = self.hyperbolic_distance(
            source_emb[:self.config.n_qubits],
            target_emb[:self.config.n_qubits]
        )

        print(f"Training on concept pair:")
        print(f"  Classical hyperbolic distance: {target_distance:.4f}")
        print(f"  Target similarity: {self.hyperbolic_similarity(target_distance):.4f}")

        # Set up IBM Quantum or simulator
        backend = None
        if use_simulator:
            from qiskit_aer.primitives import Estimator as AerEstimator

            estimator = AerEstimator()
        else:
            # Use real quantum hardware (EstimatorV2 API)
            service = QiskitRuntimeService()
            backend = service.backend(backend_name)
            # EstimatorV2 initialization - no Session for free plan
            estimator = Estimator(mode=backend)
            estimator.options.default_shots = self.config.shots

        # Build cost function
        cost_func = self.build_contrastive_cost_function(
            concept_pair=(source_emb, target_emb),
            target_distance=target_distance,
            estimator=estimator,
            use_simulator=use_simulator,
            backend=backend
        )

        # Initialize parameters randomly
        initial_params = 2 * np.pi * np.random.random(self.ansatz.num_parameters)

        # Choose optimizer
        if self.config.optimizer == 'SPSA':
            optimizer = SPSA(
                maxiter=self.config.max_iterations,
                learning_rate=self.config.learning_rate,
                perturbation=self.config.learning_rate / 2.0  # Typical choice
            )
        elif self.config.optimizer == 'COBYLA':
            optimizer = COBYLA(maxiter=self.config.max_iterations)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Run optimization
        print(f"\nStarting optimization ({self.config.optimizer})...")
        result = optimizer.minimize(
            fun=cost_func,
            x0=initial_params
        )

        print(f"\nOptimization complete!")
        print(f"  Final loss: {result.fun:.6f}")
        print(f"  Iterations: {result.nfev}")

        return {
            'optimal_params': result.x.tolist(),
            'final_loss': float(result.fun),
            'iterations': result.nfev,
            'training_history': self.training_history,
            'target_distance': float(target_distance)
        }

    def evaluate_preservation(
        self,
        concept_pairs: List[Tuple[np.ndarray, np.ndarray]],
        trained_params: np.ndarray,
        backend_name: str = None,
        use_simulator: bool = True
    ) -> Dict[str, float]:
        """
        Evaluates the trained circuit on the Semantic Preservation Benchmark.

        Tests:
        1. Fidelity-Cosine Correlation
        2. Topological Persistence
        3. Community Structure NMI
        4. Geodesic Path Coherence

        Args:
            concept_pairs: List of (source, target) embedding pairs
            trained_params: Optimized circuit parameters
            backend_name: IBM backend (if not simulator)
            use_simulator: Use simulator or hardware

        Returns:
            Preservation scores dictionary
        """
        # TODO: Implement full benchmark evaluation
        # For now, return placeholder

        # This would integrate with paper2/quantum_baselines.py benchmark suite

        print(f"\nEvaluating on {len(concept_pairs)} concept pairs...")

        results = {
            'fidelity_correlation': 0.0,
            'topology_preservation': 0.0,
            'community_nmi': 0.0,
            'path_coherence': 0.0,
            'overall_score': 0.0
        }

        # TODO: Implement actual evaluation logic

        return results


def main():
    """Example training workflow"""

    # Configuration
    config = TrainingConfig(
        n_qubits=10,
        ansatz_reps=2,
        optimizer='SPSA',
        max_iterations=50,
        shots=4096
    )

    # Initialize learner
    learner = HyperbolicContrastiveLearner(config)

    # Example: Create dummy concept embeddings
    np.random.seed(42)
    source_emb = np.random.randn(384)
    target_emb = np.random.randn(384)

    # Train on simulator first (for testing)
    results = learner.train_on_concept_pair(
        source_emb=source_emb,
        target_emb=target_emb,
        use_simulator=True
    )

    print(f"\nTraining Results:")
    print(f"  Final Loss: {results['final_loss']:.6f}")
    print(f"  Iterations: {results['iterations']}")

    # Save results
    with open('hyperbolic_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to hyperbolic_training_results.json")

    # To run on hardware (uncomment when ready):
    # results_hw = learner.train_on_concept_pair(
    #     source_emb=source_emb,
    #     target_emb=target_emb,
    #     backend_name='ibm_sherbrooke',
    #     use_simulator=False
    # )


if __name__ == "__main__":
    main()
