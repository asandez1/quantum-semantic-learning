"""
Batch SPSA Optimizer for Q-Manifold (Paper 4)

Implements mini-batch SPSA optimization compatible with:
- Qiskit Runtime Primitives V2 (Sampler + Estimator)
- Batch execution mode (no Session required)
- Shot-efficient gradient estimation

Key Innovation: Broadcasts multiple parameter sets in a single job
to minimize queue latency and maximize 10-minute quantum budget.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import time
import json


@dataclass
class SPSAConfig:
    """Configuration for SPSA optimizer"""
    max_iterations: int = 10
    batch_size: int = 8
    learning_rate: float = 0.1
    perturbation_size: float = 0.1
    shots: int = 2048
    use_estimator: bool = True  # EstimatorV2 vs SamplerV2
    decay_learning_rate: bool = True
    decay_perturbation: bool = True


class BatchSPSAOptimizer:
    """
    Mini-Batch SPSA optimizer for quantum metric refinement.

    Solves the overfitting problem from Paper 3 by averaging gradients
    over multiple concept pairs per iteration.
    """

    def __init__(
        self,
        config: SPSAConfig,
        circuit_builder,
        verbose: bool = True
    ):
        """
        Args:
            config: SPSA configuration
            circuit_builder: QManifoldCircuit instance
            verbose: Whether to print progress
        """
        self.config = config
        self.circuit = circuit_builder
        self.verbose = verbose

        self.n_params = circuit_builder.get_parameter_count()
        self.history = []

        print(f"[Optimizer] Batch SPSA initialized")
        print(f"[Optimizer] Parameters: {self.n_params}")
        print(f"[Optimizer] Batch size: {config.batch_size}")
        print(f"[Optimizer] Mode: {'EstimatorV2' if config.use_estimator else 'SamplerV2'}")

    def _get_spsa_coefficients(self, iteration: int) -> Tuple[float, float]:
        """
        Get learning rate and perturbation size for current iteration.

        Uses recommended SPSA decay schedules:
        - a_k = a / (k + 1 + A)^α
        - c_k = c / (k + 1)^γ

        Args:
            iteration: Current iteration number

        Returns:
            (learning_rate, perturbation_size) tuple
        """
        k = iteration + 1

        if self.config.decay_learning_rate:
            # Recommended: α = 0.602
            a_k = self.config.learning_rate / (k + 10) ** 0.602
        else:
            a_k = self.config.learning_rate

        if self.config.decay_perturbation:
            # Recommended: γ = 0.101
            c_k = self.config.perturbation_size / k ** 0.101
        else:
            c_k = self.config.perturbation_size

        return a_k, c_k

    def _generate_perturbation(self) -> np.ndarray:
        """
        Generate Bernoulli ±1 perturbation vector.

        Returns:
            Perturbation vector
        """
        return 2 * np.random.randint(0, 2, size=self.n_params) - 1

    def _compute_loss(
        self,
        fidelity: float,
        target_similarity: float
    ) -> float:
        """
        Compute contrastive loss.

        Loss = (fidelity - target_similarity)²

        Args:
            fidelity: Measured quantum fidelity
            target_similarity: Ground truth hyperbolic similarity

        Returns:
            Loss value
        """
        return (fidelity - target_similarity) ** 2

    def optimize_with_sampler(
        self,
        theta_init: np.ndarray,
        data: Dict,
        sampler,
        isa_circuit
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run SPSA optimization using SamplerV2.

        Args:
            theta_init: Initial parameter values
            data: Data dictionary from data_preparation
            sampler: SamplerV2 instance
            isa_circuit: Transpiled ISA circuit

        Returns:
            (optimized_theta, history) tuple
        """
        theta = theta_init.copy()
        vectors_20d = data['vectors_20d']
        batches = data['batches']

        for iteration in range(self.config.max_iterations):
            iter_start = time.time()

            # Get SPSA coefficients
            a_k, c_k = self._get_spsa_coefficients(iteration)

            # Generate perturbation
            delta = self._generate_perturbation()
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            # Select batch
            batch = batches[iteration % len(batches)]
            pair_indices = batch['pair_indices']
            target_sims = batch['target_similarities']

            # Build parameter bindings for entire batch
            # We need 2 evaluations per pair (+ and -)
            # Total: batch_size * 2 circuit evaluations
            parameter_values = []

            for (idx1, idx2), target_sim in zip(pair_indices, target_sims):
                x_vec = vectors_20d[idx1]
                y_vec = vectors_20d[idx2]

                # Bind theta_plus
                params_plus = np.concatenate([x_vec, theta_plus, y_vec])
                parameter_values.append(params_plus)

                # Bind theta_minus
                params_minus = np.concatenate([x_vec, theta_minus, y_vec])
                parameter_values.append(params_minus)

            # Convert to 2D NumPy array for proper batching
            parameter_values = np.array(parameter_values)  # Shape: (batch_size*2, num_params)

            # Submit as single PUB (Primitive Unified Bloc)
            pub = (isa_circuit, parameter_values)

            if self.verbose:
                print(f"\n[Iter {iteration + 1}/{self.config.max_iterations}] "
                      f"Submitting {len(parameter_values)} circuits...")

            # Execute
            job = sampler.run([pub], shots=self.config.shots)
            result = job.result()

            # Process results
            pub_result = result[0]

            # Compute gradient
            grad_accum = np.zeros(self.n_params)
            total_loss = 0.0

            for i in range(len(pair_indices)):
                # Get counts for each parameter set in the batch
                counts_plus = pub_result.data.meas.get_counts(2 * i)
                counts_minus = pub_result.data.meas.get_counts(2 * i + 1)

                # Extract fidelities
                from utils.quantum_circuit import FidelityMeasurement
                fid_plus = FidelityMeasurement.fidelity_from_counts(
                    counts_plus, self.circuit.n_qubits
                )
                fid_minus = FidelityMeasurement.fidelity_from_counts(
                    counts_minus, self.circuit.n_qubits
                )

                # Compute losses
                target_sim = target_sims[i]
                loss_plus = self._compute_loss(fid_plus, target_sim)
                loss_minus = self._compute_loss(fid_minus, target_sim)

                # SPSA gradient estimate
                grad_i = (loss_plus - loss_minus) / (2 * c_k) * delta
                grad_accum += grad_i

                total_loss += (loss_plus + loss_minus) / 2.0

            # Average gradient and loss
            avg_grad = grad_accum / len(pair_indices)
            avg_loss = total_loss / len(pair_indices)

            # Update parameters
            theta = theta - a_k * avg_grad

            # Log
            iter_time = time.time() - iter_start
            history_entry = {
                'iteration': iteration + 1,
                'loss': float(avg_loss),
                'learning_rate': a_k,
                'perturbation': c_k,
                'time': iter_time
            }
            self.history.append(history_entry)

            if self.verbose:
                print(f"[Iter {iteration + 1}] Loss: {avg_loss:.6f}, "
                      f"Time: {iter_time:.1f}s")

        return theta, self.history

    def optimize_with_estimator(
        self,
        theta_init: np.ndarray,
        data: Dict,
        estimator,
        isa_circuit,
        observable
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run SPSA optimization using EstimatorV2.

        More shot-efficient than SamplerV2 for fidelity measurement.

        Args:
            theta_init: Initial parameter values
            data: Data dictionary from data_preparation
            estimator: EstimatorV2 instance
            isa_circuit: Transpiled ISA circuit
            observable: SparsePauliOp observable

        Returns:
            (optimized_theta, history) tuple
        """
        theta = theta_init.copy()
        vectors_20d = data['vectors_20d']
        batches = data['batches']

        for iteration in range(self.config.max_iterations):
            iter_start = time.time()

            # Get SPSA coefficients
            a_k, c_k = self._get_spsa_coefficients(iteration)

            # Generate perturbation
            delta = self._generate_perturbation()
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            # Select batch
            batch = batches[iteration % len(batches)]
            pair_indices = batch['pair_indices']
            target_sims = batch['target_similarities']

            # Build PUBs for Estimator
            # Estimator uses (circuit, observable, parameter_values) format
            parameter_values = []

            for (idx1, idx2), target_sim in zip(pair_indices, target_sims):
                x_vec = vectors_20d[idx1]
                y_vec = vectors_20d[idx2]

                # Bind theta_plus
                params_plus = np.concatenate([x_vec, theta_plus, y_vec])
                parameter_values.append(params_plus)

                # Bind theta_minus
                params_minus = np.concatenate([x_vec, theta_minus, y_vec])
                parameter_values.append(params_minus)

            # Convert to 2D NumPy array for proper batching
            parameter_values = np.array(parameter_values)  # Shape: (batch_size*2, num_params)

            # Create PUB
            pub = (isa_circuit, observable, parameter_values)

            if self.verbose:
                print(f"\n[Iter {iteration + 1}/{self.config.max_iterations}] "
                      f"Submitting {len(parameter_values)} evaluations...")

            # Execute
            job = estimator.run([pub], precision=0.05)  # 5% precision target
            result = job.result()

            # Process results
            pub_result = result[0]
            expectation_values = pub_result.data.evs

            # Compute gradient
            grad_accum = np.zeros(self.n_params)
            total_loss = 0.0

            for i in range(len(pair_indices)):
                exp_plus = expectation_values[2 * i]
                exp_minus = expectation_values[2 * i + 1]

                # Convert to fidelity
                from utils.quantum_circuit import FidelityMeasurement
                fid_plus = FidelityMeasurement.fidelity_from_expectation(
                    exp_plus, self.circuit.n_qubits, method='avg_z'
                )
                fid_minus = FidelityMeasurement.fidelity_from_expectation(
                    exp_minus, self.circuit.n_qubits, method='avg_z'
                )

                # Compute losses
                target_sim = target_sims[i]
                loss_plus = self._compute_loss(fid_plus, target_sim)
                loss_minus = self._compute_loss(fid_minus, target_sim)

                # SPSA gradient estimate
                grad_i = (loss_plus - loss_minus) / (2 * c_k) * delta
                grad_accum += grad_i

                total_loss += (loss_plus + loss_minus) / 2.0

            # Average gradient and loss
            avg_grad = grad_accum / len(pair_indices)
            avg_loss = total_loss / len(pair_indices)

            # Update parameters
            theta = theta - a_k * avg_grad

            # Log
            iter_time = time.time() - iter_start
            history_entry = {
                'iteration': iteration + 1,
                'loss': float(avg_loss),
                'learning_rate': a_k,
                'perturbation': c_k,
                'time': iter_time
            }
            self.history.append(history_entry)

            if self.verbose:
                print(f"[Iter {iteration + 1}] Loss: {avg_loss:.6f}, "
                      f"Time: {iter_time:.1f}s")

        return theta, self.history

    def save_history(self, filepath: str):
        """Save optimization history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[Optimizer] History saved to {filepath}")


if __name__ == "__main__":
    # Test optimizer construction
    print("=" * 60)
    print("Testing Batch SPSA Optimizer")
    print("=" * 60)

    from quantum_circuit import QManifoldCircuit

    circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2)

    config = SPSAConfig(
        max_iterations=5,
        batch_size=4,
        learning_rate=0.1,
        perturbation_size=0.1,
        shots=1024,
        use_estimator=True
    )

    optimizer = BatchSPSAOptimizer(config, circuit, verbose=True)

    print(f"\n{optimizer.config}")
    print(f"Parameters to optimize: {optimizer.n_params}")

    print("\n✓ Batch SPSA optimizer ready!")
