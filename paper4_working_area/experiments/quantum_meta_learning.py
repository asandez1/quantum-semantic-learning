"""
Quantum Meta-Learning for Few-Shot Semantic Adaptation
=======================================================
Solution to circuit collapse: Learn to learn!
Uses MAML-style meta-learning to find initialization that enables rapid adaptation.

Key Innovation: Instead of learning one model for all data,
learn an initialization that can quickly adapt to specific semantic domains.

Expected to succeed because:
1. Few-shot adaptation avoids overfitting
2. Domain-specific fine-tuning preserves structure
3. Meta-parameters capture general semantic patterns
"""

import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import argparse
import copy

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile

from utils.data_preparation import QManifoldDataPreparation
from utils.quantum_circuit import QManifoldCircuit


class Task:
    """Represents a meta-learning task (semantic domain)"""

    def __init__(self, name: str, concepts: List[str], pairs: List[Tuple[str, str]]):
        self.name = name
        self.concepts = concepts
        self.pairs = pairs
        self.support_pairs = []  # For few-shot training
        self.query_pairs = []    # For evaluation


class QuantumMAML:
    """
    Model-Agnostic Meta-Learning for quantum circuits.
    Learns initialization that enables few-shot adaptation to new semantic domains.
    """

    def __init__(self, n_qubits=20, inner_lr=0.1, outer_lr=0.01, inner_steps=3):
        self.n_qubits = n_qubits
        self.inner_lr = inner_lr  # Learning rate for task adaptation
        self.outer_lr = outer_lr  # Learning rate for meta-parameters
        self.inner_steps = inner_steps  # Gradient steps per task

        # Initialize meta-parameters (shared across all tasks)
        self.theta_meta = np.random.uniform(-0.1, 0.1, 60)

        # Build quantum circuit template
        self.circuit = QManifoldCircuit(n_qubits=n_qubits, ansatz_reps=2)

        print(f"Initialized MAML with {len(self.theta_meta)} meta-parameters")
        print(f"Inner loop: {inner_steps} steps at lr={inner_lr}")
        print(f"Outer loop: lr={outer_lr}")

    def create_tasks(self, all_pairs: List[Tuple[str, str]]) -> List[Task]:
        """
        Create meta-learning tasks from semantic domains.
        Each task is a coherent subset (animals, vehicles, plants, etc.)
        """
        tasks = []

        # Define semantic domains
        domains = {
            'animals': {
                'animal', 'mammal', 'dog', 'cat', 'bird', 'fish',
                'poodle', 'siamese', 'sparrow', 'salmon', 'pet', 'reptile'
            },
            'vehicles': {
                'vehicle', 'car', 'sedan', 'truck', 'bicycle', 'airplane',
                'boat', 'motorcycle', 'bus', 'train'
            },
            'plants': {
                'plant', 'tree', 'flower', 'oak', 'pine', 'rose',
                'grass', 'bush', 'vegetable', 'fruit'
            },
            'tools': {
                'tool', 'hammer', 'screwdriver', 'saw', 'drill',
                'wrench', 'pliers', 'knife', 'scissors'
            },
            'abstract': {
                'concept', 'idea', 'entity', 'thing', 'object',
                'property', 'attribute', 'relation'
            }
        }

        for domain_name, domain_concepts in domains.items():
            # Find pairs within this domain
            domain_pairs = []
            for c1, c2 in all_pairs:
                if c1 in domain_concepts or c2 in domain_concepts:
                    domain_pairs.append((c1, c2))

            if len(domain_pairs) >= 2:  # Need at least one for support, one for query
                task = Task(domain_name, list(domain_concepts), domain_pairs)

                # Split into support (few-shot) and query (evaluation)
                np.random.shuffle(domain_pairs)
                n_support = max(1, len(domain_pairs) // 2)
                task.support_pairs = domain_pairs[:n_support]
                task.query_pairs = domain_pairs[n_support:]

                # Ensure query set is not empty
                if not task.query_pairs:
                    continue

                tasks.append(task)
                print(f"Created task '{domain_name}': {len(task.support_pairs)} support, {len(task.query_pairs)} query pairs")

        return tasks

    def compute_loss_batch(self, theta: np.ndarray, pairs_batch: List[Tuple],
                          data_prep: QManifoldDataPreparation,
                          vectors_scaled: np.ndarray,
                          vectors_pca: np.ndarray,
                          all_concepts: List[str],
                          sampler=None) -> float:
        """
        Compute loss for a batch of pairs using given parameters.
        """
        total_loss = 0.0
        circuits = []

        for c1, c2 in pairs_batch:
            try:
                idx1 = all_concepts.index(c1)
                idx2 = all_concepts.index(c2)
            except ValueError:
                continue

            # Get vectors
            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]
            v1_pca = vectors_pca[idx1]
            v2_pca = vectors_pca[idx2]

            # Compute target (hyperbolic distance)
            target = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
            target_sim = data_prep.hyperbolic_similarity(target)

            # Build circuit with current parameters
            bound_circuit = self.circuit.bind_parameters(v1, v2, theta)
            circuits.append(bound_circuit)

        if not circuits:
            return 0.0

        # Execute circuits (simulator for meta-training)
        if sampler is None:
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
            for i, qc in enumerate(circuits):
                qc_copy = qc.copy()
                transpiled_qc = transpile(qc_copy, backend)
                job = backend.run(transpiled_qc, shots=512)  # Fewer shots for speed
                counts = job.result().get_counts()
                fidelity = counts.get('0' * self.n_qubits, 0) / 512

                # MSE loss
                c1, c2 = pairs_batch[i]
                idx1 = all_concepts.index(c1)
                idx2 = all_concepts.index(c2)
                v1_pca = vectors_pca[idx1]
                v2_pca = vectors_pca[idx2]
                target = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
                target_sim = data_prep.hyperbolic_similarity(target)

                loss = (fidelity - target_sim) ** 2
                total_loss += loss

        return total_loss / len(pairs_batch) if pairs_batch else 0.0

    def compute_gradient(self, theta: np.ndarray, pairs_batch: List[Tuple],
                        data_prep: QManifoldDataPreparation,
                        vectors_scaled: np.ndarray,
                        vectors_pca: np.ndarray,
                        all_concepts: List[str],
                        epsilon: float = 0.01) -> np.ndarray:
        """
        Compute gradient using finite differences (SPSA-style).
        """
        # Generate random perturbation
        delta = np.random.choice([-1, 1], size=len(theta))

        # Compute loss at +/- perturbation
        loss_plus = self.compute_loss_batch(
            theta + epsilon * delta, pairs_batch,
            data_prep, vectors_scaled, vectors_pca, all_concepts
        )
        loss_minus = self.compute_loss_batch(
            theta - epsilon * delta, pairs_batch,
            data_prep, vectors_scaled, vectors_pca, all_concepts
        )

        # Gradient estimate
        gradient = (loss_plus - loss_minus) / (2 * epsilon) * delta

        return gradient

    def inner_loop_adaptation(self, theta_init: np.ndarray, support_pairs: List[Tuple],
                            data_prep: QManifoldDataPreparation,
                            vectors_scaled: np.ndarray,
                            vectors_pca: np.ndarray,
                            all_concepts: List[str]) -> np.ndarray:
        """
        Few-shot adaptation: Take a few gradient steps on support set.
        """
        theta = theta_init.copy()

        for step in range(self.inner_steps):
            # Compute gradient on support set
            grad = self.compute_gradient(
                theta, support_pairs,
                data_prep, vectors_scaled, vectors_pca, all_concepts
            )

            # Update parameters
            theta -= self.inner_lr * grad

            # Compute current loss
            loss = self.compute_loss_batch(
                theta, support_pairs,
                data_prep, vectors_scaled, vectors_pca, all_concepts
            )

            print(f"    Inner step {step+1}: loss={loss:.4f}")

        return theta

    def meta_train(self, tasks: List[Task], data_prep: QManifoldDataPreparation,
                  vectors_scaled: np.ndarray, vectors_pca: np.ndarray,
                  all_concepts: List[str], meta_iterations: int = 10):
        """
        Meta-training: Learn initialization that enables fast adaptation.
        """
        print("\n" + "=" * 50)
        print("META-TRAINING")
        print("=" * 50)

        # Guard against empty task list to avoid division by zero later
        if not tasks:
            print("Warning: No tasks provided for meta-training; skipping meta-training.")
            return []

        meta_losses = []

        for iteration in range(meta_iterations):
            print(f"\nMeta-iteration {iteration+1}/{meta_iterations}")

            # Accumulate gradients across tasks
            meta_gradient = np.zeros_like(self.theta_meta)
            total_query_loss = 0.0

            for task in tasks:
                print(f"\n  Task: {task.name}")

                # More efficient meta-gradient estimation using SPSA
                eps_meta = 0.01
                delta_meta = np.random.choice([-1, 1], size=len(self.theta_meta))

                # Positive perturbation
                theta_meta_plus = self.theta_meta + eps_meta * delta_meta
                theta_adapted_plus = self.inner_loop_adaptation(
                    theta_meta_plus, task.support_pairs,
                    data_prep, vectors_scaled, vectors_pca, all_concepts
                )
                loss_plus = self.compute_loss_batch(
                    theta_adapted_plus, task.query_pairs,
                    data_prep, vectors_scaled, vectors_pca, all_concepts
                )

                # Negative perturbation
                theta_meta_minus = self.theta_meta - eps_meta * delta_meta
                theta_adapted_minus = self.inner_loop_adaptation(
                    theta_meta_minus, task.support_pairs,
                    data_prep, vectors_scaled, vectors_pca, all_concepts
                )
                loss_minus = self.compute_loss_batch(
                    theta_adapted_minus, task.query_pairs,
                    data_prep, vectors_scaled, vectors_pca, all_concepts
                )

                # SPSA meta-gradient for this task
                task_meta_grad = (loss_plus - loss_minus) / (2 * eps_meta) * delta_meta
                meta_gradient += task_meta_grad

                # For logging, compute query loss with unperturbed theta
                theta_adapted_unperturbed = self.inner_loop_adaptation(
                    self.theta_meta, task.support_pairs,
                    data_prep, vectors_scaled, vectors_pca, all_concepts
                )
                query_loss = self.compute_loss_batch(
                    theta_adapted_unperturbed, task.query_pairs,
                    data_prep, vectors_scaled, vectors_pca, all_concepts
                )
                print(f"    Query loss: {query_loss:.4f}")
                total_query_loss += query_loss
                    
            # Update meta-parameters (guard against empty tasks just in case)
            if len(tasks) > 0:
                avg_query_loss = total_query_loss / len(tasks)
                meta_gradient /= len(tasks)
                self.theta_meta -= self.outer_lr * meta_gradient
            else:
                avg_query_loss = 0.0
                print("Warning: No tasks available during meta-update; skipping parameter update.")

            print(f"\n  Average query loss: {avg_query_loss:.4f}")
            meta_losses.append(avg_query_loss)

            # Early stopping if converged
            if len(meta_losses) > 2:
                if abs(meta_losses[-1] - meta_losses[-2]) < 0.001:
                    print("Converged!")
                    break

        return meta_losses

    def few_shot_test(self, test_pairs: List[Tuple],
                     support_size: int,
                     data_prep: QManifoldDataPreparation,
                     vectors_scaled: np.ndarray,
                     vectors_pca: np.ndarray,
                     all_concepts: List[str],
                     use_hardware: bool = False) -> Dict:
        """
        Test few-shot adaptation on new pairs.
        """
        print("\n" + "=" * 50)
        print("FEW-SHOT TEST")
        print("=" * 50)

        # Split test into support and query
        np.random.shuffle(test_pairs)
        support = test_pairs[:support_size]
        query = test_pairs[support_size:]

        print(f"Support set: {support_size} pairs")
        print(f"Query set: {len(query)} pairs")

        # Adapt from meta-initialization
        print("\nAdapting from meta-learned initialization...")
        theta_adapted = self.inner_loop_adaptation(
            self.theta_meta, support,
            data_prep, vectors_scaled, vectors_pca, all_concepts
        )

        # Evaluate on query set
        print("\nEvaluating on query set...")
        predictions = []
        targets = []

        for c1, c2 in query:
            try:
                idx1 = all_concepts.index(c1)
                idx2 = all_concepts.index(c2)
            except ValueError:
                continue

            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]
            v1_pca = vectors_pca[idx1]
            v2_pca = vectors_pca[idx2]

            # Target
            target = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
            target_sim = data_prep.hyperbolic_similarity(target)
            targets.append(target_sim)

            # Prediction (simplified - would use actual circuit)
            bound_circuit = self.circuit.bind_parameters(v1, v2, theta_adapted)

            # Execute
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
            qc = bound_circuit.copy()
            transpiled_qc = transpile(qc, backend)
            job = backend.run(transpiled_qc, shots=1024)
            counts = job.result().get_counts()
            fidelity = counts.get('0' * self.n_qubits, 0) / 1024

            predictions.append(fidelity)
            print(f"  {c1} ↔ {c2}: target={target_sim:.3f}, pred={fidelity:.3f}")

        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)

        if len(predictions) > 0:
            correlation = np.corrcoef(predictions, targets)[0, 1]
            mse = np.mean((predictions - targets) ** 2)
        else:
            correlation = 0.0
            mse = 1.0

        return {
            'correlation': correlation,
            'mse': mse,
            'n_support': support_size,
            'n_query': len(query)
        }


def run_experiment(n_tasks: int = 3, meta_iterations: int = 5):
    """
    Run quantum meta-learning experiment.
    """
    print("=" * 70)
    print("QUANTUM META-LEARNING (MAML) EXPERIMENT")
    print("=" * 70)

    # Initialize data
    data_prep = QManifoldDataPreparation(target_dim=20)
    all_pairs = data_prep.get_default_concept_pairs()

    # Prepare embeddings
    all_concepts = data_prep.generate_all_concepts(all_pairs)
    embeddings = data_prep.embed_concepts(all_concepts)
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # Initialize MAML
    maml = QuantumMAML(n_qubits=20, inner_lr=0.1, outer_lr=0.01, inner_steps=3)

    # Create tasks
    tasks = maml.create_tasks(all_pairs)[:n_tasks]
    print(f"\nUsing {len(tasks)} tasks for meta-training")

    # Meta-train
    meta_losses = maml.meta_train(
        tasks, data_prep, vectors_scaled, vectors_pca, all_concepts,
        meta_iterations=meta_iterations
    )

    # Test few-shot adaptation on held-out pairs
    test_pairs = all_pairs[-20:]  # Last 20 pairs for testing
    test_results = maml.few_shot_test(
        test_pairs=test_pairs,
        support_size=5,
        data_prep=data_prep,
        vectors_scaled=vectors_scaled,
        vectors_pca=vectors_pca,
        all_concepts=all_concepts
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Meta-training losses: {meta_losses}")
    print(f"Few-shot test correlation: {test_results['correlation']:.4f}")
    print(f"Few-shot test MSE: {test_results['mse']:.6f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'quantum_maml',
        'n_tasks': len(tasks),
        'meta_iterations': meta_iterations,
        'inner_steps': maml.inner_steps,
        'inner_lr': maml.inner_lr,
        'outer_lr': maml.outer_lr,
        'meta_losses': meta_losses,
        'test_correlation': test_results['correlation'],
        'test_mse': test_results['mse'],
        'test_support': test_results['n_support'],
        'test_query': test_results['n_query']
    }

    output_file = f"../results/quantum_maml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if test_results['correlation'] > 0.7:
        print("✅ SUCCESS! Meta-learning enables few-shot adaptation!")
        print("   Learned initialization generalizes across semantic domains")
    elif test_results['correlation'] > 0.4:
        print("⚠️ PARTIAL SUCCESS - Meta-learning helps but needs refinement")
    else:
        print("❌ Meta-learning insufficient for this problem")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum Meta-Learning (MAML)')
    parser.add_argument('--n_tasks', type=int, default=3, help='Number of meta-training tasks')
    parser.add_argument('--meta_iterations', type=int, default=5, help='Meta-training iterations')

    args = parser.parse_args()

    print("This experiment uses SIMULATOR ONLY for meta-training")
    print("Hardware would be used for final deployment only")

    results = run_experiment(
        n_tasks=args.n_tasks,
        meta_iterations=args.meta_iterations
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Few-shot correlation: {results['test_correlation']:.4f}")

    if results['test_correlation'] > 0.7:
        print("\n🎉 META-LEARNING SUCCESS!")
        print("Quantum circuits can learn to learn!")
        print("Few-shot adaptation avoids overfitting")