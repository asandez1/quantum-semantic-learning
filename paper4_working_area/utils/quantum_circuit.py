"""
Quantum Circuit for Metric Refinement (Paper 4)

Implements the Siamese Compute-Uncompute architecture for measuring
semantic similarity via quantum fidelity.

Key Innovation: Uses EstimatorV2 for shot-efficient expectation value
measurement instead of counting rare bitstrings.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, Optional


class QManifoldCircuit:
    """
    Builds the quantum circuit for hyperbolic metric refinement.

    Architecture:
    - Angle Encoding (RY rotations) for 20D input
    - RealAmplitudes ansatz for metric transformation
    - Compute-Uncompute structure for fidelity measurement
    - EstimatorV2-compatible observables
    """

    def __init__(
        self,
        n_qubits: int = 20,
        ansatz_reps: int = 2,
        entanglement: str = 'circular'
    ):
        """
        Args:
            n_qubits: Number of qubits (should match PCA dimension)
            ansatz_reps: Repetitions in RealAmplitudes ansatz
            entanglement: Entanglement pattern ('linear', 'circular', 'full')
        """
        self.n_qubits = n_qubits
        self.ansatz_reps = ansatz_reps
        self.entanglement = entanglement

        # Parameter vectors
        self.x_params = ParameterVector('x', n_qubits)  # Input vector 1
        self.y_params = ParameterVector('y', n_qubits)  # Input vector 2

        # Build ansatz
        self.ansatz = self._build_ansatz()
        self.theta_params = list(self.ansatz.parameters)

        # Build circuits
        self.qc_sampler, self.qc_estimator = self._build_circuits()

        print(f"[Circuit] Built {n_qubits}-qubit circuit")
        print(f"[Circuit] Ansatz: RealAmplitudes(reps={ansatz_reps}, ent={entanglement})")
        print(f"[Circuit] Trainable parameters: {len(self.theta_params)}")

    def _build_ansatz(self) -> QuantumCircuit:
        """
        Build the trainable ansatz (metric transformation layer).

        Returns:
            RealAmplitudes ansatz
        """
        ansatz = RealAmplitudes(
            self.n_qubits,
            reps=self.ansatz_reps,
            entanglement=self.entanglement,
            insert_barriers=False  # Barriers interfere with transpilation
        )
        return ansatz

    def _build_circuits(self) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """
        Build both Sampler and Estimator versions of the circuit.

        Returns:
            (sampler_circuit, estimator_circuit) tuple
        """
        # === Sampler Circuit (with measurement) ===
        qc_sampler = QuantumCircuit(self.n_qubits)

        # Encode X (Angle Encoding)
        for i in range(self.n_qubits):
            qc_sampler.ry(self.x_params[i], i)

        # Apply ansatz V(θ)
        qc_sampler.compose(self.ansatz, inplace=True)

        # Inverse encode Y
        for i in range(self.n_qubits):
            qc_sampler.ry(-self.y_params[i], i)

        # Measure
        qc_sampler.measure_all()

        # === Estimator Circuit (no measurement) ===
        qc_estimator = QuantumCircuit(self.n_qubits)

        # Same structure without measurement
        for i in range(self.n_qubits):
            qc_estimator.ry(self.x_params[i], i)

        qc_estimator.compose(self.ansatz, inplace=True)

        for i in range(self.n_qubits):
            qc_estimator.ry(-self.y_params[i], i)

        return qc_sampler, qc_estimator

    def get_observable_all_zero_projector(self) -> SparsePauliOp:
        """
        Create the observable for |0><0| projector.

        Instead of measuring P(00...0) directly, we measure the expectation
        of (I + Z) / 2 on each qubit and multiply.

        For single qubit: <(I+Z)/2> = P(0)
        For n qubits: We approximate via sum of Z expectations

        Better approach: Use ZZ...Z operator
        <ZZ...Z> is related to parity; high correlation with fidelity

        Returns:
            SparsePauliOp observable
        """
        # Create ZZ...Z operator (all-Z Pauli string)
        pauli_string = 'Z' * self.n_qubits
        observable = SparsePauliOp([pauli_string], coeffs=[1.0])

        return observable

    def get_observable_fidelity_proxy(self) -> SparsePauliOp:
        """
        Create a better fidelity proxy observable.

        Uses sum of single-qubit Z operators: (Z_0 + Z_1 + ... + Z_n) / n
        This gives the average <Z> across qubits, which correlates with
        overlap when both states are close to |00...0>

        Returns:
            SparsePauliOp observable
        """
        # Create identity-padded Z operators for each qubit
        pauli_terms = []
        coeffs = []

        for i in range(self.n_qubits):
            # Create 'III...Z...III' with Z at position i
            pauli_str = 'I' * i + 'Z' + 'I' * (self.n_qubits - i - 1)
            pauli_terms.append(pauli_str)
            coeffs.append(1.0 / self.n_qubits)  # Normalize

        observable = SparsePauliOp(pauli_terms, coeffs=coeffs)

        return observable

    def bind_parameters(
        self,
        x_vector: np.ndarray,
        y_vector: np.ndarray,
        theta_vector: np.ndarray,
        use_estimator: bool = False
    ) -> QuantumCircuit:
        """
        Bind concrete values to circuit parameters.

        Args:
            x_vector: 20D input vector for concept 1
            y_vector: 20D input vector for concept 2
            theta_vector: Trainable weights
            use_estimator: Whether to use estimator circuit (no measurement)

        Returns:
            Bound quantum circuit
        """
        # Select circuit
        qc = self.qc_estimator if use_estimator else self.qc_sampler

        # Create parameter dictionary
        param_dict = {}

        for i, val in enumerate(x_vector):
            param_dict[self.x_params[i]] = val

        for i, val in enumerate(y_vector):
            param_dict[self.y_params[i]] = val

        for i, val in enumerate(theta_vector):
            param_dict[self.theta_params[i]] = val

        # Bind
        bound_circuit = qc.assign_parameters(param_dict)

        return bound_circuit

    def get_parameter_count(self) -> int:
        """Return number of trainable parameters."""
        return len(self.theta_params)

    def get_circuit_depth(self, backend=None) -> int:
        """
        Get circuit depth (before transpilation).

        Args:
            backend: Optional backend for transpiled depth

        Returns:
            Circuit depth
        """
        return self.qc_sampler.depth()

    def __repr__(self) -> str:
        return (f"QManifoldCircuit(n_qubits={self.n_qubits}, "
                f"params={len(self.theta_params)}, "
                f"depth={self.qc_sampler.depth()})")


class FidelityMeasurement:
    """
    Utilities for extracting fidelity from quantum measurements.
    """

    @staticmethod
    def fidelity_from_counts(counts: dict, n_qubits: int) -> float:
        """
        Extract fidelity from SamplerV2 counts.

        Args:
            counts: Dictionary of bitstring counts
            n_qubits: Number of qubits

        Returns:
            Fidelity (probability of all-zero state)
        """
        zero_state = '0' * n_qubits
        total_shots = sum(counts.values())

        if total_shots == 0:
            return 0.0

        fidelity = counts.get(zero_state, 0) / total_shots
        return fidelity

    @staticmethod
    def fidelity_from_expectation(expectation: float, n_qubits: int, method: str = 'avg_z') -> float:
        """
        Convert EstimatorV2 expectation value to fidelity estimate.

        Args:
            expectation: Expectation value from observable
            n_qubits: Number of qubits
            method: 'avg_z' or 'all_z'

        Returns:
            Estimated fidelity
        """
        if method == 'avg_z':
            # <Z_avg> ∈ [-1, 1], map to [0, 1]
            # When all qubits are |0>, <Z> = 1, so <Z_avg> = 1 → fidelity = 1
            fidelity = (expectation + 1.0) / 2.0

        elif method == 'all_z':
            # <ZZ...Z> ∈ [-1, 1]
            # For |00...0>, <ZZ...Z> = 1
            # Heuristic mapping
            fidelity = (expectation + 1.0) / 2.0

        return max(0.0, min(1.0, fidelity))  # Clamp to [0, 1]


if __name__ == "__main__":
    # Test circuit construction
    print("=" * 60)
    print("Testing Q-Manifold Quantum Circuit")
    print("=" * 60)

    circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2, entanglement='circular')

    print(f"\n{circuit}")
    print(f"Circuit depth: {circuit.get_circuit_depth()}")

    # Test parameter binding
    x = np.random.rand(20) * np.pi
    y = np.random.rand(20) * np.pi
    theta = np.random.rand(circuit.get_parameter_count())

    bound_qc = circuit.bind_parameters(x, y, theta, use_estimator=False)
    print(f"\nBound circuit parameters: ✓")
    print(f"Bound circuit has measurements: {bound_qc.num_clbits > 0}")

    # Test observable
    obs = circuit.get_observable_fidelity_proxy()
    print(f"\nObservable: {len(obs.paulis)} Pauli terms")

    print("\n✓ Quantum circuit construction working!")
