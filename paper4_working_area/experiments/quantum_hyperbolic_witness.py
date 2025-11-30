# quantum_hyperbolic_witness.py
# Tests for GENUINE quantum advantage in hyperbolic metric learning

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np

# Use your BEST animal specialist theta (the one with loss 0.066434)
theta = np.load("../results/animal_specialist_final_20251123.npy")  # you'll save it tomorrow

circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2, entanglement='circular')
qc = circuit.qc_sampler.bind_parameters(np.concatenate([scaled_animal_vector, theta]))

# === TEST 1: Witness via Tree-Depth Entanglement ===
# Pick three concepts with increasing hyperbolic distance (tree depth)
# e.g., "animal" → "mammal" → "dog" → "poodle"  (depth 0 → 1 → 2 → 3)

pairs = [
    ("animal", "mammal"),    # depth 1
    ("mammal", "dog"),       # depth 2  
    ("dog", "poodle")        # depth 3
]

entanglements = []
for c1, c2 in pairs:
    v1 = scale_vector(get_pca(c1))
    v2 = scale_vector(get_pca(c2))
    params = np.concatenate([v1, theta, v2])
    bound = qc.assign_parameters(params)
    
    state = Statevector(bound)
    
    # Trace out all but 4 qubits known to be involved in animal hierarchy (from prior analysis)
    keep_qubits = [0, 5, 10, 15]  # chosen via prior expressivity analysis
    rho = partial_trace(state, [i for i in range(20) if i not in keep_qubits])
    
    S = entropy(rho)  # von Neumann entropy
    entanglements.append(S)

# === EXPECTED RESULT IF QUANTUM ADVANTAGE EXISTS ===
# Entanglement entropy should INCREASE monotonically with hyperbolic tree depth
# Classical simulation (MPS/TTN) predicts entropy plateau or decrease due to area law

print("Hyperbolic Entanglement Witness:")
for (c1,c2), S in zip(pairs, entanglements):
    d_hyp = hyperbolic_distance(get_pca_unscaled(c1), get_pca_unscaled(c2))
    print(f"{c1} → {c2} | d_hyp = {d_hyp:.3f} | S = {S:.3f}")

# === TEST 2: Bell-type violation in hyperbolic space ===
# Construct two pairs that are equidistant classically but different hyperbolically
# Measure violation of classical bound on correlation

# === TEST 3: Magic state witness (optional) ===
# Measure stabilizer Rényi entropy — drops only if non-stabilizer states used