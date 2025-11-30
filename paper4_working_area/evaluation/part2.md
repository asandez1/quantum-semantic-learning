Based on the provided research papers and your constraints, here is the formal definition, roadmap, and experimental design for **Project Q-Manifold**.

### 1\. Formal Research Definition: The "Trilogy" Synthesis

This research defines a new paradigm for Quantum Natural Language Processing (QNLP) that moves beyond the failed "Quantum Compression" narrative (Paper 2) and into **"Quantum Metric Refinement."** It is built upon a trilogy of findings that mathematically justify the Hybrid Quantum-Classical Architecture.

  * **The Discovery (Paper 1 - "Universal Intrinsic Dimensionality"):**
    We established that while LLM embedding spaces are high-dimensional (4096D), the *semantic manifold*—the geometric shape of meaning—has a universal intrinsic dimensionality of approximately **20 dimensions**. This provides the mathematical guarantee that we can compress data by \~178x without destroying semantic topology.

  * **The Limitation (Paper 2 - "Task-Asymmetric Compression"):**
    We proved that **Classical PCA** is the optimal method for reaching this 20D subspace for retrieval tasks, outperforming complex non-linear methods. Crucially, we proved that Quantum Circuits fail at *Generative Compression* (reconstructing high-dimensional statistics) due to the "Perplexity Catastrophe." This dictates that the quantum computer must *not* be used for compression or generation.

  * **The Solution (Paper 3 - "Hyperbolic Contrastive"):**
    We demonstrated that Quantum Hardware (IBM Torino) is a superior optimizer for **geometric alignment**, achieving a 68% lower loss than simulators. By combining these findings, we define **Path B**: A hybrid architecture where Classical PCA handles the bulk compression (finding the 20D manifold) and a Quantum Circuit performs **Hyperbolic Metric Refinement** (bending that 20D manifold to match the non-Euclidean curvature of semantic concepts).

-----

### 2\. The Roadmap: Path B (Quantum Metric Refinement)

This roadmap is designed for **IBM Quantum System Two** architectures (e.g., `ibm_fez`, `ibm_torino`) using the Free Tier constraints.

#### Phase 1: The Classical "Manifold Injector"

**Objective:** Map high-dimensional embeddings to the 20D quantum-ready subspace.

  * **Input:** Mistral-7B embeddings (4096D).
  * **Process:**
    1.  **PCA Reduction:** Project the full dataset $X$ into $X_{20D}$ using the principal components established in Paper 2.
    2.  **Feature Scaling:** Apply Min-Max scaling to map the 20D features to the interval $[0, 2\pi]$. This is critical for **Angle Encoding**, ensuring the features map to valid rotation angles on the Bloch sphere.
  * **Output:** A dataset of 20-dimensional vectors $\vec{v} \in [0, 2\pi]^{20}$.

#### Phase 2: Batch Contrastive Learning (The Quantum Kernel)

**Objective:** Train the quantum circuit to align the 20D vectors in a Hyperbolic Hilbert space.

  * **The Bottleneck Solution:** To solve the "single-pair overfitting" from Paper 3, we implement **Mini-Batch SPSA**. Instead of updating weights for one pair, we calculate the gradient over a batch of $N$ pairs simultaneously.
  * **Hardware Strategy:** We utilize **Qiskit Runtime Primitives V2** in **Batch Mode**. This allows us to submit a single job containing the circuits for an entire mini-batch, minimizing queue latency.

#### Phase 3: The Hybrid Inference Engine

**Objective:** Deployment.

  * **Query Processing:** New queries are PCA-projected to 20D, then passed through the trained Quantum Circuit (fixed parameters).
  * **Retrieval:** The output state fidelities are used as the similarity metric for ranking results, providing a "Quantum-Refined" retrieval ranking that respects hyperbolic semantic relationships better than linear cosine similarity.

-----

### 3\. Experimental Design (Qiskit Code Structure)

This design adheres to your constraints: **20 Qubits**, **Angle Encoding**, and **Batch Execution Mode** without Sessions.

**Core Concept:** We use `SamplerV2` to measure the "Fidelity" (overlap) between two data points. We structure the data as a **Primitive Unified Bloc (PUB)** to broadcast parameters efficiently.

#### A. Circuit Definition (The Ansatz)

We use a **Siamese Network** structure (Compute-Uncompute).

  * **Data:** Two 20D vectors, $\vec{x}$ (Anchor) and $\vec{y}$ (Positive/Negative).
  * **Circuit:** $U^\dagger(\vec{y}) V^\dagger(\theta) V(\theta) U(\vec{x})$.
  * **Success:** If $\vec{x}$ and $\vec{y}$ are semantically similar, we want the probability of measuring the all-zero state $|0\rangle^{\otimes 20}$ to be maximized.

<!-- end list -->

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes

def build_hyperbolic_circuit(n_qubits=20):
    # 1. Define Parameters
    # Data parameters (Angle Encoding for 20 dimensions)
    x_params = ParameterVector('x', n_qubits) # Anchor
    y_params = ParameterVector('y', n_qubits) # Comparison
    
    # 2. Feature Map (Angle Encoding)
    # Simple RY rotations - Hardware efficient, no 'initialize' needed
    feature_map = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        feature_map.ry(x_params[i], i)
        
    # 3. Ansatz (Hyperbolic Refinement Layer)
    # RealAmplitudes is efficient for IBM Heavy-Hex hardware
    ansatz = RealAmplitudes(n_qubits, reps=2, entanglement='linear')
    
    # 4. Composition (Compute - Uncompute)
    # We apply Enc(x) -> Ansatz -> Ansatz_Dag -> Enc_Dag(y)
    # If the ansatz maps x and y to the same point, result is |00..0>
    qc = QuantumCircuit(n_qubits)
    
    # Forward pass (Anchor)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    # Backward pass (Comparison - Inverted)
    qc.compose(ansatz.inverse(), inplace=True)
    
    # Inverse Feature Map (using y parameters)
    inv_feature_map = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        inv_feature_map.ry(y_params[i], i)
    qc.compose(inv_feature_map.inverse(), inplace=True)
    
    qc.measure_all()
    
    return qc, ansatz.parameters, x_params, y_params
```

#### B. Batch Execution Strategy (The SPSA Optimizer)

To solve the "Session" constraint, we implement a **Stateless SPSA** loop. Each step of the optimizer prepares a full batch of circuits and submits them as one job.

```python
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Configuration
BACKEND_NAME = "ibm_fez" # 156-qubit device
BATCH_SIZE = 32          # Number of pairs per update (solves overfitting)
SPSA_ITERATIONS = 50     # Total optimization steps

# 1. Setup Backend
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

# 2. Transpile Circuit Once (ISA Circuit)
qc, theta_params, x_vec, y_vec = build_hyperbolic_circuit(20)
isa_circuit = pm.run(qc)

def run_spsa_step(current_theta, data_batch):
    """
    Submits ONE job containing the entire mini-batch for gradient estimation.
    """
    # SPSA Perturbation (Bernoulli +/- 1)
    delta = 2 * np.random.randint(0, 2, size=len(current_theta)) - 1
    c_k = 0.1 # Perturbation magnitude
    
    theta_plus = current_theta + c_k * delta
    theta_minus = current_theta - c_k * delta
    
    # Prepare Parameter Bindings (Broadcasting)
    # We need to run the circuit for every pair in the batch (BATCH_SIZE)
    # AND for both theta_plus and theta_minus.
    # Total circuits = BATCH_SIZE * 2
    
    # The circuit expects [x_params, theta_params, y_params]
    # We concatenate these into a single list for broadcasting
    
    pubs =
    
    # We pack everything into ONE PUB (Primitive Unified Bloc)
    # Shape: (2 * BATCH_SIZE, Num_Total_Params)
    bindings =
    
    for pair in data_batch:
        x_val, y_val = pair['anchor'], pair['positive'] # Simplified for brevity
        
        # Bind for + perturbation
        bindings.append(np.concatenate([x_val, theta_plus, y_val]))
        
        # Bind for - perturbation
        bindings.append(np.concatenate([x_val, theta_minus, y_val]))
        
    # 3. Execute in Batch Mode
    # Note: We use 'mode=batch' to hint the scheduler, but we don't keep a session open
    # waiting for the result. We submit and wait (blocking call).
    sampler = Sampler(mode=backend)
    job = sampler.run([(isa_circuit, bindings)])
    result = job.result() 
    
    # 4. Calculate Gradient (Classical Post-Processing)
    # Extract counts for the |00..0> state (Fidelity)
    pub_result = result
    counts = pub_result.data.meas.get_counts()
    
    grad_est = np.zeros_like(current_theta)
    
    for i in range(BATCH_SIZE):
        # Get fidelity for + and -
        counts_plus = counts[2*i]
        counts_minus = counts[2*i + 1]
        
        fid_plus = counts_plus.get('0'*20, 0) / 1024 # Assuming 1024 shots
        fid_minus = counts_minus.get('0'*20, 0) / 1024
        
        # Gradient approximation
        gi = (fid_plus - fid_minus) / (2 * c_k)
        grad_est += gi * delta
        
    # Average gradient over batch
    avg_grad = grad_est / BATCH_SIZE
    
    # Update parameters
    new_theta = current_theta - 0.1 * avg_grad # Learning rate 0.1
    return new_theta

# Training Loop
theta = np.random.rand(len(theta_params)) # Initialize weights

print("Starting Batch SPSA Training...")
for i in range(SPSA_ITERATIONS):
    # Get next data batch (x, y pairs from PCA data)
    batch_data = get_next_batch(BATCH_SIZE) 
    
    # Run step (Submits 1 Job)
    theta = run_spsa_step(theta, batch_data)
    print(f"Step {i+1} Complete.")
```

### 4\. Why This Solves the Issues

1.  **Data Loading:** We strictly use **20 dimensions**, which maps 1:1 to the 20 qubits via `AngleEncoding`. This respects the findings of Paper 1 and 2.
2.  **Overfitting:** By averaging the gradient over `BATCH_SIZE=32` distinct pairs in every step, the optimizer cannot memorize a single rotation. It is forced to learn a **generalizable metric** that separates all 32 pairs simultaneously.
3.  **Hardware Constraints:** We use `SamplerV2` with broadcasting. We send **one job per iteration**. While this incurs queue time between steps, it fits the "Batch Mode" requirement and avoids the "Session" timeout/resource lock issues.
4.  **No `initialize`:** We use purely $R_y$ rotations, which are native and efficient on IBM hardware, avoiding the expensive state preparation transpilation.