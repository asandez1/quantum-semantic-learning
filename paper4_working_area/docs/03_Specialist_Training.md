# Step 2: Training a Patch-Specialist Circuit

This phase, executed by scripts like `run_atlas_probe.py`, is where the quantum circuit learns the geometry of a specific, coherent semantic area (a "patch"). The goal is to produce a set of optimized parameters (`theta`) that represent the learned relationships for that patch.

The process is an optimization loop run on a real quantum backend:

1.  **Select Coherent Subset**: A small, semantically related group of concept pairs is selected from the main dataset (e.g., 16 pairs related to animals and living things).
2.  **Prepare Training Data**: The 20D scaled vectors for these pairs (`v1`, `v2`) and their target similarities are bundled for training.
3.  **Define Quantum Circuit**: A `QManifoldCircuit` is defined. It's a variational quantum circuit with 20 qubits (one for each vector dimension) and a certain number of tunable parameters (`theta`), which is 60 in this case. The vectors `v1` and `v2` are also loaded into the circuit's parameters.
4.  **Connect to Quantum Hardware**: The script connects to an IBM Quantum backend (e.g., `ibm_fez`).
5.  **Run Optimizer (SPSA)**: The Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer is used. In each iteration, it:
    a. Prepares a batch of circuits, one for each training pair, with the current `theta`.
    b. Submits them to the quantum hardware.
    c. Gets back the measured outcomes (fidelities).
    d. Calculates the loss (how different the quantum fidelities are from the target similarities).
    e. Updates `theta` to minimize this loss.
6.  **Store Optimized Parameters**: After several iterations, the best-performing `theta` vector is saved. This vector is the "Animal Specialist" knowledge.

```mermaid
graph TD
    A[("📚<br>Coherent Training Pairs<br>(e.g., Animal Hierarchy)")] --> B{Prepare Batch Data};
    B --> C["("✨<br>Scaled Vectors &<br>Target Similarities")"];

    subgraph "SPSA Optimization Loop on Quantum Hardware"
        direction TB
        D[("Theta_initial<br>Random Parameters")];
        D --> E{Iteration};
        E --> F["Prepare Circuits<br>[v1, Theta, v2]"];
        F --> G{{"⚡️<br>Run on<br>Quantum Computer"}};
        G --> H["("📊<br>Measured<br>Fidelities")"];
        H --> I{Calculate Loss};
        I -- "minimize" --> J{Update Theta};
        J --> E;
    end

    C --> I;
    J --> K[("💾<br>Theta_Optimized<br>The 'Animal Specialist'")];

    style K fill:#cde4f9,stroke:#5a91c8,stroke-width:2px
```
