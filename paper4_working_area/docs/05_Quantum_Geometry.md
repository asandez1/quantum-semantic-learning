# Step 4: The Quantum Geometry Engine

This document explains the theoretical underpinnings of the quantum circuit's operation, addressing how it learns and represents the geometry of meaning.

### Core Idea: Learning a Semantic Metric

The fundamental goal is to make the quantum circuit behave like a "semantic ruler." Given two concepts, it should output a high similarity score if they are close in meaning (like "dog" and "mammal") and a low score if they are far apart (like "dog" and "screwdriver").

The key insight is that the circuit itself can be programmed to define a local "metric" on the semantic manifold. The circuit's tunable parameters, `theta` (θ), define this metric.

### How it Works: From Vectors to Fidelity

The process involves parameterizing a quantum state and measuring its fidelity. Fidelity in this context is a measure of the similarity between two quantum states. Here, it's simplified to be the survival probability of the initial state.

```mermaid
graph TD
    subgraph "Input Preparation"
        A[("Concept 1<br>'dog'")];
        B[("Concept 2<br>'mammal'")];
        A --> C{Data Prep};
        B --> C;
        C --> D[("Vector 1 (v1)<br>[0.1, -0.5, ...]")];
        C --> E[("Vector 2 (v2)<br>[0.2, -0.4, ...]")];
    end

    subgraph "Quantum Circuit Execution"
        F[("θ_animal<br>Learned Specialist<br>Parameters")];
        D --> G{QManifoldCircuit};
        E --> G;
        F --> G;
        G -- "Parameters:<br>[v1, θ, v2]" --> H{Prepare Initial State};
        H --> I{Apply Unitary U(v1, θ, v2)};
        I --> J{Measure System};
    end

    subgraph "Output: Similarity Score"
        J --> K["Count Occurrences<br>of |00...0> state"];
        K --> L["Fidelity =<br>Counts('0...0') / Total Shots"];
        L --> M[("📈<br>Similarity Score<br>e.g., 0.95")];
    end

    style G fill:#cde4f9,stroke:#5a91c8,stroke-width:2px
```

### Answering Your Questions

**1. How is the geometry represented?**
The geometry isn't a fixed background like hyperbolic space. Instead, the circuit *learns* a function that approximates the hyperbolic metric for a specific semantic region. The 60 parameters of `theta` (θ) encode the local curvature and relationships. The vectors `v1` and `v2` act as coordinates that select a point on this learned landscape, and the circuit `U(v1, θ, v2)` calculates the "distance" between them, outputting it as a fidelity.

**2. How is the information (knowledge) stored?**
The knowledge is stored entirely in the classical floating-point vector `theta_optimized`. This ~60-element array *is* the specialist model. It's not stored in the qubits themselves. The qubits are a computational medium, not a persistent storage medium. The `quantum_atlas_lightning_proof.py` script demonstrates this by loading the `theta_animal` parameters from a JSON file.

**3. How are the relations given to the model (training)?**
The relations are taught through supervised learning. During training (`run_atlas_probe.py`), the model is given:
*   A pair of vectors (`v1`, `v2`).
*   A **target similarity score**. This target is derived from a trusted classical model (e.g., the hyperbolic distance in a Poincaré ball model, `exp(-d)`).

The SPSA optimizer's job is to tune `theta` so that the fidelity produced by the quantum circuit for (`v1`, `v2`) gets as close as possible to the target similarity.

**4. How is the response retrieved?**
The "response" is not a word or concept, but a single number: **fidelity**.
1.  The circuit is constructed with the parameters `[v1, theta_optimized, v2]`.
2.  It is executed on the quantum hardware for a fixed number of `shots` (e.g., 4096).
3.  We count how many times the measurement resulted in the all-zeros state (`'00...0'`).
4.  The fidelity is this count divided by the total shots.

This fidelity score **is the response**. It's a quantitative measure of semantic similarity as judged by the trained quantum circuit. The `quantum_atlas_lightning_proof.py` script then correlates these fidelity scores against the known targets to prove the model has learned successfully.
