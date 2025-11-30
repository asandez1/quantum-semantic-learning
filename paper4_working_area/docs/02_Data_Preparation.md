# Step 1: Data Preparation and Vectorization

The first crucial step in the pipeline is transforming abstract concepts into a geometric representation. This is handled by the `QManifoldDataPreparation` class found in `utils/data_preparation.py`. The process ensures that semantic relationships are encoded into vectors that the quantum circuit can process.

The flow is as follows:

1.  **Concept Extraction**: The process starts with a list of concept pairs (e.g., "animal-mammal", "car-wheel"). All unique concepts are extracted from these pairs.
2.  **LLM Embedding**: Each unique concept is passed to a classical Large Language Model (LLM) to get a high-dimensional embedding vector (e.g., >1000 dimensions). This vector represents the concept's position in the LLM's semantic space.
3.  **Dimensionality Reduction (PCA)**: The high-dimensional embeddings are projected down to a manageable size (20 dimensions) using Principal Component Analysis (PCA). This is critical because each dimension will correspond to a qubit in the quantum circuit.
4.  **Data Scaling**: The resulting 20D vectors are scaled. This is important for the stability of the optimization process during training. The script maintains both scaled (for quantum input) and unscaled (for classical Poincaré calculations) versions of the vectors.

```mermaid
graph TD
    A[("📝<br>List of<br>Concept Pairs")] --> B{Extract Unique Concepts};
    B --> C{LLM Embedding};
    C --> D["("📚<br>High-Dimensional<br>Embeddings")"];
    D --> E{PCA: Principal Component Analysis};
    E --> F["("🧊<br>20-Dimensional<br>Vectors<br>(Unscaled)")"];
    F --> G{Feature Scaling};
    G --> H["("✨<br>20-Dimensional<br>Vectors<br>(Scaled)")"];

    subgraph "Outputs"
        direction LR
        F --> Z1[("For Classical<br>Poincaré Fallback")];
        H --> Z2[("For Quantum<br>Circuit Input")];
    end

    style D fill:#d1fecb,stroke:#5a91c8
    style F fill:#f9f2cd,stroke:#c8b55a
    style H fill:#cde4f9,stroke:#5a91c8
```
