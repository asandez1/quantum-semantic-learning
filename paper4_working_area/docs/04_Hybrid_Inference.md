# Step 3: Hybrid Inference with the Quantum Atlas

This is the core of the "Quantum Atlas" architecture, demonstrated in `quantum_atlas_lightning_proof.py`. It uses the pre-trained specialist from the previous step to make predictions on new, unseen data. Its key innovation is the **classical router**, which decides whether to use the resource-intensive quantum circuit or a fast classical fallback.

The inference flow for each validation pair is:

1.  **Receive Input Pair**: The system takes a new pair of concepts (c1, c2) from the validation set.
2.  **Classical Router**: It checks if both concepts belong to the specialist's domain (e.g., are they both in the `ANIMAL_CONCEPTS` set?).
3.  **Quantum Path**: If both are in the domain:
    a. The 20D **scaled** vectors (`v1`, `v2`) are retrieved.
    b. The full parameter set is constructed by concatenating the input vectors with the specialist's optimized parameters: `[v1, theta_animal, v2]`.
    c. This single circuit is run on the quantum hardware.
    d. The similarity is the measured fidelity (the probability of measuring the all-zeros state `|00...0>`).
4.  **Classical Path (Fallback)**: If the pair is outside the specialist's domain:
    a. The 20D **unscaled** vectors are retrieved.
    b. The classical Poincaré hyperbolic distance `d` is computed between them.
    c. The similarity is calculated as `exp(-d)`.
5.  **Collect and Validate**: The similarity score, whether from the quantum or classical path, is recorded. After processing all pairs, the list of predicted similarities is correlated with the ground-truth target similarities to produce the final score (0.779).

```mermaid
graph TD
    A[("INPUT<br>Validation Pair<br>(c1, c2)")] --> B{Classical Router};
    B -- "c1 & c2 in ANIMAL_CONCEPTS?" --> C{Yes};
    B -- "else" --> D{No};

    subgraph "Quantum Path"
        direction LR
        C --> C1{"Get Scaled<br>Vectors v1, v2"};
        C1 --> C2{Construct Params<br>[v1, θ_animal, v2]};
        C2 --> C3{{"⚡️<br>Run on<br>Quantum Computer"}};
        C3 --> C4["("📊<br>Fidelity")"];
    end

    subgraph "Classical Fallback Path"
        direction LR
        D --> D1{"Get Unscaled<br>Vectors v1, v2"};
        D1 --> D2{Compute<br>Poincaré Distance};
        D2 --> D3["("📐<br>exp(-distance)")"];
    end

    C4 --> E[("Output<br>Similarity Score")];
    D3 --> E;

    subgraph "Final Validation"
      E --> F{Collect All Scores};
      F --> G["Correlate vs. Targets"];
      G --> H[("✅<br>Final Correlation<br>0.779")];
    end

    style B fill:#f9f2cd,stroke:#c8b55a,stroke-width:2px
    style C3 fill:#cde4f9,stroke:#5a91c8
    style D2 fill:#d1fecb,stroke:#5aa882
```
