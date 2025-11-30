# Quantum Atlas: System Overview

This diagram provides a high-level overview of the complete Quantum Atlas architecture, from initial data processing to the final hybrid validation. It shows the two main phases: **Specialist Training** (offline) and **Hybrid Inference** (online).

```mermaid
graph TD
    subgraph "Phase 1: Specialist Training (Offline, e.g., run_atlas_probe.py)"
        A1[("📚<br>Semantic Pairs<br>(e.g., 'animal-mammal')")] --> A2{Data Preparation};
        A2 --> A3["PCA(20D) + Scaling"];
        A3 --> A4[("🏋️<br>Train Specialist Circuit<br>on Quantum Hardware")];
        A4 --> A5[("💾<br>Store Optimized<br>Parameters θ_animal")];
    end

    subgraph "Phase 2: Hybrid Inference (Online, e.g., quantum_atlas_lightning_proof.py)"
        B1[("INPUT<br>New Concept Pair<br>(c1, c2)")] --> B2{Classical Router};
        B2 -- "Is it an 'Animal' pair?" --> B3{Yes};
        B2 -- "Otherwise" --> B4{No};

        B3 --> B5["Quantum Path"];
        B5 --> B6[("⚡️<br>Run Specialist Circuit<br>with θ_animal, v1, v2")];
        B6 --> B7[("Quantum Similarity<br>(Fidelity)")];

        B4 --> B8["Classical Path"];
        B8 --> B9[("📐<br>Calculate Poincaré<br>Hyperbolic Distance")];
        B9 --> B10[("Classical Similarity<br>(exp(-d))")];

        B7 --> B11{Output Similarity};
        B10 --> B11;
    end

    subgraph "Validation"
       C1[("🎯<br>Target Similarities")]
       B11 --> C2{Compare & Correlate}
       C1 --> C2
       C2 --> C3[("✅<br>Final Correlation<br>e.g., 0.779")]
    end

    style A5 fill:#cde4f9,stroke:#5a91c8,stroke-width:2px
    style B1 fill:#f9f2cd,stroke:#c8b55a,stroke-width:2px
```
