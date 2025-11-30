# Paper 4: Q-Manifold - Quantum Metric Refinement for Semantic Embeddings

## Executive Summary

**Status**: Implementation Complete, Ready for Hardware Execution
**Quantum Budget**: 10 minutes (strategic 3-phase probe)
**Innovation**: Hybrid quantum-classical architecture where PCA handles compression (20D) and quantum circuits refine the geometry (Euclidean → Hyperbolic)

### The Research Arc

1. **Paper 1**: Discovered semantic space is intrinsically ~20D
2. **Paper 2**: Proved quantum compression fails, classical PCA is optimal
3. **Paper 3**: Showed quantum hardware excels at geometric optimization (68% better than simulator)
4. **Paper 4** (This Work): Uses quantum circuits for **metric refinement**, not compression

## Key Innovation: The Paradigm Shift

**Old Approach** (Paper 2/3): Quantum circuits compress high-D data → **FAILED**
**New Approach** (Paper 4): Classical PCA compresses → Quantum refines geometry → **TESTABLE**

### Why This Should Work

- **Classical PCA**: Optimal for reaching 20D intrinsic manifold (Paper 1 finding)
- **Quantum Circuit**: Learns to "bend" 20D Euclidean space into hyperbolic geometry
- **Hyperbolic Space**: Natural home for semantic hierarchies (exponential branching)
- **Hardware Advantage**: Paper 3 showed 68% better optimization on real hardware

## Architecture Overview

```
Input: Semantic Concepts
    ↓
[Sentence-BERT Embeddings] (384D)
    ↓
[Classical PCA] → 20D (intrinsic dimensionality)
    ↓
[MinMax Scaling] → [0.1, π-0.1] (for angle encoding)
    ↓
[Quantum Circuit: Angle Encoding + RealAmplitudes Ansatz]
    ↓
[Batch SPSA Optimizer] → Learn hyperbolic metric
    ↓
Output: Quantum-refined similarity scores
```

## Implementation Details

### Quantum Circuit Design

- **Qubits**: 20 (matches intrinsic dimensionality)
- **Encoding**: Angle encoding (RY rotations) - hardware efficient
- **Ansatz**: RealAmplitudes (reps=2, circular entanglement)
- **Structure**: Siamese compute-uncompute for fidelity measurement
- **Observable**: Average Z expectation (shot-efficient via EstimatorV2)

### Optimization Strategy

- **Algorithm**: Mini-Batch SPSA (solves Paper 3 overfitting problem)
- **Batch Size**: 8-12 pairs per gradient update
- **Loss**: MSE between quantum fidelity and hyperbolic similarity
- **Primitives**: EstimatorV2 (more shot-efficient than Sampler)

### Budget Allocation (10 Minutes)

**Phase 0** (Simulator, Free):
- Verify circuit works
- Confirm SPSA converges
- Establish baseline

**Phase 1A** (Hardware, ~2 min):
- Proof-of-concept: 3 iterations, 8 pairs
- Validate hardware advantage
- Goal: Loss improvement vs simulator

**Phase 1B** (Hardware, ~5 min):
- Convergence demonstration: 7 iterations, 12 pairs
- Higher precision (4096 shots)
- Goal: Strong correlation with hyperbolic distances

**Phase 1C** (Reserve, ~2 min):
- Emergency buffer for retries

## Installation & Setup

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- qiskit >= 1.0.0
- qiskit-ibm-runtime >= 0.20.0
- qiskit-aer >= 0.13.0
- sentence-transformers >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

### IBM Quantum Account

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your credentials (one-time setup)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN"
)
```

## Usage

### Phase 0: Simulator Validation (Recommended First)

```bash
cd paper4/experiments
python run_qmanifold_probe.py --phase 0
```

**Expected Output**:
- Loss decreases over 10 iterations
- Final loss < initial loss (convergence)
- Results saved to `results/phase0_simulator_*.json`

### Phase 1A: Hardware Probe

```bash
python run_qmanifold_probe.py --phase 1A --backend ibm_kyiv
```

**Cost**: ~2 minutes of quantum time
**Goal**: Verify hardware advantage exists

### Phase 1B: Full Convergence

```bash
# Only run after successful Phase 1A
python run_qmanifold_probe.py --phase all --backend ibm_kyiv
```

**Cost**: ~7 minutes total
**Goal**: Demonstrate convergence and measure correlation

## Success Criteria

### Phase 0 (Simulator)
- ✓ Loss decreases monotonically
- ✓ Improvement > 0.01
- ✓ No errors in circuit execution

### Phase 1A (Hardware Probe)
- ✓ Job completes successfully
- ✓ Loss improvement vs simulator baseline
- ✓ Confirms Paper 3 hardware advantage

### Phase 1B (Hardware Convergence)
- ✓ Continued loss decrease over 7 iterations
- ✓ Fidelity-distance correlation > 0.70 (target: 0.80)
- ✓ Better than Paper 2 baseline (0.76 correlation)

## Expected Results

### Hypothesis

**If Paper 3's hardware advantage generalizes**, we expect:
1. Hardware loss < simulator loss (at same iteration)
2. Quantum fidelity ≈ hyperbolic similarity (correlation > 0.70)
3. Retrieval performance > classical PCA baseline

### Alternative Outcomes

**If results are negative**:
- Still validates quantum metric learning approach
- Identifies specific failure mode (shot noise, ansatz limitation, etc.)
- Publishable as negative result with diagnostic value

## File Structure

```
paper4/
├── evaluation/
│   ├── newExperiment.md          # Theoretical foundation (Part 1)
│   └── part2.md                   # Implementation details (Part 2)
├── experiments/
│   └── run_qmanifold_probe.py     # Main execution script
├── utils/
│   ├── data_preparation.py        # PCA + scaling pipeline
│   ├── quantum_circuit.py         # Circuit construction
│   └── batch_optimizer.py         # Batch SPSA optimizer
├── results/                       # Experiment results (JSON)
├── data/                          # Cached embeddings/models
├── requirements.txt
└── README.md                      # This file
```

## Troubleshooting

### "No IBM Quantum account found"
```bash
# Re-run account setup
qiskit-ibm-runtime account setup
```

### "Circuit too deep for backend"
- Reduce `ansatz_reps` from 2 to 1
- Or use a backend with higher gate fidelity

### "Job failed on hardware"
- Check backend status: `backend.status()`
- Reduce `shots` from 4096 to 2048
- Try different backend with lower queue time

### "Phase 0 not converging"
- Increase `max_iterations` from 10 to 20
- Adjust `learning_rate` (try 0.05 or 0.15)
- Check data scaling (should be in [0.1, π-0.1])

## Next Steps

### After Successful Hardware Execution

1. **Expand Dataset**: Use full 150 ConceptNet pairs from Paper 2
2. **Benchmark**: Run full semantic preservation metrics
3. **Comparison**: vs Paper 2 (classical), Paper 3 (single-pair quantum)
4. **Retrieval Test**: Actual WordNet hierarchy retrieval task
5. **Publication**: ACL 2026 Workshop on Quantum NLP

### If Hardware Advantage Confirmed

- Investigate scaling to 30-40 qubits (higher-dimensional refinement)
- Test on other semantic datasets (WordNet, FrameNet)
- Explore different ansatz architectures (EfficientSU2, etc.)

## Citation

If you use this work, please cite:

```bibtex
@article{qmanifold2025,
  title={Quantum Metric Refinement for Semantic Embeddings: A Hybrid Classical-Quantum Approach},
  author={[Your Name]},
  journal={ACL 2026 Workshop on Quantum Natural Language Processing},
  year={2025},
  note={Paper 4 in the DiscoveryAI research trilogy}
}
```

## Acknowledgments

- **Paper 1**: Universal intrinsic dimensionality discovery
- **Paper 2**: Quantum compression failure diagnosis
- **Paper 3**: Hardware advantage validation
- **IBM Quantum**: Hardware access (ibm_kyiv, 127+ qubits)
- **Anthropic Claude**: Research assistance and implementation

---

**Last Updated**: November 22, 2025
**Status**: Ready for Phase 0 simulator testing
**Quantum Budget Remaining**: 10 minutes (full allocation)

**"Classical PCA finds the manifold. Quantum circuits refine the geometry."**
