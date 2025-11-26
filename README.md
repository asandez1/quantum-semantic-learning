# Quantum Semantic Learning on NISQ Hardware

**Demonstrated Plasticity, Entanglement Requirement, and Classical-Like Scaling**

[![Paper](https://img.shields.io/badge/Paper-Preprint-blue)](manuscript.md)
[![Platform](https://img.shields.io/badge/Platform-IBM%20Quantum-purple)](https://quantum.ibm.com/)
[![Qubits](https://img.shields.io/badge/Qubits-156-green)](https://quantum.ibm.com/)

## Abstract

We demonstrate that parameterized quantum circuits can natively learn high-dimensional semantic relationships on real NISQ hardware. Using IBM's 156-qubit `ibm_fez` processor, we establish:

- **Encoding Hierarchy**: 132x performance gap between encoding strategies
- **Quantum Learning**: +1.61 training effect transforming random projections into semantic manifolds
- **Entanglement Advantage**: +0.81 correlation improvement (entangled vs product circuits)
- **Hardware Transfer**: +18% better performance on hardware vs simulation
- **Scaling Law**: 4.25x generalization improvement (12 to 40 training pairs)

## Key Results

| Phenomenon | Effect Size | Platform | Evidence |
|:---|:---|:---|:---|
| Entanglement | +0.81 | ibm_fez | Entangled (r=0.69) vs Product (r=-0.11) |
| Hardware Transfer | +18% | ibm_fez | Hardware (r=0.608) vs Simulation (r=0.515) |
| Superposition | +1.61 | ibm_fez | Random (r=-0.92) to Trained (r=+0.69) |

## Repository Structure

```
quantum-semantic-learning/
├── manuscript.md                    # Full paper
├── generate_final_figures.py        # Figure generation script
├── figures/
│   ├── fig1_encoding_hierarchy.png  # Encoding strategy comparison
│   ├── fig2_quantum_advantage.png   # Quantum advantage evidence
│   ├── fig3_ablation_study.png      # Seven-version ablation
│   ├── fig4_v3_architecture.png     # V3 circuit architecture
│   ├── fig5_summary_dashboard*.png  # Summary of all results
│   ├── fig6_scaling_law.png         # Empirical scaling law
│   └── *.html                       # Interactive 3D visualizations
├── experiments/
│   ├── quantum_learning_v3.py       # V3 Sparse Ancilla Architecture
│   ├── quantum_learning_v3_hardware.py  # Hardware transfer experiment
│   ├── quantum_entanglement_test.py # Entanglement ablation
│   ├── quantum_v3_full_benchmark.py # Full 83-pair benchmark
│   ├── quantum_v3_moredata.py       # Scaling experiment (40 pairs)
│   └── results/                     # Experiment result JSONs
└── results/                         # Hardware execution results
```

## The V3 Sparse Ancilla Architecture

The winning architecture from our seven-version ablation study:

- **Qubits**: 7 total (3 for v1, 3 for v2, 1 ancilla)
- **Parameters**: 21 trainable (7 qubits x 3 layers)
- **Encoding**: RY gates scaled to [0.1, π-0.1]
- **Entanglement**: CX gates connecting data qubits to ancilla
- **Measurement**: Ancilla only — P(|1⟩) indicates dissimilarity

### Why V3 Works

1. **Sparse Encoding**: Leaves Hilbert space headroom for learning
2. **Ancilla Measurement**: Clean gradient signal (vs brittle global parity)
3. **Non-Aliased Scaling**: Prevents quantum state aliasing at boundaries

## Quick Start

### Requirements

```bash
pip install qiskit==2.2.3 qiskit-ibm-runtime==0.43.1 qiskit-aer==0.17.2
pip install sentence-transformers==5.1.2 scikit-learn numpy scipy matplotlib
```

### Run Simulation

```python
# Run V3 architecture on simulator
python experiments/quantum_learning_v3.py
```

### Run on IBM Quantum Hardware

```python
# Requires IBM Quantum account
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

# Run hardware transfer experiment
python experiments/quantum_learning_v3_hardware.py
```

## Figures

### Figure 1: Encoding Hierarchy
![Encoding Hierarchy](figures/fig1_encoding_hierarchy.png)

### Figure 4: Quantum Advantage Evidence
![Quantum Advantage](figures/fig4_quantum_advantage_updated.png)

### Figure 6: Empirical Scaling Law
![Scaling Law](figures/fig6_scaling_law.png)

## Citation

```bibtex
@article{sandez2025quantum,
  title={Quantum Semantic Learning on NISQ Hardware: Demonstrated Plasticity,
         Entanglement Requirement, and Classical-Like Scaling},
  author={Sandez, Ariel},
  journal={Preprint},
  year={2025},
  note={IBM Quantum ibm\_fez (156 qubits)}
}
```

## Author

**Ariel Sandez**
AI/ML Independent Researcher, Argentina
ORCID: [0009-0004-7623-6287](https://orcid.org/0009-0004-7623-6287)
Email: ariel.sandez@fortegrp.com
LinkedIn: [linkedin.com/in/sandez](https://www.linkedin.com/in/sandez/)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- IBM Quantum for hardware access on ibm_fez (156 qubits, Eagle r3 processor)
- Qiskit development team for the quantum computing framework
