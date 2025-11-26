# Zenodo Upload Metadata

Use this information when uploading to Zenodo.

---

## Basic Information

### Title
```
Quantum Semantic Learning on NISQ Hardware: Demonstrated Plasticity, Entanglement Requirement, and Classical-Like Scaling
```

### Authors
```
Sandez, Ariel
```

### ORCID
```
0009-0004-7623-6287
```

### Affiliation
```
Independent Researcher, Argentina
```

### Publication Date
```
2025-11-26
```

### DOI (will be assigned by Zenodo)
```
10.5281/zenodo.XXXXXXX
```

---

## Description (Abstract)

```
We address the open question of whether parameterized quantum circuits can natively learn high-dimensional semantic relationships, and whether quantum phenomena provide measurable advantages. Using IBM's 156-qubit quantum processor (ibm_fez), we demonstrate that the perceived limitations of quantum semantic learning are not fundamental, but rather artifacts of input encoding strategy.

Key Results:
• Encoding Hierarchy: 132× performance gap between Direct Angle Encoding (ρ=0.989) and Difference Encoding (ρ=0.007)
• Learning Effect: +1.61 training effect transforming random projections (ρ=-0.92) into semantic manifolds (ρ=+0.69)
• Entanglement Advantage: +0.81 correlation improvement (entangled vs product circuits)
• Hardware Transfer: +18% better performance on hardware vs simulation
• Scaling Law: 4.25× generalization improvement (12→40 training pairs), projecting classical parity at ~100 pairs

We identify three architectural prerequisites for quantum learning: Sparse Encoding, Ancilla-Based Measurement, and Non-Aliased Scaling. All results validated on real 156-qubit NISQ hardware (IBM Quantum ibm_fez, Eagle r3 processor).

This repository contains the full manuscript, experiment code, trained weights, hardware execution results, and publication-ready figures.
```

---

## Keywords/Tags

```
quantum machine learning
quantum natural language processing
NISQ
quantum computing
semantic learning
IBM Quantum
variational quantum circuits
entanglement
quantum advantage
sentence embeddings
Qiskit
```

---

## Resource Type

```
Dataset / Software / Preprint
```

(Select "Dataset" if uploading code+data, or "Preprint" if uploading primarily the paper)

---

## License

```
MIT License
```

---

## Related Identifiers

### GitHub Repository
```
Type: IsSupplementTo
Identifier: https://github.com/YOUR_USERNAME/quantum-semantic-learning
```

### ORCID Profile
```
Type: IsCreatedBy
Identifier: https://orcid.org/0009-0004-7623-6287
```

---

## Subjects (Zenodo Communities)

Consider adding to these communities:
- Quantum Computing
- Machine Learning
- Natural Language Processing
- Open Science

---

## Funding (if applicable)

```
Self-funded independent research
```

---

## Additional Notes

```
All quantum experiments were executed on IBM Quantum ibm_fez (156 qubits, Eagle r3 processor)
using the IBM Quantum Free Tier. Total quantum execution time: <10 minutes.

Software versions:
- Qiskit: 2.2.3
- qiskit-ibm-runtime: 0.43.1
- qiskit-aer: 0.17.2
- sentence-transformers: 5.1.2
- Python: 3.12.3
```

---

## Files to Upload

### Required Files
| File | Description | Size |
|------|-------------|------|
| `manuscript.pdf` | Full paper (generate from HTML) | ~1 MB |
| `manuscript.md` | Source manuscript (Markdown) | 30 KB |
| `README.md` | Repository documentation | 4 KB |

### Code Files
| File | Description |
|------|-------------|
| `experiments/quantum_learning_v3.py` | V3 Sparse Ancilla Architecture |
| `experiments/quantum_learning_v3_hardware.py` | Hardware transfer experiment |
| `experiments/quantum_entanglement_test.py` | Entanglement ablation |
| `experiments/quantum_v3_full_benchmark.py` | Full 83-pair benchmark |
| `experiments/quantum_v3_moredata.py` | Scaling experiment (40 pairs) |
| `generate_final_figures.py` | Figure generation script |

### Results (JSON)
| File | Description |
|------|-------------|
| `experiments/results/v3_best_theta.json` | Trained V3 weights (12 pairs) |
| `experiments/results/v3_moredata_theta.json` | Trained V3 weights (40 pairs) |
| `experiments/results/v3_hardware_ibm_fez_*.json` | Hardware execution results |
| `experiments/results/entanglement_test_hardware_*.json` | Entanglement ablation results |

### Figures (PNG + PDF)
| File | Description |
|------|-------------|
| `figures/fig1_encoding_hierarchy.*` | Encoding strategy comparison |
| `figures/fig2_quantum_advantage.*` | Quantum advantage evidence |
| `figures/fig3_ablation_study.*` | Seven-version ablation |
| `figures/fig4_v3_architecture.*` | V3 circuit architecture |
| `figures/fig5_summary_dashboard_updated.*` | Summary dashboard |
| `figures/fig6_scaling_law.*` | Empirical scaling law |

---

## Recommended Zenodo Upload Options

1. **Upload type**: Dataset (includes code, data, and paper)
2. **Access right**: Open Access
3. **License**: MIT License
4. **Embargo**: None
5. **Reserve DOI**: Yes (get DOI before publishing)

---

## Citation Format (after DOI assigned)

### BibTeX
```bibtex
@dataset{sandez2025quantum,
  author       = {Sandez, Ariel},
  title        = {{Quantum Semantic Learning on NISQ Hardware:
                   Demonstrated Plasticity, Entanglement Requirement,
                   and Classical-Like Scaling}},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

### APA
```
Sandez, A. (2025). Quantum Semantic Learning on NISQ Hardware: Demonstrated Plasticity,
Entanglement Requirement, and Classical-Like Scaling [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## Checklist Before Upload

- [ ] Generate PDF from HTML (`manuscript.pdf`)
- [ ] Verify all figures render correctly
- [ ] Test experiment scripts run without errors
- [ ] Remove any credential files (`save_cred*.py`)
- [ ] Update GitHub URL in related identifiers
- [ ] Review all metadata fields
- [ ] Reserve DOI first, then publish
