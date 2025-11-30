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

### DOI
```
10.5281/zenodo.17728126
```
**URL**: https://doi.org/10.5281/zenodo.17728126

---

## Description (Abstract)

```
We settle the long-standing open question of whether parameterized quantum circuits can learn high-dimensional semantic
  relationships on NISQ hardware and whether quantum phenomena provide measurable advantages.

  Using IBM's 156-qubit Eagle processor ibm_fez, we show that previous limitations of quantum natural language processing were not 
  fundamental but originated from poor encoding design. All headline results below are obtained on real hardware (no error 
  mitigation, free-tier access, November 2025).

  Key hardware-validated findings:

  • Encoding hierarchy: 132× performance gap between faithful Direct Angle Encoding (r = 0.989) and destructive Difference Encoding
   (r = 0.007)

  • Genuine quantum learning: +1.61 training effect on hardware, transforming actively anti-correlated random unitaries (r = −0.92)
   into semantically meaningful manifolds (r = +0.69) with only 21 parameters

  • Entanglement is required: entangled circuits achieve r = +0.69 while product-state ablation collapses to constant output (r = 
  −0.11), yielding +0.81 entanglement advantage on real hardware

  • Hardware outperforms simulation: +18% correlation improvement versus noiseless simulator

  • First empirical scaling law for quantum semantic learning: increasing training data from 12 → 40 pairs improves generalization 
  4.25× (r = 0.08 → 0.34), projecting parity with classical cosine baseline (r ≈ 0.86) at ~100 training examples

  We establish three architectural prerequisites for quantum semantic learning on NISQ devices: (1) Sparse Encoding, (2) 
  Ancilla-Based Measurement, and (3) Non-Aliased Angle Scaling [0.1, π−0.1].

  This record contains:
  • Complete manuscript (PDF)
  • All source code (Qiskit 2.2.3)
  • Trained parameters (best_theta)
  • Raw IBM Quantum hardware results (JSON)
  • Publication-ready figures
  • 83-pair test set and interactive visualizations

  All experiments are fully reproducible with an IBM Quantum free-tier account.
   Related software: https://doi.org/10.5281/zenodo.17728126
  Zenodo Form Fields


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
Identifier: https://github.com/asandez1/quantum-semantic-learning
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
@software{sandez2025quantum,
  author       = {Sandez, Ariel},
  title        = {{Quantum Semantic Learning on NISQ Hardware:
                   Demonstrated Plasticity, Entanglement Requirement,
                   and Classical-Like Scaling}},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17728126},
  url          = {https://doi.org/10.5281/zenodo.17728126}
}
```

### APA
```
Sandez, A. (2025). Quantum Semantic Learning on NISQ Hardware: Demonstrated Plasticity,
Entanglement Requirement, and Classical-Like Scaling (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.17728126
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
