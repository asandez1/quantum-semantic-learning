# Q-Manifold Setup Guide

## Quick Start

### 1. Install Required Packages

```bash
cd /home/qstar/AI/DiscoveryAI/paper4

# Install all requirements
pip3 install --user -r requirements.txt

# Or install individually if needed:
pip3 install --user qiskit qiskit-ibm-runtime qiskit-aer
pip3 install --user sentence-transformers scikit-learn scipy
pip3 install --user matplotlib pandas tqdm
```

### 2. Configure IBM Quantum Account

```bash
# Save your IBM Quantum API token (one-time setup)
python3 -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum',
    token='YOUR_IBM_QUANTUM_TOKEN',
    overwrite=True
)
"
```

Get your token from: https://quantum.ibm.com/account

### 3. Verify Installation

```bash
cd experiments
python3 -c "
import qiskit
import qiskit_ibm_runtime
from sentence_transformers import SentenceTransformer
print('✓ All packages installed successfully!')
print(f'Qiskit version: {qiskit.__version__}')
"
```

### 4. Run Simulator Test (Phase 0)

```bash
cd experiments
python3 run_qmanifold_probe.py --phase 0
```

This will:
- Generate 50 concept pairs from semantic hierarchies
- Apply PCA to reduce to 20D
- Build quantum circuit (20 qubits, RealAmplitudes ansatz)
- Run 10 iterations of SPSA optimization
- Save results to `results/phase0_simulator_*.json`

**Expected time**: 2-3 minutes
**Expected result**: Loss should decrease from ~0.2 to ~0.05

### 5. Run Hardware Probe (Phase 1A)

**⚠ WARNING**: This uses ~2 minutes of your 10-minute quantum budget!

```bash
python3 run_qmanifold_probe.py --phase 1A --backend ibm_kyiv
```

Check available backends:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backends = service.backends(simulator=False, operational=True)
for b in backends:
    print(f"{b.name}: {b.num_qubits} qubits, queue: {b.status().pending_jobs}")
```

Recommended backends for 20-qubit circuits:
- `ibm_kyiv` (127 qubits)
- `ibm_sherbrooke` (127 qubits)
- `ibm_brisbane` (127 qubits)

### 6. Run Full Hardware Test (Phase 1B)

**⚠ WARNING**: This uses ~5 more minutes (7 minutes total)!

```bash
python3 run_qmanifold_probe.py --phase all --backend ibm_kyiv
```

This runs both Phase 1A and 1B sequentially.

## Troubleshooting

### "No module named 'sklearn'"
```bash
pip3 install --user scikit-learn
```

### "No module named 'sentence_transformers'"
```bash
pip3 install --user sentence-transformers
```

### "No IBM Quantum account found"
Re-run the account setup in step 2 above.

### "Backend too busy" or "Queue time > 10 minutes"
Try a different backend with lower queue time.

### "Circuit transpilation failed"
Your backend might not support the circuit topology. Try:
- Reducing `ansatz_reps` from 2 to 1 in the code
- Using a backend with better connectivity

## Understanding the Results

### Phase 0 (Simulator)

Success indicators:
- `final_loss < initial_loss` ✓
- `improvement > 0.01` ✓
- Loss curve shows steady decrease ✓

The JSON file contains:
- `history`: Loss per iteration
- `theta_optimized`: Trained circuit parameters
- `improvement`: Total loss reduction

### Phase 1A (Hardware Probe)

Success indicators:
- Job completes without errors ✓
- `final_loss < simulator_final_loss` (hardware advantage) ✓
- Time used < 3 minutes ✓

### Phase 1B (Hardware Convergence)

Success indicators:
- Continued loss decrease over 7 iterations ✓
- Final loss < 0.05 (strong correlation) ✓
- Time used < 6 minutes ✓

## Next Steps After Successful Hardware Run

1. **Analyze Correlation**: Compute Pearson correlation between quantum fidelities and hyperbolic distances
2. **Expand Dataset**: Run on full 150 pairs from Paper 2
3. **Benchmark**: Compare to classical PCA baseline and Paper 2/3 results
4. **Publish**: Write up results for ACL 2026 Workshop

## Cost Estimation

| Phase | Quantum Time | Purpose | Required? |
|-------|--------------|---------|-----------|
| 0 | Free (simulator) | Validation | ✓ Yes |
| 1A | ~2 minutes | Proof-of-concept | ✓ Yes |
| 1B | ~5 minutes | Convergence | Optional |
| 1C | ~2 minutes | Reserve/retry | Buffer |

**Total Budget**: 10 minutes
**Recommended Strategy**: Run Phase 0 first, then Phase 1A. Only run 1B if 1A shows hardware advantage.

## Emergency Debugging

If Phase 0 fails, check:
```bash
# Test data preparation
cd utils
python3 -c "
from data_preparation import QManifoldDataPreparation
prep = QManifoldDataPreparation(target_dim=20)
pairs = prep.get_default_concept_pairs()
print(f'Generated {len(pairs)} concept pairs')
"

# Test circuit construction
python3 -c "
from quantum_circuit import QManifoldCircuit
circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2)
print(f'Circuit depth: {circuit.get_circuit_depth()}')
print(f'Parameters: {circuit.get_parameter_count()}')
"
```

## Getting Help

If you encounter issues:
1. Check this guide first
2. Review the error message carefully
3. Check IBM Quantum system status: https://quantum.ibm.com/
4. Consult the main README.md for architecture details
5. Review the theoretical foundation in evaluation/newExperiment.md

---

**Good luck with your quantum experiment!** 🚀
