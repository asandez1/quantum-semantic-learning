# Paper 4: Q-Manifold Continuation Prompt

## Context for New Chat Session

I'm continuing work on **Paper 4: Q-Manifold** - a quantum-classical hybrid system for hyperbolic metric refinement of semantic embeddings. This is part of a three-paper research arc documented in `/home/qstar/AI/DiscoveryAI/CLAUDE.md`.

## Project Overview

**Q-Manifold** uses:
- **Classical PCA**: Compress semantic embeddings from 384D → 20D (intrinsic dimensionality from Paper 1)
- **Quantum Circuits**: Refine the 20D geometry using quantum metric learning on IBM Quantum hardware
- **Goal**: Achieve >0.90 semantic preservation (measured via fidelity-distance correlation)

**Novel Paradigm**: Quantum circuits don't compress - they refine geometry. Classical methods handle dimensionality reduction.

## Current Status: READY FOR HARDWARE EXECUTION ✅

### What's Been Accomplished

#### 1. Full Implementation Complete ✅
Located in `/home/qstar/AI/DiscoveryAI/paper4/`:

**Core Modules:**
- `utils/data_preparation.py` - PCA compression, hyperbolic distance computation
- `utils/quantum_circuit.py` - 20-qubit angle encoding + RealAmplitudes ansatz
- `utils/batch_optimizer.py` - Batch SPSA with shot-efficient sampling
- `experiments/run_qmanifold_probe.py` - Main execution script (3-phase strategy)

**Documentation:**
- `README.md` - Architecture overview
- `SETUP_GUIDE.md` - Installation instructions
- `IMPLEMENTATION_COMPLETE.md` - Status summary
- `manuscript.md` - Comprehensive research paper draft

#### 2. Bug Fixes Applied ✅

**Bug #1: Parameter Values Array Format**
- **Issue**: `parameter_values` was list of arrays, not 2D numpy array
- **Fix**: Added `parameter_values = np.array(parameter_values)` in batch_optimizer.py:180
- **Status**: FIXED ✅

**Bug #2: SamplerV2 get_counts() Indexing**
- **Issue**: `counts_list[i]` failed with KeyError - get_counts() needs index parameter
- **Fix**: Changed to `pub_result.data.meas.get_counts(i)` in batch_optimizer.py:202-203
- **Status**: FIXED ✅

#### 3. Phase 0 Simulator Validation ✅

**Results** (from `results/phase0_simulator_20251122_132313.json`):
```
Initial Loss: 0.333
Final Loss:   0.257
Reduction:    23% improvement
Convergence:  Excellent ✅
Status:       GREEN LIGHT for hardware
```

#### 4. Manuscript Written ✅
- Complete research paper in `paper4/manuscript.md`
- Sections: Introduction, Architecture, Experimental Design, Results (Phase 0), Discussion
- Ready for Phase 1A/1B results to be added

### What's Blocking: Quantum Budget Exhausted ❌

**IBM Quantum Usage:**
- Used: 10m 5s / 10m (100% of cycle quota)
- Remaining: 0s
- Current Cycle: Oct 25, 2025 - Nov 22, 2025
- **Next Cycle Starts: Nov 23, 2025** ← Fresh 10 minute budget

**Available Backends:**
- `ibm_marrakesh` (156 qubits) ← Use this one
- `ibm_fez` (156 qubits) - may have usage limits
- `ibm_torino` (133 qubits) - may have usage limits

## Next Steps: Execute Phase 1A on Hardware 🚀

### Immediate Actions (Once Quantum Budget Resets)

**Step 1: Activate Environment**
```bash
cd /home/qstar/AI/DiscoveryAI/paper4/experiments
source /home/qstar/AI/DiscoveryAI/venv/bin/activate
```

**Step 2: Run Phase 1A (Hardware Probe - ~2 minutes)**
```bash
python -u run_qmanifold_probe.py --phase 1A --backend ibm_marrakesh --yes 2>&1 | tee /tmp/phase1a_hardware.log
```

**Expected Output:**
```
[Phase 1A] Starting hardware optimization...
[Iter 1/3] Submitting 16 circuits...
[Iter 1/3] Loss: X.XXX
[Iter 2/3] Submitting 16 circuits...
[Iter 2/3] Loss: X.XXX
[Iter 3/3] Submitting 16 circuits...
[Iter 3/3] Loss: X.XXX
[Phase 1A] Final loss: X.XXX
```

**Step 3: Evaluate Go/No-Go Criteria**

Based on Phase 1A final loss:
- **Loss < 0.25**: 🟢 GREEN - Excellent! Proceed to Phase 1B immediately
- **Loss 0.25-0.35**: 🟡 YELLOW - Acceptable, proceed with caution to Phase 1B
- **Loss > 0.35**: 🔴 RED - Stop. Worse than simulator (0.257). Redesign needed.

**Step 4: If GREEN/YELLOW, Run Phase 1B (~5 minutes)**
```bash
python -u run_qmanifold_probe.py --phase 1B --backend ibm_marrakesh --yes 2>&1 | tee /tmp/phase1b_hardware.log
```

**Step 5: Analyze Results**
```bash
# Check results
ls -lrt results/*.json | tail -5

# Phase 1A results
cat results/phase1A_ibm_marrakesh_*.json | python -m json.tool

# Phase 1B results (if ran)
cat results/phase1B_ibm_marrakesh_*.json | python -m json.tool
```

**Step 6: Update Manuscript**
- Add Phase 1A results to Section 5.2
- Add Phase 1B results to Section 5.3 (if applicable)
- Update abstract with hardware loss values
- Add interpretation to Discussion section

## Key Technical Details

### Circuit Architecture
- **Qubits**: 20 (matches PCA-compressed dimensionality)
- **Encoding**: Angle encoding via RY gates (O(1) depth)
- **Ansatz**: RealAmplitudes with circular entanglement (2 reps, 60 parameters)
- **Architecture**: Siamese compute-uncompute measuring fidelity via |0⟩⊗20 probability
- **Transpiled Depth**: ~680 gates on 156-qubit Heavy-Hex lattice

### Optimizer Configuration
- **Algorithm**: Batch-mode SPSA (Simultaneous Perturbation Stochastic Approximation)
- **Batch Size**: 8 concept pairs per iteration
- **Learning Rate**: α=0.1 with decay (α=0.602 schedule)
- **Perturbation**: c=0.1 with decay (γ=0.101 schedule)
- **Shots**: 4096 per circuit
- **Primitive**: SamplerV2 (no observable mismatch issues)

### Data Configuration
- **Source**: 150 ConceptNet concept pairs from Paper 2
- **Phase 0/1A**: 16 pairs (2 batches) for quick validation
- **Phase 1B**: 16 pairs with more iterations for convergence
- **Full Benchmark**: 150 pairs (not yet run - Phase 2 task)

### Success Metrics
- **Primary**: Final loss < 0.25 (better than simulator)
- **Secondary**: Fidelity-Distance correlation > 0.90 (full benchmark)
- **Comparison**: Beat Paper 2 best (0.76) and approach Paper 1 classical baseline (0.927)

## File Structure

```
paper4/
├── experiments/
│   └── run_qmanifold_probe.py       # Main execution script ⭐
├── utils/
│   ├── data_preparation.py          # PCA + hyperbolic distances
│   ├── quantum_circuit.py           # Circuit construction
│   ├── batch_optimizer.py           # SPSA optimizer (BUGS FIXED ✅)
│   └── ibm_quantum_helpers.py       # Error mitigation utilities
├── results/
│   └── phase0_simulator_*.json      # Phase 0 validation results
├── data/
│   └── embeddings_cache.pkl         # Cached sentence embeddings
├── manuscript.md                     # Research paper (needs Phase 1A/1B results)
├── README.md                         # Quick start guide
├── SETUP_GUIDE.md                   # Installation instructions
├── IMPLEMENTATION_COMPLETE.md       # Status summary
└── requirements.txt                  # Python dependencies

Key files at repo root:
├── CLAUDE.md                        # Three-paper research arc overview
└── venv/                            # Python virtual environment
```

## Critical Bugs That Were Fixed

### Bug #1: Parameter Binding Format (FIXED ✅)
**Location**: `paper4/utils/batch_optimizer.py` line 180

**Original Code** (BROKEN):
```python
parameter_values = []
for (idx1, idx2), target_sim in zip(pair_indices, target_sims):
    params_plus = np.concatenate([x_vec, theta_plus, y_vec])
    parameter_values.append(params_plus)
```

**Fixed Code**:
```python
parameter_values = []
for (idx1, idx2), target_sim in zip(pair_indices, target_sims):
    params_plus = np.concatenate([x_vec, theta_plus, y_vec])
    parameter_values.append(params_plus)

# CRITICAL FIX: Convert to 2D NumPy array
parameter_values = np.array(parameter_values)  # Shape: (batch_size*2, num_params)
```

### Bug #2: SamplerV2 Result Access (FIXED ✅)
**Location**: `paper4/utils/batch_optimizer.py` lines 202-203

**Original Code** (BROKEN):
```python
pub_result = result[0]
counts_list = pub_result.data.meas.get_counts()  # Returns dict, not list!

for i in range(len(pair_indices)):
    counts_plus = counts_list[2 * i]  # KeyError: 0
```

**Fixed Code**:
```python
pub_result = result[0]

for i in range(len(pair_indices)):
    # Get counts for each parameter set in the batch
    counts_plus = pub_result.data.meas.get_counts(2 * i)     # Correct API usage
    counts_minus = pub_result.data.meas.get_counts(2 * i + 1)
```

## Quantum Budget Management

**3-Phase Strategic Allocation:**
1. **Phase 0**: Simulator validation (FREE) ✅ Complete
2. **Phase 1A**: Hardware probe - 3 iterations (~2 min) ⏳ Ready to run
3. **Phase 1B**: Convergence test - 7 iterations (~5 min) ⏳ Conditional on 1A
4. **Reserve**: 2-3 minutes for re-runs or extended benchmarks

**Total Available**: 10 minutes per cycle
**Used This Cycle**: 10m 5s (100%)
**Next Cycle Start**: Nov 23, 2025

## Expected Timeline (Once Budget Resets)

```
T+0:00  │ Start Phase 1A execution
T+0:15  │ Circuit transpilation complete
T+0:30  │ First quantum job submitted
T+1:30  │ Phase 1A complete (3 iterations)
        │ ↓ Evaluate loss
T+1:35  │ Decision: GREEN/YELLOW/RED
        │ ↓ If GREEN or YELLOW
T+1:40  │ Start Phase 1B execution
T+6:40  │ Phase 1B complete (7 iterations)
        │ ↓ Analyze results
T+7:00  │ Update manuscript with results
T+8:00  │ Paper 4 complete! 🎉
```

## Questions to Answer with Hardware Results

1. **Does quantum hardware improve over simulator?**
   - Compare Phase 1A loss vs Phase 0 loss (0.257)

2. **Does the approach converge?**
   - Track loss trajectory across Phase 1B iterations

3. **How does it compare to baselines?**
   - vs Paper 1 classical (0.927)
   - vs Paper 2 best attempt (0.76)
   - vs Paper 3 hardware failure (0.165)

4. **Is the paradigm shift valid?**
   - Does "quantum metric refinement" work better than "quantum compression"?

## Instructions for Continuation

**Tell Claude:**

```
Continue work on Paper 4: Q-Manifold quantum-classical hybrid system.

Current status: All code implemented and debugged. Phase 0 simulator validation
succeeded (loss 0.333 → 0.257). Ready to run Phase 1A on IBM Quantum hardware.

Previous session exhausted quantum budget (10m/10m used). New cycle should have
started, giving us fresh 10 minutes.

Tasks:
1. Check IBM Quantum budget availability
2. Execute Phase 1A on ibm_marrakesh (156 qubits, ~2 min)
3. Evaluate results using Go/No-Go criteria
4. If GREEN/YELLOW: Execute Phase 1B (~5 min)
5. Update manuscript.md with hardware results
6. Analyze final semantic preservation metrics

Reference files:
- /home/qstar/AI/DiscoveryAI/paper4/experiments/run_qmanifold_probe.py
- /home/qstar/AI/DiscoveryAI/paper4/manuscript.md
- /home/qstar/AI/DiscoveryAI/CLAUDE.md (project context)

All bugs have been fixed (parameter array formatting + SamplerV2 indexing).
Code is production-ready.
```

## Additional Context

### Why This Matters
- Paper 1: Classical baseline (0.927 correlation) ✅
- Paper 2: Diagnosed why quantum NLP fails (geometry destruction) ✅
- Paper 3: Failed attempt on hardware (0.165 preservation) ❌
- **Paper 4**: First attempt at quantum-NATIVE solution (not classical adaptation)

### Key Innovation
**Paradigm Shift**: Use quantum circuits for geometric transformation, not compression.
- Classical PCA: Handles 384D → 20D compression (97.8% variance retained)
- Quantum Circuits: Refine 20D geometry using Hilbert space operators
- Hyperbolic Target: Poincaré disk distances guide quantum metric learning

### Success Criteria
- **Minimum Viable**: Loss < 0.35 (better than Paper 3)
- **Strong Success**: Fidelity-Distance correlation > 0.90
- **Breakthrough**: Approach or exceed Paper 1 classical baseline (0.927)

---

**Status**: Ready for hardware execution. Waiting for quantum budget reset.

**Next Session Goal**: Complete Phase 1A/1B hardware runs and update manuscript.

**Estimated Time to Completion**: ~2 hours (including analysis and writing)
