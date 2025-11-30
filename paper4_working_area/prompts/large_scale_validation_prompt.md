# Large-Scale Validation of Quantum Semantic Encoding

## Context
I've been developing a quantum semantic encoding circuit that achieved **0.73 correlation** with cosine similarity on 12 test pairs. I need to validate this result with a larger sample (50-100 pairs) to confirm it's not an artifact of small sample size.

## Current Status

**Working Circuit:**
- **Platform:** IBM Quantum ibm_fez (156 qubits)
- **Architecture:** 12-qubit attention circuit
- **Encoding:** Angle encoding (RY gates)
- **Entanglement:** Local pairs + cross-group
- **Similarity Metric:** Normalized Hamming weight
- **Proven Correlation:** 0.73 (n=12 pairs)

**Key Discovery:** Must use **cosine similarity** from raw embeddings as targets, NOT hyperbolic distance (which is broken).

## Files Available

**Working Code:**
- `paper4/experiments/cosine_diagnostic.py` - Validated diagnostic (0.73 correlation)
- `paper4/experiments/large_scale_validation.py` - Last experiment for large scale
- `paper4/utils/data_preparation.py` - Embedding preparation
- `paper4/utils/quantum_circuit.py` - Circuit building utilities

## Task

Detail of the large-scale validation experiment:
paper4/experiments/large_scale_validation.py

1. **Test 50-100 diverse concept pairs** spanning:
   - HIGH similarity (cosine > 0.7): synonyms, closely related
   - MEDIUM similarity (0.4-0.7): related concepts
   - LOW similarity (< 0.4): unrelated concepts

2. **Measure:**
   - Overall correlation with cosine similarity
   - Correlation by category (HIGH/MED/LOW)
   - Error distribution
   - Systematic biases (compression, overestimation)

3. **Validate hypotheses:**
   - Correlation remains ~0.73 with larger sample
   - HIGH similarity is compressed by ~20%
   - MEDIUM similarity is most accurate
   - Predictions follow correct ordering

## Output

```
python large_scale_validation.py
======================================================================
LARGE-SCALE VALIDATION: 75 CONCEPT PAIRS
======================================================================
Testing correlation between quantum circuit and cosine similarity
Target: Correlation >= 0.70 with p < 0.01
======================================================================

Total pairs: 75
  HIGH similarity:   25 pairs
  MEDIUM similarity: 25 pairs
  LOW similarity:    25 pairs
[Data Prep] Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
[Data Prep] Embedding dim: 384 → Target: 12D
qiskit_runtime_service._discover_account:WARNING:2025-11-24 22:50:30,485: Loading account with the given token. A saved account will not be used.

Backend: ibm_fez (156 qubits)
Unique concepts: 139

Generating embeddings...
[Data Prep] Loaded 139 embeddings from cache

======================================================================
BATCH 1: Pairs 1-25
======================================================================
Transpiling 25 circuits...
Submitted job: d4igmriv0j9c73e2ardg
Waiting for job to complete...
Job finished!
  dog ↔ puppy: target=0.804, pred=0.300
  car ↔ automobile: target=0.865, pred=0.383
  happy ↔ joyful: target=0.684, pred=0.785
  big ↔ large: target=0.807, pred=0.573
  small ↔ tiny: target=0.912, pred=0.385
  fast ↔ quick: target=0.652, pred=0.662
  smart ↔ intelligent: target=0.749, pred=0.663
  beautiful ↔ gorgeous: target=0.834, pred=0.587
  angry ↔ furious: target=0.453, pred=0.612
  sad ↔ sorrowful: target=0.408, pred=0.601
  house ↔ home: target=0.804, pred=0.504
  road ↔ street: target=0.806, pred=0.406
  forest ↔ woods: target=0.817, pred=0.339
  ocean ↔ sea: target=0.871, pred=0.717
  hill ↔ mountain: target=0.581, pred=0.280
  doctor ↔ physician: target=0.860, pred=0.317
  lawyer ↔ attorney: target=0.887, pred=0.420
  movie ↔ film: target=0.843, pred=0.651
  song ↔ music: target=0.598, pred=0.576
  book ↔ novel: target=0.766, pred=0.560
  child ↔ kid: target=0.810, pred=0.658
  woman ↔ lady: target=0.773, pred=0.484
  man ↔ gentleman: target=0.538, pred=0.506
  stone ↔ rock: target=0.706, pred=0.475
  river ↔ stream: target=0.521, pred=0.552

======================================================================
BATCH 2: Pairs 26-50
======================================================================
Transpiling 25 circuits...
Submitted job: d4igqp2v0j9c73e2b070
Waiting for job to complete...
Job finished!
  dog ↔ cat: target=0.661, pred=0.331
  car ↔ truck: target=0.689, pred=0.424
  tree ↔ plant: target=0.584, pred=0.431
  bird ↔ animal: target=0.639, pred=0.455
  apple ↔ fruit: target=0.537, pred=0.525
  coffee ↔ tea: target=0.616, pred=0.498
  bread ↔ cake: target=0.474, pred=0.516
  chair ↔ table: target=0.447, pred=0.594
  shirt ↔ pants: target=0.552, pred=0.796
  pen ↔ pencil: target=0.673, pred=0.438
  school ↔ teacher: target=0.617, pred=0.530
  hospital ↔ doctor: target=0.597, pred=0.289
  kitchen ↔ cooking: target=0.653, pred=0.495
  garden ↔ flower: target=0.603, pred=0.448
  office ↔ work: target=0.521, pred=0.449
  rain ↔ weather: target=0.669, pred=0.457
  night ↔ moon: target=0.406, pred=0.571
  summer ↔ heat: target=0.420, pred=0.614
  winter ↔ snow: target=0.805, pred=0.604
  spring ↔ flower: target=0.387, pred=0.581
  run ↔ walk: target=0.457, pred=0.529
  read ↔ write: target=0.564, pred=0.483
  eat ↔ drink: target=0.515, pred=0.579
  sleep ↔ rest: target=0.552, pred=0.667
  talk ↔ speak: target=0.726, pred=0.618

======================================================================
BATCH 3: Pairs 51-75
======================================================================
Transpiling 25 circuits...
Submitted job: d4igqu43tdfc73dmfhu0
Waiting for job to complete...
Job finished!
  dog ↔ computer: target=0.425, pred=0.315
  car ↔ happiness: target=0.388, pred=0.265
  tree ↔ mathematics: target=0.227, pred=0.391
  music ↔ geology: target=0.348, pred=0.506
  book ↔ volcano: target=0.246, pred=0.586
  love ↔ hammer: target=0.254, pred=0.584
  fear ↔ bicycle: target=0.285, pred=0.591
  hope ↔ refrigerator: target=0.164, pred=0.338
  anger ↔ telescope: target=0.153, pred=0.318
  joy ↔ microscope: target=0.101, pred=0.364
  banana ↔ democracy: target=0.186, pred=0.616
  guitar ↔ philosophy: target=0.210, pred=0.376
  pizza ↔ astronomy: target=0.235, pred=0.487
  soccer ↔ quantum: target=0.132, pred=0.669
  dance ↔ chemistry: target=0.315, pred=0.557
  elephant ↔ internet: target=0.314, pred=0.511
  coffee ↔ pyramid: target=0.179, pred=0.477
  tennis ↔ bacteria: target=0.244, pred=0.569
  painting ↔ gravity: target=0.180, pred=0.628
  violin ↔ tornado: target=0.113, pred=0.437
  sandwich ↔ algebra: target=0.224, pred=0.308
  laptop ↔ rainbow: target=0.172, pred=0.550
  bicycle ↔ poetry: target=0.266, pred=0.421
  umbrella ↔ politics: target=0.234, pred=0.558
  camera ↔ earthquake: target=0.223, pred=0.634

======================================================================
LARGE-SCALE VALIDATION RESULTS
======================================================================

Sample Size: 75 pairs
Overall Correlation: 0.0179
P-value: 8.79e-01
Statistically Significant: NO

----------------------------------------------------------------------
BY CATEGORY:
----------------------------------------------------------------------

HIGH (n=25):
  Target range: [0.408, 0.912]
  Pred range:   [0.280, 0.785]
  Mean target:  0.734
  Mean pred:    0.520
  Mean error:   -0.214 ± 0.216
  Correlation:  -0.232 (p=2.652e-01)

MEDIUM (n=25):
  Target range: [0.387, 0.805]
  Pred range:   [0.289, 0.796]
  Mean target:  0.575
  Mean pred:    0.517
  Mean error:   -0.058 ± 0.166
  Correlation:  -0.280 (p=1.748e-01)

LOW (n=25):
  Target range: [0.101, 0.425]
  Pred range:   [0.265, 0.669]
  Mean target:  0.233
  Mean pred:    0.482
  Mean error:   +0.250 ± 0.152
  Correlation:  -0.155 (p=4.590e-01)

----------------------------------------------------------------------
ORDERING CHECK:
----------------------------------------------------------------------
  HIGH mean:   0.520
  MEDIUM mean: 0.517
  LOW mean:    0.482
  ✅ Predictions correctly ordered: HIGH > MEDIUM > LOW

----------------------------------------------------------------------
ERROR ANALYSIS:
----------------------------------------------------------------------
  Mean absolute error: 0.226
  RMSE: 0.264
  Systematic bias: -0.007
  ✅ No significant systematic bias

======================================================================
VALIDATION VERDICT
======================================================================
❌ Correlation ≥ 0.70 (achieved: 0.018)
❌ Not statistically significant (p = 0.879)
✅ Predictions correctly ordered

❌ VALIDATION FAILED

Error: Object of type bool is not JSON serializable
Traceback (most recent call last):
  File "/home/qstar/AI/DiscoveryAI/paper4/experiments/large_scale_validation.py", line 452, in <module>
    correlation, results = run_large_scale_validation()
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qstar/AI/DiscoveryAI/paper4/experiments/large_scale_validation.py", line 444, in run_large_scale_validation
    json.dump(output, f, indent=2)
  File "/usr/lib/python3.12/json/__init__.py", line 179, in dump
    for chunk in iterable:
  File "/usr/lib/python3.12/json/encoder.py", line 432, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/usr/lib/python3.12/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.12/json/encoder.py", line 439, in _iterencode
    o = _default(o)
        ^^^^^^^^^^^
  File "/usr/lib/python3.12/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type bool is not JSON serializable

```

## Success Criteria

- ✅ Correlation ≥ 0.70 (confirms small sample wasn't lucky)
- ✅ Statistically significant (p < 0.01)
- ✅ Consistent across similarity ranges
- ✅ Predictions correctly ordered: HIGH > MED > LOW

## Starting Point
IF the experiment was a failure, then remove the compression, maybe that is the wrong part and what we want to test is if the quantum can train