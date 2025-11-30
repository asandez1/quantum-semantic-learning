#!/usr/bin/env python3
"""
Large-Scale Validation of Quantum Semantic Encoding
====================================================
Tests 75 concept pairs across HIGH/MEDIUM/LOW similarity ranges
to validate the 0.73 correlation result from the 12-pair diagnostic.

Target: Confirm correlation >= 0.70 with statistical significance.
"""

import numpy as np
import json
from datetime import datetime
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from utils.data_preparation import QManifoldDataPreparation

# Configuration
BACKEND_NAME = "ibm_fez"
N_DATA_QUBITS = 12
SHOTS = 2048
BATCH_SIZE = 25  # Process in batches to avoid timeout


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(dot / (norm1 * norm2))


def compute_similarity_from_counts(counts: dict, n_qubits: int) -> float:
    """Compute similarity using normalized Hamming weight."""
    total_shots = sum(counts.values())
    weighted_hamming = 0.0
    for bitstring, count in counts.items():
        bs = bitstring.zfill(n_qubits)
        hamming_weight = bs.count('1')
        weighted_hamming += hamming_weight * count
    avg_hamming = weighted_hamming / total_shots
    similarity = 1.0 - (avg_hamming / (n_qubits / 2))
    return float(max(0.0, min(1.0, similarity)))


def build_attention_circuit(v1: np.ndarray, v2: np.ndarray) -> QuantumCircuit:
    """Build the 12-qubit attention circuit."""
    n_qubits = N_DATA_QUBITS
    q_data = QuantumRegister(n_qubits, 'q')
    c_main = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(q_data, c_main)

    # Encode v1
    for i in range(min(len(v1), n_qubits)):
        qc.ry(float(v1[i]), q_data[i])

    # Attention layer 1: local entanglement
    for i in range(0, n_qubits - 1, 2):
        qc.cx(q_data[i], q_data[i + 1])

    # Attention layer 2: cross-group entanglement
    qc.barrier()
    for i in range(n_qubits // 2):
        if i + n_qubits // 2 < n_qubits:
            qc.cx(q_data[i], q_data[i + n_qubits // 2])

    # Encode -v2 (interference)
    for i in range(min(len(v2), n_qubits)):
        qc.ry(float(-v2[i]), q_data[i])

    qc.measure(q_data, c_main)
    return qc


def get_test_pairs():
    """Get 75 diverse concept pairs spanning similarity ranges."""

    # HIGH similarity (cosine > 0.6) - 25 pairs
    high_pairs = [
        # Synonyms
        ('dog', 'puppy'),
        ('car', 'automobile'),
        ('happy', 'joyful'),
        ('big', 'large'),
        ('small', 'tiny'),
        ('fast', 'quick'),
        ('smart', 'intelligent'),
        ('beautiful', 'gorgeous'),
        ('angry', 'furious'),
        ('sad', 'sorrowful'),
        # Near-synonyms
        ('house', 'home'),
        ('road', 'street'),
        ('forest', 'woods'),
        ('ocean', 'sea'),
        ('hill', 'mountain'),
        # Related concepts
        ('doctor', 'physician'),
        ('lawyer', 'attorney'),
        ('movie', 'film'),
        ('song', 'music'),
        ('book', 'novel'),
        # More pairs
        ('child', 'kid'),
        ('woman', 'lady'),
        ('man', 'gentleman'),
        ('stone', 'rock'),
        ('river', 'stream'),
    ]

    # MEDIUM similarity (cosine 0.35-0.6) - 25 pairs
    medium_pairs = [
        # Related but different
        ('dog', 'cat'),
        ('car', 'truck'),
        ('tree', 'plant'),
        ('bird', 'animal'),
        ('apple', 'fruit'),
        # Same domain
        ('coffee', 'tea'),
        ('bread', 'cake'),
        ('chair', 'table'),
        ('shirt', 'pants'),
        ('pen', 'pencil'),
        # Conceptual relation
        ('school', 'teacher'),
        ('hospital', 'doctor'),
        ('kitchen', 'cooking'),
        ('garden', 'flower'),
        ('office', 'work'),
        # Moderate relation
        ('rain', 'weather'),
        ('night', 'moon'),
        ('summer', 'heat'),
        ('winter', 'snow'),
        ('spring', 'flower'),
        # Action pairs
        ('run', 'walk'),
        ('read', 'write'),
        ('eat', 'drink'),
        ('sleep', 'rest'),
        ('talk', 'speak'),
    ]

    # LOW similarity (cosine < 0.35) - 25 pairs
    low_pairs = [
        # Unrelated concepts
        ('dog', 'computer'),
        ('car', 'happiness'),
        ('tree', 'mathematics'),
        ('music', 'geology'),
        ('book', 'volcano'),
        # Abstract vs concrete
        ('love', 'hammer'),
        ('fear', 'bicycle'),
        ('hope', 'refrigerator'),
        ('anger', 'telescope'),
        ('joy', 'microscope'),
        # Different domains
        ('banana', 'democracy'),
        ('guitar', 'philosophy'),
        ('pizza', 'astronomy'),
        ('soccer', 'quantum'),
        ('dance', 'chemistry'),
        # Random pairings
        ('elephant', 'internet'),
        ('coffee', 'pyramid'),
        ('tennis', 'bacteria'),
        ('painting', 'gravity'),
        ('violin', 'tornado'),
        # More unrelated
        ('sandwich', 'algebra'),
        ('laptop', 'rainbow'),
        ('bicycle', 'poetry'),
        ('umbrella', 'politics'),
        ('camera', 'earthquake'),
    ]

    return high_pairs, medium_pairs, low_pairs


def run_large_scale_validation():
    """Run validation on 75 concept pairs."""

    print("=" * 70)
    print("LARGE-SCALE VALIDATION: 75 CONCEPT PAIRS")
    print("=" * 70)
    print("Testing correlation between quantum circuit and cosine similarity")
    print("Target: Correlation >= 0.70 with p < 0.01")
    print("=" * 70)

    # Get test pairs
    high_pairs, medium_pairs, low_pairs = get_test_pairs()
    all_pairs = high_pairs + medium_pairs + low_pairs

    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"  HIGH similarity:   {len(high_pairs)} pairs")
    print(f"  MEDIUM similarity: {len(medium_pairs)} pairs")
    print(f"  LOW similarity:    {len(low_pairs)} pairs")

    # Initialize data prep
    data_prep = QManifoldDataPreparation(target_dim=N_DATA_QUBITS)

    # Setup hardware
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::",
    
    )
    backend = service.backend(BACKEND_NAME)
    print(f"\nBackend: {backend.name} ({backend.num_qubits} qubits)")

    # Get all unique concepts
    all_concepts = list(set(c for pair in all_pairs for c in pair))
    print(f"Unique concepts: {len(all_concepts)}")

    # Embed concepts
    print("\nGenerating embeddings...")
    embeddings = data_prep.embed_concepts(all_concepts)

    # Get scaled vectors for circuit encoding
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # Process in batches
    all_results = []
    batch_num = 0

    for batch_start in range(0, len(all_pairs), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_pairs))
        batch_pairs = all_pairs[batch_start:batch_end]
        batch_num += 1

        print(f"\n{'=' * 70}")
        print(f"BATCH {batch_num}: Pairs {batch_start+1}-{batch_end}")
        print(f"{'=' * 70}")

        # Build circuits for this batch
        circuits = []
        for c1, c2 in batch_pairs:
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)
            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]
            circuits.append(build_attention_circuit(v1, v2))

        # Transpile
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        print(f"Transpiling {len(circuits)} circuits...")
        isa_circuits = pm.run(circuits)

        # Submit job
        sampler = SamplerV2(mode=backend)
        job = sampler.run(isa_circuits, shots=SHOTS)
        print(f"Submitted job: {job.job_id()}")
        print("Waiting for job to complete...")

        result = job.result()
        print("Job finished!")

        # Process results
        for i, (c1, c2) in enumerate(batch_pairs):
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)

            # Cosine similarity from RAW embeddings
            target = cosine_similarity(embeddings[idx1], embeddings[idx2])

            # Circuit prediction
            counts = result[i].data.c.get_counts()
            pred = compute_similarity_from_counts(counts, N_DATA_QUBITS)

            # Categorize
            if batch_start + i < len(high_pairs):
                category = "HIGH"
            elif batch_start + i < len(high_pairs) + len(medium_pairs):
                category = "MEDIUM"
            else:
                category = "LOW"

            all_results.append({
                'pair': (c1, c2),
                'target': float(target),
                'pred': float(pred),
                'error': float(pred - target),
                'category': category
            })

            print(f"  {c1} ↔ {c2}: target={target:.3f}, pred={pred:.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("LARGE-SCALE VALIDATION RESULTS")
    print("=" * 70)

    targets = np.array([r['target'] for r in all_results])
    preds = np.array([r['pred'] for r in all_results])
    errors = preds - targets

    # Overall statistics
    correlation, p_value = stats.pearsonr(preds, targets)

    print(f"\nSample Size: {len(all_results)} pairs")
    print(f"Overall Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Statistically Significant: {'YES' if p_value < 0.01 else 'NO'}")

    # By category
    print("\n" + "-" * 70)
    print("BY CATEGORY:")
    print("-" * 70)

    category_stats = {}
    for cat in ['HIGH', 'MEDIUM', 'LOW']:
        cat_data = [r for r in all_results if r['category'] == cat]
        if cat_data:
            cat_targets = np.array([r['target'] for r in cat_data])
            cat_preds = np.array([r['pred'] for r in cat_data])
            cat_errors = cat_preds - cat_targets

            if len(cat_targets) >= 3 and np.std(cat_targets) > 0.01:
                cat_corr, cat_p = stats.pearsonr(cat_preds, cat_targets)
            else:
                cat_corr, cat_p = 0.0, 1.0

            print(f"\n{cat} (n={len(cat_data)}):")
            print(f"  Target range: [{cat_targets.min():.3f}, {cat_targets.max():.3f}]")
            print(f"  Pred range:   [{cat_preds.min():.3f}, {cat_preds.max():.3f}]")
            print(f"  Mean target:  {cat_targets.mean():.3f}")
            print(f"  Mean pred:    {cat_preds.mean():.3f}")
            print(f"  Mean error:   {cat_errors.mean():+.3f} ± {cat_errors.std():.3f}")
            print(f"  Correlation:  {cat_corr:.3f} (p={cat_p:.3e})")

            category_stats[cat] = {
                'n': len(cat_data),
                'mean_target': float(cat_targets.mean()),
                'mean_pred': float(cat_preds.mean()),
                'mean_error': float(cat_errors.mean()),
                'std_error': float(cat_errors.std()),
                'correlation': float(cat_corr),
                'p_value': float(cat_p)
            }

    # Check ordering
    print("\n" + "-" * 70)
    print("ORDERING CHECK:")
    print("-" * 70)

    high_mean = category_stats['HIGH']['mean_pred']
    med_mean = category_stats['MEDIUM']['mean_pred']
    low_mean = category_stats['LOW']['mean_pred']

    print(f"  HIGH mean:   {high_mean:.3f}")
    print(f"  MEDIUM mean: {med_mean:.3f}")
    print(f"  LOW mean:    {low_mean:.3f}")

    if high_mean > med_mean > low_mean:
        print("  ✅ Predictions correctly ordered: HIGH > MEDIUM > LOW")
        ordering_correct = True
    else:
        print("  ❌ Ordering incorrect!")
        ordering_correct = False

    # Error analysis
    print("\n" + "-" * 70)
    print("ERROR ANALYSIS:")
    print("-" * 70)

    print(f"  Mean absolute error: {np.abs(errors).mean():.3f}")
    print(f"  RMSE: {np.sqrt((errors**2).mean()):.3f}")
    print(f"  Systematic bias: {errors.mean():+.3f}")

    if errors.mean() > 0.05:
        print("  ⚠️  Predictions tend to overestimate similarity")
    elif errors.mean() < -0.05:
        print("  ⚠️  Predictions tend to underestimate (compress) similarity")
    else:
        print("  ✅ No significant systematic bias")

    # Summary verdict
    print("\n" + "=" * 70)
    print("VALIDATION VERDICT")
    print("=" * 70)

    success_criteria = []

    if correlation >= 0.70:
        success_criteria.append(f"✅ Correlation ≥ 0.70 (achieved: {correlation:.3f})")
    else:
        success_criteria.append(f"❌ Correlation ≥ 0.70 (achieved: {correlation:.3f})")

    if p_value < 0.01:
        success_criteria.append(f"✅ Statistically significant (p < 0.01)")
    else:
        success_criteria.append(f"❌ Not statistically significant (p = {p_value:.3f})")

    if ordering_correct:
        success_criteria.append("✅ Predictions correctly ordered")
    else:
        success_criteria.append("❌ Ordering incorrect")

    for criterion in success_criteria:
        print(criterion)

    overall_success = correlation >= 0.70 and p_value < 0.01 and ordering_correct
    print(f"\n{'✅ VALIDATION PASSED' if overall_success else '❌ VALIDATION FAILED'}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'backend': BACKEND_NAME,
        'n_qubits': N_DATA_QUBITS,
        'shots': SHOTS,
        'n_pairs': len(all_results),
        'overall_correlation': float(correlation),
        'p_value': float(p_value),
        'ordering_correct': ordering_correct,
        'validation_passed': overall_success,
        'category_stats': category_stats,
        'error_stats': {
            'mean_absolute_error': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt((errors**2).mean())),
            'systematic_bias': float(errors.mean())
        },
        'results': all_results
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"large_scale_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return correlation, all_results


if __name__ == "__main__":
    try:
        correlation, results = run_large_scale_validation()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
