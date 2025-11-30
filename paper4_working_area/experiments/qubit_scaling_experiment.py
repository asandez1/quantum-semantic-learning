#!/usr/bin/env python3
"""
QUBIT SCALING EXPERIMENT: Finding the Sweet Spot
=================================================
Tests how correlation scales with qubit count using CONCAT encoding.

Previous results:
- 20 qubits (10D × 2): r = 0.586
- Target: r > 0.80

Hypothesis: More qubits = more PCA dimensions = better correlation.
IBM Fez has 156 qubits, so we can test up to ~140 qubits.

Test points: 20, 40, 60, 80, 100 qubits
"""

import numpy as np
import json
from datetime import datetime
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Configuration
BACKEND_NAME = "ibm_fez"
SHOTS = 4096
QUBIT_COUNTS = [20, 40, 60, 80, 100]  # Test these qubit counts


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


def build_concat_circuit(v1: np.ndarray, v2: np.ndarray, n_qubits: int) -> QuantumCircuit:
    """
    CONCAT encoding: Split qubits between v1 and v2 with cross-entanglement.

    For n_qubits total:
    - First n_qubits/2 encode v1
    - Last n_qubits/2 encode v2
    - Cross-CNOT gates compare corresponding dimensions
    """
    half = n_qubits // 2
    qc = QuantumCircuit(n_qubits)

    # Encode v1 on first half
    for i in range(min(len(v1), half)):
        qc.ry(float(v1[i]), i)

    # Encode v2 on second half
    for i in range(min(len(v2), half)):
        qc.ry(float(v2[i]), i + half)

    # Cross-entanglement: compare corresponding dimensions
    for i in range(half):
        qc.cx(i, i + half)

    # Additional local entanglement within each half (capture correlations)
    for i in range(0, half - 1, 2):
        qc.cx(i, i + 1)
        qc.cx(i + half, i + half + 1)

    qc.measure_all()
    return qc


def get_test_pairs():
    """Same 75 pairs for fair comparison."""
    high_pairs = [
        ('dog', 'puppy'), ('car', 'automobile'), ('happy', 'joyful'),
        ('big', 'large'), ('small', 'tiny'), ('fast', 'quick'),
        ('smart', 'intelligent'), ('beautiful', 'gorgeous'), ('angry', 'furious'),
        ('sad', 'sorrowful'), ('house', 'home'), ('road', 'street'),
        ('forest', 'woods'), ('ocean', 'sea'), ('hill', 'mountain'),
        ('doctor', 'physician'), ('lawyer', 'attorney'), ('movie', 'film'),
        ('song', 'music'), ('book', 'novel'), ('child', 'kid'),
        ('woman', 'lady'), ('man', 'gentleman'), ('stone', 'rock'),
        ('river', 'stream'),
    ]

    medium_pairs = [
        ('dog', 'cat'), ('car', 'truck'), ('tree', 'plant'),
        ('bird', 'animal'), ('apple', 'fruit'), ('coffee', 'tea'),
        ('bread', 'cake'), ('chair', 'table'), ('shirt', 'pants'),
        ('pen', 'pencil'), ('school', 'teacher'), ('hospital', 'doctor'),
        ('kitchen', 'cooking'), ('garden', 'flower'), ('office', 'work'),
        ('rain', 'weather'), ('night', 'moon'), ('summer', 'heat'),
        ('winter', 'snow'), ('spring', 'flower'), ('run', 'walk'),
        ('read', 'write'), ('eat', 'drink'), ('sleep', 'rest'),
        ('talk', 'speak'),
    ]

    low_pairs = [
        ('dog', 'computer'), ('car', 'happiness'), ('tree', 'mathematics'),
        ('music', 'geology'), ('book', 'volcano'), ('love', 'hammer'),
        ('fear', 'bicycle'), ('hope', 'refrigerator'), ('anger', 'telescope'),
        ('joy', 'microscope'), ('banana', 'democracy'), ('guitar', 'philosophy'),
        ('pizza', 'astronomy'), ('soccer', 'quantum'), ('dance', 'chemistry'),
        ('elephant', 'internet'), ('coffee', 'pyramid'), ('tennis', 'bacteria'),
        ('painting', 'gravity'), ('violin', 'tornado'), ('sandwich', 'algebra'),
        ('laptop', 'rainbow'), ('bicycle', 'poetry'), ('umbrella', 'politics'),
        ('camera', 'earthquake'),
    ]

    return high_pairs + medium_pairs + low_pairs


def run_qubit_test(n_qubits: int, pairs, embeddings, concepts, backend):
    """Run correlation test for a specific qubit count."""
    print(f"\n{'='*70}")
    print(f"TESTING {n_qubits} QUBITS ({n_qubits//2}D per vector)")
    print(f"{'='*70}")

    half = n_qubits // 2

    # Fit PCA for this dimension
    print(f"Fitting PCA: 384D → {half}D...")
    pca = PCA(n_components=half)
    pca.fit(embeddings)
    variance_explained = sum(pca.explained_variance_ratio_)
    print(f"Variance explained: {variance_explained:.3f}")

    # Transform all embeddings
    embeddings_reduced = pca.transform(embeddings)

    # Scale to [0, π]
    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    embeddings_scaled = scaler.fit_transform(embeddings_reduced)

    # Build circuits
    circuits = []
    targets = []

    for c1, c2 in pairs:
        idx1 = concepts.index(c1)
        idx2 = concepts.index(c2)

        # Target: raw 384D cosine similarity
        target = cosine_similarity(embeddings[idx1], embeddings[idx2])
        targets.append(target)

        # Build circuit with scaled vectors
        v1 = embeddings_scaled[idx1]
        v2 = embeddings_scaled[idx2]
        qc = build_concat_circuit(v1, v2, n_qubits)
        circuits.append(qc)

    # Transpile
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    print(f"Transpiling {len(circuits)} circuits...")
    isa_circuits = pm.run(circuits)

    # Check circuit depth
    sample_depth = isa_circuits[0].depth()
    print(f"Sample circuit depth: {sample_depth}")

    # Submit job
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"Submitted job: {job.job_id()}")
    print("Waiting for job to complete...")

    result = job.result()
    print("Job finished!")

    # Extract predictions
    preds = []
    for i in range(len(circuits)):
        counts = result[i].data.meas.get_counts()
        pred = compute_similarity_from_counts(counts, n_qubits)
        preds.append(pred)

    # Compute correlation
    targets = np.array(targets)
    preds = np.array(preds)

    correlation, p_value = stats.pearsonr(preds, targets)

    print(f"\nResults for {n_qubits} qubits:")
    print(f"  Variance explained: {variance_explained:.3f}")
    print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"  Pred range:   [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  Correlation:  {correlation:.4f} (p={p_value:.3e})")
    print(f"  Circuit depth: {sample_depth}")

    if correlation >= 0.80:
        print(f"  ✅ TARGET ACHIEVED! r >= 0.80")
    elif correlation >= 0.70:
        print(f"  ⚠️  Close to target (r >= 0.70)")
    else:
        print(f"  ❌ Below target")

    return {
        'n_qubits': n_qubits,
        'dim_per_vector': half,
        'variance_explained': float(variance_explained),
        'correlation': float(correlation),
        'p_value': float(p_value),
        'circuit_depth': sample_depth,
        'target_range': [float(targets.min()), float(targets.max())],
        'pred_range': [float(preds.min()), float(preds.max())],
        'targets': targets.tolist(),
        'preds': preds.tolist()
    }


def main():
    print("=" * 70)
    print("QUBIT SCALING EXPERIMENT")
    print("=" * 70)
    print(f"Testing: {QUBIT_COUNTS} qubits")
    print("Target: r >= 0.80")
    print("=" * 70)

    # Get pairs
    pairs = get_test_pairs()
    print(f"\nTotal pairs: {len(pairs)}")

    # Load embeddings
    print("\nLoading sentence-transformers model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    all_concepts = list(set(c for pair in pairs for c in pair))
    print(f"Unique concepts: {len(all_concepts)}")

    print("Generating 384D embeddings...")
    embeddings = model.encode(all_concepts, show_progress_bar=True)
    print(f"Embedding shape: {embeddings.shape}")

    # Setup hardware
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::",
    )
    backend = service.backend(BACKEND_NAME)
    print(f"\nBackend: {backend.name} ({backend.num_qubits} qubits)")

    # Run tests for each qubit count
    all_results = []

    for n_qubits in QUBIT_COUNTS:
        try:
            result = run_qubit_test(n_qubits, pairs, embeddings, all_concepts, backend)
            all_results.append(result)

            # Early stopping if we achieve target
            if result['correlation'] >= 0.85:
                print(f"\n🎉 EXCELLENT CORRELATION ACHIEVED! Stopping early.")
                break

        except Exception as e:
            print(f"\n❌ Error with {n_qubits} qubits: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Qubits':>8} | {'Dim/Vec':>8} | {'Var Expl':>8} | {'Corr':>8} | {'Status'}")
    print("-" * 70)

    for r in all_results:
        status = "✅ TARGET" if r['correlation'] >= 0.80 else "⚠️ CLOSE" if r['correlation'] >= 0.70 else "❌"
        print(f"{r['n_qubits']:>8} | {r['dim_per_vector']:>8} | {r['variance_explained']:>8.3f} | {r['correlation']:>8.4f} | {status}")

    # Find best result
    if all_results:
        best = max(all_results, key=lambda x: x['correlation'])
        print(f"\nBest result: {best['n_qubits']} qubits with r={best['correlation']:.4f}")

        if best['correlation'] >= 0.80:
            print("\n🎉 SUCCESS: Achieved r >= 0.80!")
            print(f"   Optimal configuration: {best['n_qubits']} qubits ({best['dim_per_vector']}D per vector)")
        else:
            # Extrapolate
            if len(all_results) >= 2:
                qubits = [r['n_qubits'] for r in all_results]
                corrs = [r['correlation'] for r in all_results]
                # Log-linear extrapolation
                log_qubits = np.log(qubits)
                slope, intercept = np.polyfit(log_qubits, corrs, 1)
                target_qubits = int(np.exp((0.80 - intercept) / slope))
                print(f"\n📈 Extrapolation: ~{target_qubits} qubits needed for r=0.80")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'backend': BACKEND_NAME,
        'shots': SHOTS,
        'n_pairs': len(pairs),
        'qubit_counts_tested': QUBIT_COUNTS,
        'results': all_results
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"qubit_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
