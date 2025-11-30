#!/usr/bin/env python3
"""
NO COMPRESSION TEST: Direct Similarity Encoding
=================================================
Tests whether quantum circuits can capture semantic similarity
when given MORE DIRECT access to the similarity signal.

Hypothesis: The 384D → 12D PCA compression destroys fine-grained
semantic distinctions, causing all predictions to collapse to ~0.5

This experiment tests THREE encodings:
1. DIFFERENCE encoding: encode |v1 - v2| directly (similarity as magnitude)
2. CONCAT encoding: concatenate PCA components (10D each = 20 qubits)
3. DIRECT encoding: encode cosine similarity as a single angle

If none work, the problem is NOT compression - it's the circuit architecture.
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
from sentence_transformers import SentenceTransformer

# Configuration
BACKEND_NAME = "ibm_fez"
N_QUBITS = 20  # More qubits = less compression
SHOTS = 4096

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
    # Normalize: 0 Hamming weight = 1.0 similarity, n_qubits/2 = 0.0
    similarity = 1.0 - (avg_hamming / (n_qubits / 2))
    return float(max(0.0, min(1.0, similarity)))


def build_difference_circuit(diff_vector: np.ndarray) -> QuantumCircuit:
    """
    ENCODING 1: Encode the difference vector |v1 - v2|.

    If v1 ≈ v2, diff is small → small angles → close to |000...0⟩
    If v1 ≠ v2, diff is large → large angles → far from |000...0⟩

    This directly encodes similarity information!
    """
    n_qubits = len(diff_vector)
    qc = QuantumCircuit(n_qubits)

    # Encode difference magnitude directly
    for i in range(n_qubits):
        qc.ry(float(diff_vector[i]), i)

    # Local entanglement (correlate dimensions)
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)

    qc.measure_all()
    return qc


def build_concat_circuit(v1_10d: np.ndarray, v2_10d: np.ndarray) -> QuantumCircuit:
    """
    ENCODING 2: Concatenate top-10 PCA components from each vector.

    Uses 20 qubits total: qubits 0-9 for v1, qubits 10-19 for v2.
    Then applies cross-entanglement to compare.
    """
    qc = QuantumCircuit(20)

    # Encode v1 on first 10 qubits
    for i in range(10):
        qc.ry(float(v1_10d[i]), i)

    # Encode v2 on last 10 qubits
    for i in range(10):
        qc.ry(float(v2_10d[i]), i + 10)

    # Cross-entanglement: compare corresponding dimensions
    for i in range(10):
        qc.cx(i, i + 10)  # Entangle v1[i] with v2[i]

    qc.measure_all()
    return qc


def build_direct_circuit(cosine_sim: float) -> QuantumCircuit:
    """
    ENCODING 3: Encode cosine similarity DIRECTLY as a single angle.

    This is the ultimate test: if we TELL the circuit the answer,
    can it recover it?

    angle = π * (1 - cosine_sim)
    If cosine = 1.0 → angle = 0 → |0⟩ → hamming = 0 → pred = 1.0
    If cosine = 0.0 → angle = π → |1⟩ → hamming = 1 → pred = 0.5
    """
    qc = QuantumCircuit(N_QUBITS)

    # Spread the similarity across all qubits (noise averaging)
    angle = np.pi * (1.0 - cosine_sim)

    for i in range(N_QUBITS):
        qc.ry(angle, i)

    # Entangle for error correction
    for i in range(0, N_QUBITS - 1, 2):
        qc.cx(i, i + 1)

    qc.measure_all()
    return qc


def get_test_pairs():
    """Same 75 pairs as large_scale_validation for fair comparison."""
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

    return high_pairs, medium_pairs, low_pairs


def run_encoding_test(encoding_name: str, circuit_builder, pairs, embeddings, concepts, backend):
    """Run a single encoding test."""
    print(f"\n{'='*70}")
    print(f"ENCODING: {encoding_name}")
    print(f"{'='*70}")

    # Build circuits
    circuits = []
    targets = []

    for c1, c2 in pairs:
        idx1 = concepts.index(c1)
        idx2 = concepts.index(c2)
        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]

        # Target is always raw 384D cosine
        target = cosine_similarity(emb1, emb2)
        targets.append(target)

        if encoding_name == "DIFFERENCE":
            # Compute difference and compress to 20D
            diff = emb1 - emb2
            diff_norm = np.abs(diff)  # Magnitude of difference
            # Simple compression: take top-20 components by variance
            top_indices = np.argsort(np.abs(diff))[-N_QUBITS:]
            diff_20d = diff_norm[top_indices]
            # Scale to [0, π]
            diff_20d = diff_20d / (np.max(diff_20d) + 1e-10) * np.pi
            qc = circuit_builder(diff_20d)

        elif encoding_name == "CONCAT":
            # Use top-10 PCA components for each
            from sklearn.decomposition import PCA
            # Fit PCA on full dataset
            if not hasattr(run_encoding_test, 'pca'):
                run_encoding_test.pca = PCA(n_components=10)
                run_encoding_test.pca.fit(embeddings)
            v1_10d = run_encoding_test.pca.transform([emb1])[0]
            v2_10d = run_encoding_test.pca.transform([emb2])[0]
            # Scale to [0, π]
            v1_10d = (v1_10d - v1_10d.min()) / (v1_10d.max() - v1_10d.min() + 1e-10) * np.pi
            v2_10d = (v2_10d - v2_10d.min()) / (v2_10d.max() - v2_10d.min() + 1e-10) * np.pi
            qc = circuit_builder(v1_10d, v2_10d)

        elif encoding_name == "DIRECT":
            qc = circuit_builder(target)

        circuits.append(qc)

    # Transpile
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    print(f"Transpiling {len(circuits)} circuits...")
    isa_circuits = pm.run(circuits)

    # Run
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
        pred = compute_similarity_from_counts(counts, N_QUBITS)
        preds.append(pred)

    # Compute correlation
    targets = np.array(targets)
    preds = np.array(preds)

    if np.std(targets) > 0.01 and np.std(preds) > 0.01:
        correlation, p_value = stats.pearsonr(preds, targets)
    else:
        correlation, p_value = 0.0, 1.0

    print(f"\nResults for {encoding_name}:")
    print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"  Pred range:   [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  Correlation:  {correlation:.4f} (p={p_value:.3e})")

    return {
        'encoding': encoding_name,
        'correlation': float(correlation),
        'p_value': float(p_value),
        'targets': targets.tolist(),
        'preds': preds.tolist()
    }


def main():
    print("=" * 70)
    print("NO COMPRESSION TEST: Direct Similarity Encoding")
    print("=" * 70)
    print("Testing if compression (384D→12D) is the failure mode")
    print("=" * 70)

    # Get pairs
    high_pairs, medium_pairs, low_pairs = get_test_pairs()
    all_pairs = high_pairs + medium_pairs + low_pairs

    print(f"\nTotal pairs: {len(all_pairs)}")

    # Load embeddings
    print("\nLoading sentence-transformers model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    all_concepts = list(set(c for pair in all_pairs for c in pair))
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

    # TEST 1: DIRECT encoding (ultimate sanity check)
    # If this fails, the circuit architecture itself is broken
    print("\n" + "="*70)
    print("TEST 1: DIRECT ENCODING (Sanity Check)")
    print("Encodes cosine similarity DIRECTLY as circuit angles")
    print("If this fails, circuit architecture is fundamentally broken")
    print("="*70)

    direct_results = run_encoding_test(
        "DIRECT",
        build_direct_circuit,
        all_pairs[:25],  # Use 25 pairs for quick test
        embeddings,
        all_concepts,
        backend
    )

    # Decide whether to continue based on DIRECT test
    if direct_results['correlation'] < 0.5:
        print("\n⚠️  DIRECT encoding failed! Circuit cannot even preserve given similarity.")
        print("Problem is NOT compression - it's the circuit/measurement architecture.")
        print("Skipping other encodings.")

        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'verdict': 'CIRCUIT_ARCHITECTURE_BROKEN',
            'direct_encoding': direct_results,
            'message': 'Circuit cannot preserve directly-encoded similarity'
        }
    else:
        print("\n✅ DIRECT encoding works! Testing other encodings...")

        # TEST 2: DIFFERENCE encoding
        diff_results = run_encoding_test(
            "DIFFERENCE",
            build_difference_circuit,
            all_pairs,
            embeddings,
            all_concepts,
            backend
        )

        # TEST 3: CONCAT encoding
        concat_results = run_encoding_test(
            "CONCAT",
            build_concat_circuit,
            all_pairs,
            embeddings,
            all_concepts,
            backend
        )

        output = {
            'timestamp': datetime.now().isoformat(),
            'n_qubits': N_QUBITS,
            'shots': SHOTS,
            'direct_encoding': direct_results,
            'difference_encoding': diff_results,
            'concat_encoding': concat_results,
        }

        # Summary
        print("\n" + "="*70)
        print("SUMMARY: Encoding Comparison")
        print("="*70)
        print(f"  DIRECT:     r={direct_results['correlation']:.4f}")
        print(f"  DIFFERENCE: r={diff_results['correlation']:.4f}")
        print(f"  CONCAT:     r={concat_results['correlation']:.4f}")

        if diff_results['correlation'] > 0.5 or concat_results['correlation'] > 0.5:
            print("\n✅ Compression WAS the problem - better encoding works!")
            output['verdict'] = 'COMPRESSION_WAS_PROBLEM'
        else:
            print("\n❌ Compression NOT the only problem - circuit limitations exist")
            output['verdict'] = 'CIRCUIT_LIMITATIONS'

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"no_compression_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

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
