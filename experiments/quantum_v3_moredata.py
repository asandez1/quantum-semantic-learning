#!/usr/bin/env python3
"""
V3 MORE DATA: Same Architecture (7 qubits), More Training Pairs
================================================================

Keep the proven V3 architecture, just train on more data.
This tests if the generalization gap is due to data, not capacity.
"""

import numpy as np
import json
import os
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

CONFIG = {
    "n_qubits_per_vector": 3,  # SAME as original V3
    "n_ancilla": 1,
    "shots": 4096,
    "spsa_iterations": 400,  # More iterations
    "spsa_lr": 0.15,
    "use_hardware": True,
    "backend_name": "ibm_fez",
}

# EXPANDED training set (40 pairs instead of 12)
TRAIN_SIMILAR = [
    ('dog', 'puppy'), ('cat', 'kitten'), ('horse', 'pony'), ('wolf', 'dog'),
    ('lion', 'tiger'), ('mouse', 'rat'), ('bird', 'sparrow'), ('fish', 'salmon'),
    ('bear', 'grizzly'), ('deer', 'elk'), ('rabbit', 'hare'), ('pig', 'boar'),
    ('car', 'automobile'), ('phone', 'telephone'), ('computer', 'laptop'),
    ('chair', 'seat'), ('happy', 'joy'), ('sad', 'sorrow'), ('love', 'affection'),
    ('tree', 'plant'),
]

TRAIN_DISSIMILAR = [
    ('dog', 'car'), ('cat', 'mountain'), ('bird', 'computer'), ('fish', 'happiness'),
    ('horse', 'music'), ('lion', 'book'), ('eagle', 'chair'), ('shark', 'flower'),
    ('bear', 'telephone'), ('wolf', 'painting'), ('mouse', 'ocean'), ('snake', 'cloud'),
    ('car', 'happiness'), ('phone', 'tree'), ('computer', 'river'),
    ('chair', 'emotion'), ('table', 'dream'), ('book', 'wind'),
    ('tree', 'phone'), ('water', 'anger'),
]

# Test: Original 8 + more diverse pairs
TEST_PAIRS = [
    # Original 8
    ('eagle', 'hawk'), ('shark', 'dolphin'), ('cow', 'bull'),
    ('happy', 'car'), ('snake', 'lizard'), ('apple', 'rocket'),
    # Additional diverse pairs
    ('monkey', 'ape'), ('door', 'gate'), ('key', 'lock'),
    ('anger', 'rage'), ('fear', 'terror'), ('ocean', 'sea'),
    ('forest', 'woods'), ('sun', 'star'), ('bed', 'mattress'),
    # Dissimilar
    ('monkey', 'table'), ('door', 'emotion'), ('key', 'sadness'),
    ('fear', 'pencil'), ('hope', 'rock'), ('fire', 'happiness'),
]


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(dot / (n1 * n2)) if n1 > 1e-10 and n2 > 1e-10 else 0.0


def build_v3_circuit(v1, v2, theta):
    n = len(v1)
    n_total = 2 * n + 1
    ancilla = n_total - 1

    qc = QuantumCircuit(n_total)

    for i in range(n):
        qc.ry(float(v1[i]), i)
    for i in range(n):
        qc.ry(float(v2[i]), n + i)
    qc.barrier()

    for i in range(n_total):
        qc.ry(float(theta[i]), i)
    qc.barrier()

    for i in range(n):
        qc.cx(i, ancilla)
    for i in range(n):
        qc.cx(n + i, ancilla)
    qc.barrier()

    for i in range(n_total):
        qc.ry(float(theta[n_total + i]), i)
    for i in range(n):
        qc.cx(i, n + i)
    qc.barrier()

    for i in range(n_total):
        qc.ry(float(theta[2 * n_total + i]), i)
    qc.cx(0, ancilla)
    qc.cx(n, ancilla)

    qc.measure_all()
    return qc


def get_ancilla_prob(counts, n_total):
    total = sum(counts.values())
    ones = sum(c for b, c in counts.items() if b.replace(' ', '').zfill(n_total)[0] == '1')
    return ones / total


def bce(pred, target, eps=1e-7):
    pred = np.clip(pred, eps, 1 - eps)
    return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))


def train(sampler, train_data, n_total):
    n_params = n_total * 3
    theta = np.random.uniform(-0.01, 0.01, n_params)
    best_theta, best_loss = theta.copy(), float('inf')

    print(f"  Training: {CONFIG['spsa_iterations']} iters, {n_params} params, {len(train_data)} pairs")

    for it in range(CONFIG['spsa_iterations']):
        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        c_k = 0.1 / (it + 1) ** 0.101
        a_k = CONFIG['spsa_lr'] / (it + 1) ** 0.602

        theta_p = theta + c_k * delta
        theta_m = theta - c_k * delta

        # Mini-batch
        batch_idx = np.random.choice(len(train_data), min(12, len(train_data)), replace=False)
        batch = [train_data[i] for i in batch_idx]

        circuits = []
        for v1, v2, _ in batch:
            circuits.append(build_v3_circuit(v1, v2, theta_p))
            circuits.append(build_v3_circuit(v1, v2, theta_m))

        result = sampler.run(circuits, shots=CONFIG['shots']).result()

        loss_p, loss_m = 0.0, 0.0
        for i, (_, _, t) in enumerate(batch):
            loss_p += bce(get_ancilla_prob(result[2*i].data.meas.get_counts(), n_total), t)
            loss_m += bce(get_ancilla_prob(result[2*i+1].data.meas.get_counts(), n_total), t)
        loss_p /= len(batch)
        loss_m /= len(batch)

        theta = theta - a_k * (loss_p - loss_m) / (2 * c_k) * delta
        curr_loss = (loss_p + loss_m) / 2

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_theta = theta.copy()

        if (it + 1) % 100 == 0:
            print(f"    Iter {it+1}: loss={curr_loss:.4f} (best={best_loss:.4f})")

    return best_theta


def evaluate(sampler, test_data, theta, n_total, backend=None):
    circuits = [build_v3_circuit(v1, v2, theta) for v1, v2, _ in test_data]

    if backend:
        circuits = transpile(circuits, backend=backend, optimization_level=1)
        print(f"  Transpiled {len(circuits)} circuits, depth: {np.mean([c.depth() for c in circuits]):.0f}")

    preds = []
    for i in range(0, len(circuits), 20):
        batch = circuits[i:i+20]
        result = sampler.run(batch, shots=CONFIG['shots']).result()
        for j in range(len(batch)):
            preds.append(1.0 - get_ancilla_prob(result[j].data.meas.get_counts(), n_total))

    targets = [d[2] for d in test_data]
    corr, p_val = stats.pearsonr(preds, targets)
    return np.array(preds), np.array(targets), corr, p_val


def main():
    print("=" * 60)
    print("V3 MORE DATA: Same 7-qubit architecture, 40 training pairs")
    print("=" * 60)

    np.random.seed(42)

    # Data prep
    all_pairs = TRAIN_SIMILAR + TRAIN_DISSIMILAR + TEST_PAIRS
    all_concepts = list(set(c for p in all_pairs for c in p))
    print(f"Concepts: {len(all_concepts)}")

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb_raw = model.encode(all_concepts, show_progress_bar=False)

    pca = PCA(n_components=CONFIG['n_qubits_per_vector'])
    emb_pca = pca.fit_transform(emb_raw)
    print(f"PCA variance: {sum(pca.explained_variance_ratio_):.3f}")

    scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))
    vectors = scaler.fit_transform(emb_pca)

    # Training data
    train_data = []
    for c1, c2 in TRAIN_SIMILAR:
        i1, i2 = all_concepts.index(c1), all_concepts.index(c2)
        train_data.append((vectors[i1], vectors[i2], 0.0))
    for c1, c2 in TRAIN_DISSIMILAR:
        i1, i2 = all_concepts.index(c1), all_concepts.index(c2)
        train_data.append((vectors[i1], vectors[i2], 1.0))
    np.random.shuffle(train_data)
    print(f"Training pairs: {len(train_data)}")

    # Test data
    test_data = []
    for c1, c2 in TEST_PAIRS:
        i1, i2 = all_concepts.index(c1), all_concepts.index(c2)
        sim = cosine_similarity(emb_raw[i1], emb_raw[i2])
        test_data.append((vectors[i1], vectors[i2], sim))
    print(f"Test pairs: {len(test_data)}")

    n_total = CONFIG['n_qubits_per_vector'] * 2 + CONFIG['n_ancilla']
    n_params = n_total * 3

    weights_file = os.path.join(os.path.dirname(__file__), 'results', 'v3_moredata_theta.json')

    if CONFIG['use_hardware']:
        print("\n[HARDWARE MODE]")
        service = QiskitRuntimeService(channel="ibm_cloud")
        backend = service.backend(CONFIG['backend_name'])
        sampler = SamplerV2(mode=backend)

        with open(weights_file) as f:
            theta = np.array(json.load(f)['best_theta'])

        preds, targets, corr, p_val = evaluate(sampler, test_data, theta, n_total, backend)
    else:
        print("\n[SIMULATION MODE]")
        sampler = SamplerV2(mode=AerSimulator())

        # Random baseline
        theta_rand = np.random.uniform(-np.pi, np.pi, n_params)
        _, _, corr_rand, _ = evaluate(sampler, test_data, theta_rand, n_total)
        print(f"Random: {corr_rand:.4f}")

        # Train
        theta = train(sampler, train_data, n_total)

        # Evaluate
        preds, targets, corr, p_val = evaluate(sampler, test_data, theta, n_total)
        print(f"\nTrained: {corr:.4f}")
        print(f"Training effect: {corr - corr_rand:+.4f}")

        # Save
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        with open(weights_file, 'w') as f:
            json.dump({
                'best_theta': theta.tolist(),
                'train_pairs': len(train_data),
                'sim_correlation': float(corr),
                'random_correlation': float(corr_rand),
            }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULT: r = {corr:.4f} (p = {p_val:.4f})")
    print("=" * 60)
    print("\n[COMPARISON]")
    print(f"Original V3 (12 pairs): r = 0.08 generalization")
    print(f"V3 MoreData (40 pairs): r = {corr:.2f} generalization")


if __name__ == "__main__":
    main()
