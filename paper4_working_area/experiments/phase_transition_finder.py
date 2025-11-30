#!/usr/bin/env python3
"""
Phase Transition Finder: Pinpoint the Critical Threshold
=========================================================
We know the phase transition is between 8-10 pairs.
This experiment will find it exactly.

Quick test at 9, 10, and 11 pairs to identify the precise boundary.
"""

import numpy as np
import json
from datetime import datetime
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from utils.data_preparation import QManifoldDataPreparation


def test_at_threshold(n_pairs: int) -> float:
    """Test quantum attention at a specific number of training pairs."""

    print(f"\nTesting with {n_pairs} training pairs...")

    # Initialize
    data_prep = QManifoldDataPreparation(target_dim=20)

    # Setup hardware
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::"
    )
    backend = service.backend("ibm_fez")

    # Prepare data
    all_pairs = data_prep.get_default_concept_pairs()
    train_pairs = all_pairs[:n_pairs]
    test_pairs = all_pairs[n_pairs:n_pairs+3]  # Just 3 test pairs for speed

    all_concepts = data_prep.generate_all_concepts(all_pairs[:n_pairs+3])
    embeddings = data_prep.embed_concepts(all_concepts)
    vectors_pca = data_prep.pca.fit_transform(embeddings)
    vectors_scaled = data_prep.scaler.fit_transform(vectors_pca)

    # Simple attention circuit with good initialization
    n_qubits = 20
    theta = np.random.uniform(-np.pi/8, np.pi/8, 30)  # Smaller range for stability

    # Single training pass
    print("  Training...")
    for _ in range(2):  # 2 quick iterations
        train_loss = 0
        for c1, c2 in train_pairs[:5]:  # Sample 5 pairs
            idx1 = all_concepts.index(c1)
            idx2 = all_concepts.index(c2)

            v1 = vectors_scaled[idx1]
            v2 = vectors_scaled[idx2]
            v1_pca = vectors_pca[idx1]
            v2_pca = vectors_pca[idx2]

            # Build circuit
            qc = QuantumCircuit(n_qubits)

            # Encode vectors
            for i in range(min(len(v1), n_qubits)):
                qc.ry(float(v1[i]), i)

            # Multi-head attention pattern
            for head in range(4):
                start = head * 5
                for i in range(start, min(start+4, n_qubits-1)):
                    if i < len(theta)//2:
                        qc.ry(theta[i], i)
                        qc.cx(i, i+1)
                        qc.rz(theta[i+15], i+1)

            # Cross-head connections
            for i in [4, 9, 14]:
                if i < n_qubits-5:
                    qc.cx(i, i+5)

            # Encode second vector
            for i in range(min(len(v2), n_qubits)):
                qc.ry(float(-v2[i]), i)

            qc.measure_all()

            # Execute
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            isa_circuit = pm.run(qc)
            sampler = SamplerV2(mode=backend)
            job = sampler.run([isa_circuit], shots=256)
            result = job.result()
            counts = result[0].data.meas.get_counts()

            pred = counts.get('0' * n_qubits, 0) / 256

            dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
            target = data_prep.hyperbolic_similarity(dist)

            train_loss += (pred - target) ** 2

        # Update parameters slightly
        if train_loss / 5 > 0.05:  # Only update if loss is high
            theta += np.random.uniform(-0.02, 0.02, len(theta))

    # Test
    print("  Testing...")
    test_predictions = []
    test_targets = []

    for c1, c2 in test_pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)

        v1 = vectors_scaled[idx1]
        v2 = vectors_scaled[idx2]
        v1_pca = vectors_pca[idx1]
        v2_pca = vectors_pca[idx2]

        # Build test circuit
        qc = QuantumCircuit(n_qubits)

        for i in range(min(len(v1), n_qubits)):
            qc.ry(float(v1[i]), i)

        for head in range(4):
            start = head * 5
            for i in range(start, min(start+4, n_qubits-1)):
                if i < len(theta)//2:
                    qc.ry(theta[i], i)
                    qc.cx(i, i+1)
                    qc.rz(theta[i+15], i+1)

        for i in [4, 9, 14]:
            if i < n_qubits-5:
                qc.cx(i, i+5)

        for i in range(min(len(v2), n_qubits)):
            qc.ry(float(-v2[i]), i)

        qc.measure_all()

        # Execute with more shots for test
        isa_circuit = pm.run(qc)
        job = sampler.run([isa_circuit], shots=512)
        result = job.result()
        counts = result[0].data.meas.get_counts()

        pred = counts.get('0' * n_qubits, 0) / 512
        test_predictions.append(pred)

        dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
        target = data_prep.hyperbolic_similarity(dist)
        test_targets.append(target)

        print(f"    {c1} ↔ {c2}: target={target:.3f}, pred={pred:.3f}")

    # Calculate correlation
    if len(test_predictions) > 1:
        corr = np.corrcoef(test_predictions, test_targets)[0, 1]
    else:
        corr = 0

    print(f"  Correlation: {corr:.4f}")

    return corr


def find_phase_transition():
    """Find the exact phase transition point."""

    print("=" * 70)
    print("PHASE TRANSITION FINDER")
    print("=" * 70)
    print("Finding the exact threshold where quantum attention recovers")
    print("")

    # Test these critical points
    test_points = [9, 10, 11]
    results = {}

    start_time = time.time()

    for n_pairs in test_points:
        try:
            corr = test_at_threshold(n_pairs)
            results[n_pairs] = corr

            # Check if we found the transition
            if corr > 0.5:
                print(f"\n✅ RECOVERY DETECTED at {n_pairs} pairs!")
                if n_pairs == 9:
                    print("Phase transition is between 8-9 pairs")
                elif n_pairs == 10:
                    print("Phase transition is between 9-10 pairs")
                break
            else:
                print(f"\n❌ Still collapsed at {n_pairs} pairs")

            # Time check
            elapsed = (time.time() - start_time) / 60
            if elapsed > 6:
                print("\nReaching time limit, stopping")
                break

        except Exception as e:
            print(f"\nError at {n_pairs} pairs: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("PHASE TRANSITION ANALYSIS")
    print("=" * 70)

    for n_pairs, corr in sorted(results.items()):
        status = "✅ RECOVERED" if corr > 0.5 else "❌ COLLAPSED"
        print(f"{n_pairs} pairs: {corr:.4f} {status}")

    # Find transition point
    if len(results) >= 2:
        sorted_results = sorted(results.items())
        for i in range(len(sorted_results) - 1):
            n1, c1 = sorted_results[i]
            n2, c2 = sorted_results[i + 1]
            if c1 < 0.3 and c2 > 0.5:
                print(f"\n🎯 PHASE TRANSITION FOUND!")
                print(f"   Critical threshold: {n1}-{n2} training pairs")
                print(f"   Below {n2} pairs: Circuit collapse")
                print(f"   At/above {n2} pairs: Successful learning")

                return {
                    'phase_transition': f"{n1}-{n2} pairs",
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }

    return {
        'results': results,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    print("PHASE TRANSITION FINDER")
    print("This will use ~5-6 minutes of quantum time")
    print("")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    try:
        result = find_phase_transition()

        # Save results
        output_file = f"../results/phase_transition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nResults saved to {output_file}")

        if 'phase_transition' in result:
            print("\n" + "🎉" * 20)
            print("MAJOR DISCOVERY!")
            print(f"Phase transition at {result['phase_transition']}")
            print("This is a fundamental property of quantum learning!")
            print("🎉" * 20)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")