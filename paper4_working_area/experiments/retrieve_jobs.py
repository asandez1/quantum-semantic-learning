#!/usr/bin/env python3
"""
Retrieve and Analyze Completed Qiskit Runtime Jobs
====================================================
This script fetches the results of previously run jobs from IBM Quantum
and calculates the correlation against the expected target values.
"""

import numpy as np
import sys
import os
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit_ibm_runtime import QiskitRuntimeService
from utils.data_preparation import QManifoldDataPreparation

def safe_corrcoef(preds, targets):
    """Calculate correlation safely, handling zero variance cases."""
    preds = np.array(preds)
    targets = np.array(targets)
    if len(preds) < 2 or np.std(preds) < 1e-12 or np.std(targets) < 1e-12:
        return 0.0
    return np.corrcoef(preds, targets)[0, 1]

def retrieve_and_analyze_jobs(job_ids: List[str]):
    """
    Retrieves job results, identifies their type (Standard or Dynamic),
    and computes the final correlation score.
    """
    print("=" * 70)
    print("Retrieving and Analyzing Completed Jobs")
    print("=" * 70)

    try:
        # --- Service and Data Initialization ---
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token="Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL",
            instance="crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"
        )
        data_prep = QManifoldDataPreparation(target_dim=20)
        all_pairs = data_prep.get_default_concept_pairs()[:30]
        
        n_pairs_train = 8
        train_pairs = all_pairs[:n_pairs_train]
        test_pairs = all_pairs[n_pairs_train:n_pairs_train + 3]

        # --- Corrected Data Preparation for Target Calculation ---
        all_concepts = []
        for pairs in [train_pairs, test_pairs]:
            for c1, c2 in pairs:
                all_concepts.extend([c1, c2])
        all_concepts = list(set(all_concepts))

        if len(all_concepts) < 22:
            extra_pairs = all_pairs[n_pairs_train + 3:]
            for c1, c2 in extra_pairs:
                all_concepts.extend([c1, c2])
                if len(set(all_concepts)) >= 22:
                    break
        all_concepts = list(set(all_concepts))

        embeddings = data_prep.embed_concepts(all_concepts)
        vectors_pca = data_prep.pca.fit_transform(embeddings)
        
        test_predictions = []
        test_targets = []
        job_modes = []

        # --- Job Retrieval and Processing ---
        for i, job_id in enumerate(job_ids):
            print(f"\n--- Retrieving Job {i+1}/{len(job_ids)}: {job_id} ---")
            
            try:
                job = service.job(job_id)
                circuit = None
                is_dynamic = False
                
                # *** CORRECT: Get circuit from 'pubs' for SamplerV2 ***
                if 'pubs' in job.inputs:
                    circuit = job.inputs['pubs'][0][0]
                elif 'circuits' in job.inputs: # Fallback for older formats
                    circuit = job.inputs['circuits'][0]
                else:
                    print(f"  ERROR: Cannot find circuit data in job {job_id}.")
                    print(f"  Available job input keys: {list(job.inputs.keys())}")
                    continue

                # Determine circuit type
                if len(circuit.cregs) > 1:
                    is_dynamic = True
                    print("Job identified as DYNAMIC (multiple classical registers)")
                else:
                    is_dynamic = False
                    print(f"Job identified as STANDARD ({circuit.num_clbits} classical bits)")
                job_modes.append("Dynamic" if is_dynamic else "Standard")

                # Get results (a single job from the old script corresponds to a single pub)
                result_data = job.result()[0].data
                
                # Parse counts based on circuit structure
                if is_dynamic:
                    counts = result_data.c.get_counts()
                    total_shots = sum(counts.values())
                    similarity = counts.get('0', 0) / total_shots
                else: # Handle old, single-register format
                    counts = result_data.c.get_counts()
                    total_shots = sum(counts.values())
                    bitstring_len = circuit.num_clbits
                    similarity = sum(c for bitstring, c in counts.items() if bitstring.zfill(bitstring_len)[:20] == '0'*20) / total_shots

                test_predictions.append(similarity)

                # Calculate corresponding target
                # Handle cases where there are fewer job IDs than test pairs
                if i >= len(test_pairs):
                    print(f"  Warning: More job IDs than test pairs. Cannot calculate target for job {job_id}.")
                    test_targets.append(0) # Append a dummy value
                    continue

                c1, c2 = test_pairs[i]
                idx1 = all_concepts.index(c1)
                idx2 = all_concepts.index(c2)
                v1_pca = vectors_pca[idx1]
                v2_pca = vectors_pca[idx2]
                dist = data_prep.compute_hyperbolic_distance(v1_pca, v2_pca)
                target = data_prep.hyperbolic_similarity(dist)
                test_targets.append(target)
                
                print(f"  {c1} ↔ {c2}: target={target:.4f}, pred={similarity:.4f}")

            except Exception as e:
                print(f"  ERROR processing job {job_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # --- Final Correlation Calculation ---
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        if not test_predictions:
            print("No job results could be successfully processed.")
            return

        # Infer overall mode from the first successfully processed job
        mode = job_modes[0] if job_modes else "Unknown"
        corr = safe_corrcoef(test_predictions, test_targets)
        signal_range = max(test_predictions) - min(test_predictions) if test_predictions else 0
        
        print(f"Mode: {mode}")
        print(f"Correlation: {corr:.4f}")
        print(f"Signal Range: {signal_range:.4f}")
        print("\nPredictions:", [round(p, 4) for p in test_predictions])
        print("Targets:    ", [round(t, 4) for t in test_targets])

    except Exception as e:
        print(f"\nAn overall error occurred: {e}")
        import traceback
        traceback.print_exc()


def debug_job(job_id: str):
    """Debug a specific job to understand why it failed."""
    print(f"\n{'=' * 70}")
    print(f"DEBUG JOB: {job_id}")
    print('=' * 70)

    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"
    )

    try:
        job = service.job(job_id)
        print(f"Status: {job.status()}")
        print(f"Backend: {job.backend().name if job.backend() else 'N/A'}")

        # Check for error message
        if hasattr(job, 'error_message') and callable(job.error_message):
            err = job.error_message()
            if err:
                print(f"Error Message: {err}")

        # Try to get circuit info
        if 'pubs' in job.inputs:
            circuit = job.inputs['pubs'][0][0]
            print(f"Circuit depth: {circuit.depth()}")
            print(f"Num qubits: {circuit.num_qubits}")
            print(f"Num classical registers: {len(circuit.cregs)}")
            for creg in circuit.cregs:
                print(f"  - {creg.name}: {creg.size} bits")

        # Try to get result anyway
        status = str(job.status())
        if status == "DONE" or "DONE" in status:
            result = job.result()
            print(f"Result type: {type(result)}")
            print(f"Num results: {len(result)}")
            for i, pub_result in enumerate(result):
                print(f"\nCircuit {i}:")
                data = pub_result.data
                # Filter to show only count-related attributes
                data_attrs = [a for a in dir(data) if not a.startswith('_')]
                print(f"  Data attributes: {data_attrs}")
                if hasattr(data, 'c'):
                    counts = data.c.get_counts()
                    total = sum(counts.values())
                    print(f"  Main register 'c' - {len(counts)} unique bitstrings, {total} shots")
                    top_counts = sorted(counts.items(), key=lambda x: -x[1])[:3]
                    for bs, cnt in top_counts:
                        print(f"    '{bs}': {cnt} ({cnt/total*100:.1f}%)")
                    # Check all-zeros specifically
                    all_zeros = '0' * 20
                    zero_cnt = counts.get(all_zeros, 0)
                    print(f"  All-zeros probability: {zero_cnt}/{total} = {zero_cnt/total:.6f}")
                if hasattr(data, 'syndrome'):
                    syn_counts = data.syndrome.get_counts()
                    print(f"  Syndrome register - {len(syn_counts)} unique values")
        elif "ERROR" in status:
            print("Job is in ERROR state - cannot retrieve results")

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


def list_recent_jobs(limit=20):
    """List recent jobs to find the ones we need."""
    print("=" * 70)
    print("RECENT IBM QUANTUM JOBS")
    print("=" * 70)

    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::",
    )

    jobs = service.jobs(limit=limit)
    print(f"\n{'Job ID':<28} | {'Status':<12} | {'Created':<20}")
    print("-" * 70)
    for job in jobs:
        status = str(job.status()).replace("JobStatus.", "")
        created = str(job.creation_date)[:19] if job.creation_date else "N/A"
        print(f"{job.job_id():<28} | {status:<12} | {created}")


def analyze_qubit_scaling_jobs():
    """Analyze the qubit scaling experiment jobs."""
    print("=" * 70)
    print("QUBIT SCALING EXPERIMENT - JOB ANALYSIS")
    print("=" * 70)

    # Job IDs from qubit_scaling_experiment.py
    QUBIT_JOBS = {
        20: "d4ihl190i6jc73dcqlc0",
        40: "d4ihlot74pkc73851sqg",
        # Add 60, 80, 100 if they were submitted
    }

    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token="HSP1Wlz3khkZBy8BvtynbTXJLS_6jWTiVYyeavQqXUsA",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/5dc1b48400fa455fa966bf1ce087ccf1:ff0e794b-e42a-4ead-a9a4-2b9a8444e29f::",
    )

    # Load embeddings for target calculation
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from scipy import stats

    print("\nLoading embeddings...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Same pairs as qubit_scaling_experiment
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
    pairs = high_pairs + medium_pairs + low_pairs

    all_concepts = list(set(c for pair in pairs for c in pair))
    embeddings = model.encode(all_concepts, show_progress_bar=False)

    def cosine_similarity(v1, v2):
        dot = np.dot(v1, v2)
        return float(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def compute_similarity_from_counts(counts, n_qubits):
        total_shots = sum(counts.values())
        weighted_hamming = 0.0
        for bitstring, count in counts.items():
            bs = bitstring.zfill(n_qubits)
            hamming_weight = bs.count('1')
            weighted_hamming += hamming_weight * count
        avg_hamming = weighted_hamming / total_shots
        similarity = 1.0 - (avg_hamming / (n_qubits / 2))
        return float(max(0.0, min(1.0, similarity)))

    # Calculate targets
    targets = []
    for c1, c2 in pairs:
        idx1 = all_concepts.index(c1)
        idx2 = all_concepts.index(c2)
        targets.append(cosine_similarity(embeddings[idx1], embeddings[idx2]))
    targets = np.array(targets)

    print(f"\nTargets: {len(targets)} pairs")
    print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")

    # Process each job
    results = []
    for n_qubits, job_id in QUBIT_JOBS.items():
        print(f"\n{'='*70}")
        print(f"{n_qubits} QUBITS - Job: {job_id}")
        print("=" * 70)

        try:
            job = service.job(job_id)
            status = str(job.status())
            print(f"Status: {status}")

            if "DONE" in status:
                result = job.result()
                print(f"Circuits: {len(result)}")

                preds = []
                for i in range(len(result)):
                    counts = result[i].data.meas.get_counts()
                    pred = compute_similarity_from_counts(counts, n_qubits)
                    preds.append(pred)

                preds = np.array(preds)
                correlation, p_value = stats.pearsonr(preds, targets[:len(preds)])

                print(f"Pred range: [{preds.min():.3f}, {preds.max():.3f}]")
                print(f"Correlation: {correlation:.4f} (p={p_value:.3e})")

                results.append({
                    'n_qubits': n_qubits,
                    'correlation': correlation,
                    'p_value': p_value
                })
            else:
                print(f"Job not complete: {status}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in results:
            status = "✅" if r['correlation'] >= 0.80 else "❌"
            print(f"{r['n_qubits']} qubits: r={r['correlation']:.4f} {status}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        debug_job("d4ifnhd74pkc7384vvdg")
        debug_job("d4ifnj2v0j9c73e29rn0")
    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_recent_jobs()
    elif len(sys.argv) > 1 and sys.argv[1] == "--scaling":
        analyze_qubit_scaling_jobs()
    else:
        job_ids_str = "d4ie2kd74pkc7384ube0,d4ie2is3tdfc73dmcpng,d4ie2hh0i6jc73dcn510"

        if len(sys.argv) > 1:
            job_ids_str = sys.argv[1]

        job_ids = [job_id.strip() for job_id in job_ids_str.split(',')]

        if not all(job_ids):
            print("Usage: python retrieve_jobs.py <job_id_1>,<job_id_2>,...")
            print("       python retrieve_jobs.py --debug  (debug dynamic circuits)")
            print("       python retrieve_jobs.py --list   (list recent jobs)")
            print("       python retrieve_jobs.py --scaling (analyze qubit scaling jobs)")
        else:
            retrieve_and_analyze_jobs(job_ids)
