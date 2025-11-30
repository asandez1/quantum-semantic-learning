# animal_specialist_now.py
# Train a TRUE animal-only specialist on ibm_fez — 2.1 minutes left → GO!

import os, sys, numpy as np
from datetime import datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER4_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER4_DIR)

from utils.data_preparation import QManifoldDataPreparation
from utils.quantum_circuit import QManifoldCircuit
from utils.batch_optimizer import BatchSPSAOptimizer, SPSAConfig
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

data_prep = QManifoldDataPreparation(target_dim=20)
all_pairs = data_prep.get_default_concept_pairs()

# === BUILD PURE ANIMAL HIERARCHY TRAINING SET ===
ANIMALS = {"animal","mammal","dog","poodle","cat","siamese","bird","sparrow","reptile","snake","fish","salmon"}
animal_pairs = [(c1,c2) for c1,c2 in all_pairs if c1 in ANIMALS and c2 in ANIMALS]
print(f"Found {len(animal_pairs)} pure animal pairs for training")

if len(animal_pairs) == 0:
    print("ERROR: No animal pairs found. Check ANIMALS set matches dataset.")
    sys.exit(1)

# CRITICAL FIX: Fit PCA on ALL concepts (need 20+ for 20D space)
# Then extract only animal pairs for training
print(f"\n[Fix] Fitting PCA on ALL {len(all_pairs)} pairs ({data_prep.generate_all_concepts(all_pairs).__len__()} concepts)")
print(f"[Fix] Training only on {len(animal_pairs)} animal pairs")

# Prepare data with ALL pairs for PCA fitting
data = data_prep.prepare_training_batch(all_pairs, batch_size=len(all_pairs))

# Filter to only animal pair indices
animal_indices = []
for i, (c1, c2) in enumerate(all_pairs):
    if c1 in ANIMALS and c2 in ANIMALS:
        animal_indices.append(i)

# Extract only animal training data
data['pair_indices'] = [data['pair_indices'][i] for i in animal_indices]
data['target_similarities'] = [data['target_similarities'][i] for i in animal_indices]

# Rebuild batches with only animal pairs
data['batches'] = [{
    'pair_indices': data['pair_indices'][i:i+len(animal_pairs)],
    'target_similarities': data['target_similarities'][i:i+len(animal_pairs)]
} for i in range(0, len(animal_pairs), len(animal_pairs))]

print(f"Training data: {len(animal_pairs)} animal pairs")
print(f"Target similarity range: [{min(data['target_similarities']):.3f}, {max(data['target_similarities']):.3f}]")

# Circuit
circuit = QManifoldCircuit(n_qubits=20, ansatz_reps=2, entanglement='circular')
service = QiskitRuntimeService(channel="ibm_cloud", instance="crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::")
backend = service.backend("ibm_fez")
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_circuit = pm.run(circuit.qc_sampler)

# Optimizer (tight for time budget)
config = SPSAConfig(max_iterations=4, batch_size=len(animal_pairs), learning_rate=0.15, perturbation_size=0.1, shots=2048)
print(f"Optimizer config: {config.max_iterations} iterations, batch size {config.batch_size}")
optimizer = BatchSPSAOptimizer(config, circuit, verbose=True)

theta_init = np.random.uniform(0, 0.1, 60)
sampler = SamplerV2(mode=backend)

print(f"\nSTARTING HARDWARE TRAINING — {datetime.now()}")
theta_opt, history = optimizer.optimize_with_sampler(theta_init, data, sampler, isa_circuit)

best_theta = history[np.argmin([h['loss'] for h in history])]['theta']
best_loss = min(h['loss'] for h in history)

print(f"\nANIMAL SPECIALIST TRAINED!")
print(f"Best loss: {best_loss:.6f}")
print(f"Final loss: {history[-1]['loss']:.6f}")

# Save
result = {
    "timestamp": datetime.now().isoformat(),
    "job_ids": [h.get('job_id') for h in history if 'job_id' in h],
    "best_loss": float(best_loss),
    "final_loss": float(history[-1]['loss']),
    "theta_opt": best_theta.tolist(),
    "training_pairs": len(animal_pairs),
    "note": "PURE ANIMAL SPECIALIST — NO COLLAPSE EXPECTED"
}
import json
with open("../results/animal_specialist_final_20251123.json", "w") as f:
    json.dump(result, f, indent=2)

print("SAVED: animal_specialist_final_20251123.json")
print("Now run lightning proof with this theta → correlation > 0.90 expected")