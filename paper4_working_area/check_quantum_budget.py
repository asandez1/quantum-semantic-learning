#!/usr/bin/env python3
"""Check IBM Quantum budget and available backends"""

from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime

print("=" * 70)
print("IBM QUANTUM BUDGET CHECK")
print("=" * 70)

try:
    service = QiskitRuntimeService()

    print(f"\n✓ Connected to IBM Quantum")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get available backends
    print("\n" + "=" * 70)
    print("AVAILABLE BACKENDS")
    print("=" * 70)

    backends = service.backends()

    print(f"\nTotal backends: {len(backends)}")
    print("\nDetails:")

    for backend in backends:
        print(f"\n  {backend.name}")
        print(f"    Qubits: {backend.num_qubits}")
        print(f"    Status: {backend.status().status_msg}")
        print(f"    Pending jobs: {backend.status().pending_jobs}")

    # Recommend backend for experiments
    print("\n" + "=" * 70)
    print("RECOMMENDED FOR Q-MANIFOLD")
    print("=" * 70)

    # Look for ibm_marrakesh, ibm_fez, or other 156-qubit systems
    recommended = None
    for backend in backends:
        if backend.num_qubits >= 127 and backend.status().operational:
            recommended = backend
            break

    if recommended:
        print(f"\n✓ Use: {recommended.name}")
        print(f"  Qubits: {recommended.num_qubits}")
        print(f"  Pending jobs: {recommended.status().pending_jobs}")
        print(f"\nCommand to run Phase 1A:")
        print(f"  python experiments/run_qmanifold_probe.py --phase 1A --backend {recommended.name} --yes")
    else:
        print("\n⚠ No large backends available")
        print("  Using any available backend with 20+ qubits")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Check token is saved correctly")
    print("  2. Verify internet connection")
    print("  3. Confirm IBM Quantum account is active")
