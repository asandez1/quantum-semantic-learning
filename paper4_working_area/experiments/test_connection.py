#!/usr/bin/env python3
"""Test IBM Cloud Quantum connection with explicit credentials"""

from qiskit_ibm_runtime import QiskitRuntimeService

API_KEY = "Rd9UxeZc4lBW_ChSFJ7Mo5Jx6LjDw8nc_erA1TspO9rL"
CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/738d7ccf41ae4801b99ee1b2464c437e:04679de9-a8e2-4158-b865-8d11786dc449::"

print("=" * 70)
print("IBM CLOUD QUANTUM CONNECTION TEST")
print("=" * 70)

print("\nAttempt 1: Using saved account...")
try:
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"✓ Success! Found {len(backends)} backends")
    for b in backends:
        print(f"  - {b.name} ({b.num_qubits} qubits)")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nAttempt 2: Using explicit credentials (channel=ibm_cloud)...")
try:
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=API_KEY,
        instance=CRN
    )
    backends = service.backends()
    print(f"✓ Success! Found {len(backends)} backends")

    if backends:
        print("\nAvailable backends:")
        for backend in backends:
            status = backend.status()
            print(f"\n  {backend.name}")
            print(f"    Qubits: {backend.num_qubits}")
            print(f"    Status: {status.status_msg}")
            print(f"    Pending jobs: {status.pending_jobs}")
    else:
        print("\n⚠ No backends found - instance might not be provisioned")

except Exception as e:
    print(f"✗ Failed: {e}")
    print("\nPossible issues:")
    print("  1. IBM Cloud Quantum instance not fully provisioned")
    print("  2. API key doesn't have access to this instance")
    print("  3. Need to wait a few minutes for provisioning")
    print("\nPlease check:")
    print("  - https://cloud.ibm.com/quantum/instances")
    print("  - Verify instance status is 'Active'")

print("\nAttempt 3: Using explicit credentials (no channel)...")
try:
    service = QiskitRuntimeService(
        token=API_KEY,
        instance=CRN
    )
    backends = service.backends()
    print(f"✓ Success! Found {len(backends)} backends")
    for b in backends:
        print(f"  - {b.name} ({b.num_qubits} qubits)")
except Exception as e:
    print(f"✗ Failed: {e}")
