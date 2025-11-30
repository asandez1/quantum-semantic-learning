#!/usr/bin/env python3
"""Verify and re-save IBM Quantum token with proper format"""

from qiskit_ibm_runtime import QiskitRuntimeService

# Your new token
TOKEN = "04-hy-JWDk85ToIJ5tVgi4gw4jy2M1FOyckBcFljN5r"

print("=" * 70)
print("IBM QUANTUM TOKEN VERIFICATION")
print("=" * 70)

# Try different approaches to save the account
print("\nAttempt 1: Save with ibm_quantum channel (legacy)...")
try:
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token=TOKEN,
        overwrite=True
    )
    print("✓ Saved with ibm_quantum channel")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nAttempt 2: Save without explicit channel (auto-detect)...")
try:
    QiskitRuntimeService.save_account(
        token=TOKEN,
        overwrite=True
    )
    print("✓ Saved without explicit channel")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nAttempt 3: Test connection...")
try:
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"✓ Connection successful! Found {len(backends)} backends")

    # List them
    if backends:
        print("\nAvailable backends:")
        for b in backends[:5]:  # Show first 5
            print(f"  - {b.name} ({b.num_qubits} qubits)")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nThis might indicate:")
    print("  1. Token is invalid or expired")
    print("  2. Account needs to be activated on IBM Quantum website")
    print("  3. Token is for IBM Cloud (not IBM Quantum)")
    print("\nPlease verify at: https://quantum.ibm.com/account")
