#!/usr/bin/env python3
"""
Update IBM Quantum API token for paper4 experiments
"""

from qiskit_ibm_runtime import QiskitRuntimeService

# New API token
NEW_TOKEN = "04-hy-JWDk85ToIJ5tVgi4gw4jy2M1FOyckBcFljN5r"

print("Updating IBM Quantum API credentials...")
print(f"New token: {NEW_TOKEN[:10]}...{NEW_TOKEN[-10:]}")

try:
    # Save the new account credentials (overwrite existing)
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=NEW_TOKEN,
        overwrite=True
    )
    print("\n✓ Successfully saved IBM Quantum API token!")
    print("\nThe token has been saved to your Qiskit configuration.")
    print("All scripts in paper4/ will now use this new token.")
    print("\nYou can verify it works by running:")
    print("  cd paper4")
    print("  python experiments/run_qmanifold_probe.py --phase 0")

except Exception as e:
    print(f"\n✗ Error saving credentials: {e}")
    print("\nPlease check:")
    print("  1. Token is correct")
    print("  2. You have write permissions")
    print("  3. Qiskit is properly installed")
