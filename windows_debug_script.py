import numpy as np
import sys

print("--- Windows Debug Script Starting ---")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("-" * 20)

try:
    from numbacs.flows import get_predefined_flow
    from numbacs.integration import flowmap

    print("SUCCESS: Imported numbacs functions successfully.")
except Exception as e:
    print(f"ERROR: Failed during import: {e}")
    sys.exit(1)

print("--- Starting Test Execution ---")
print("1. Getting predefined flow 'double_gyre'...")
funcptr, params, domain = get_predefined_flow("double_gyre")
print("   ... Success.")

print("2. Creating test data points...")
pts = np.array([[0.5, 0.5]], np.float64)
print("   ... Success.")

print("3. Calling flowmap function...")
try:
    fm = flowmap(funcptr, t0=0.0, T=1.0, pts=pts, params=params)
    print("   ... SUCCESS: flowmap executed without a crash!")
    print("Resulting flowmap:", fm)
except Exception as e:
    print(f"ERROR: The flowmap function failed with a Python exception: {e}")
    sys.exit(1)

print("--- Script finished ---")
