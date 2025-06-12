import numpy as np
import sys

print("--- Windows Integration Check Script Starting ---")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("-" * 20)

try:
    from numbacs.flows import get_predefined_flow
    from numbacs.integration import flowmap_grid_2D

    print("SUCCESS: Imported numbacs functions successfully.")
except Exception as e:
    print(f"ERROR: Failed during import: {e}")
    sys.exit(1)

print("--- Starting Test Execution ---")
print("1. Getting predefined flow 'double_gyre'...")
funcptr, params, domain = get_predefined_flow("double_gyre")
print("   ... Success.")

print("2. Creating test data points...")
nx, ny = 21, 11
x = np.linspace(domain[0][0], domain[0][1], nx)
y = np.linspace(domain[1][0], domain[1][1], ny)
t0 = 0.0
T = 8.0
print("   ... Success.")

print("3. Calling flowmap_grid_2D function...")
try:
    fm = flowmap_grid_2D(funcptr, t0, T, x, y, params)
    print("   ... SUCCESS: flowmap_grid_2D executed.")
except Exception as e:
    print(f"ERROR: The flowmap_grid_2D function failed with a Python exception: {e}")
    sys.exit(1)

print("4. Loading in expected data and comparing result...")
fm_expected = np.load("./tests/testing_data/fm.npy")
assert np.allclose(fm, fm_expected)
print("   ... SUCCESS: test data and expected data match.")
print("--- Script finished successfully ---")
