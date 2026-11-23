"""verify.py -- One helper to print every comparison in the same format.

report() is the helper from the Iteration 3 handout. It also stores each
comparison so the scripts can write the verification table.
"""
import numpy as np

ROWS = []


def report(name, mine, ref, tol=None, data="", compared_with=""):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    ok = "" if tol is None else ("PASS" if abs_err <= tol or rel_err <= tol else "FAIL")
    print(f"{name:40s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}   {ok}")
    ROWS.append(dict(data=data, quantity=name, compared_with=compared_with,
                     abs_err=abs_err, rel_err=rel_err, ok=ok))
    return abs_err, rel_err
