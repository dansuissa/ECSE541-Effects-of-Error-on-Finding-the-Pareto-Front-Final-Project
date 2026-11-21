"""Checks for reference_front.py (v2).   Run:  python test_reference_front.py
Every metric is compared against something independent: a slow brute-force
version, hand-worked answers, a Monte Carlo estimate, or the pymoo library."""
import warnings
import numpy as np
import pandas as pd
import reference_front as rf
from nb201_like import make_benchmark

rng = np.random.default_rng(0)
passed = 0


def ok(msg):
    global passed
    passed += 1
    print("ok ", msg)


def brute_force_mask(e, l):
    keep = np.ones(len(e), bool)
    for i in range(len(e)):
        dom = (e <= e[i]) & (l <= l[i]) & ((e < e[i]) | (l < l[i]))
        keep[i] = not dom.any()
    return keep


# 1. Pareto front: fast == brute force, with many ties and exact duplicates
for _ in range(100):
    e = rng.integers(0, 15, 300).astype(float)
    l = rng.integers(0, 15, 300).astype(float)
    assert (rf.pareto_mask(e, l) == brute_force_mask(e, l)).all()
ok("Pareto front == brute force on 100 random sets with ties")

df = pd.DataFrame({"arch_id": [0, 1, 2], "error": [30., 30., 40.], "latency": [10., 10., 5.]})
assert list(rf.pareto_front(df).arch_id) == [0, 1, 2]
ok("exact twins both kept (v1 dropped one)")

# 2. hypervolume: hand cases
assert np.isclose(rf.hypervolume_2d([0], [0], (1, 1)), 1.0)
assert np.isclose(rf.hypervolume_2d([0, .5], [.5, 0], (1, 1)), 0.75)
assert np.isclose(rf.hypervolume_2d([0, .5, .6], [.5, 0, .6], (1, 1)), 0.75)   # dominated point ignored
assert np.isclose(rf.hypervolume_2d([0, 0], [.5, .5], (1, 1)), 0.5)          # duplicate adds nothing
assert rf.hypervolume_2d([2.0], [2.0], (1, 1)) == 0.0                        # outside ref
ok("hypervolume hand cases")

# 3. hypervolume, IGD, IGD+ vs pymoo (independent implementation)
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
for _ in range(30):
    T = rng.random((25, 2)); F = rng.random((15, 2))
    assert np.isclose(rf.hypervolume_2d(F[:, 0], F[:, 1], (1.1, 1.1)), HV(ref_point=np.array([1.1, 1.1]))(F))
    assert np.isclose(rf.igd(F[:, 0], F[:, 1], T[:, 0], T[:, 1]), IGD(T)(F))
    assert np.isclose(rf.igd_plus(F[:, 0], F[:, 1], T[:, 0], T[:, 1]), IGDPlus(T)(F))
ok("hypervolume, IGD and IGD+ match pymoo on 30 random cases")

# 4. on the NAS-Bench-201 stand-in
bench = make_benchmark(seed=0)
bench["lat_exact"] = bench["latency"]
r = rf.evaluate_selection(bench, "lat_exact")
assert r["recovery"] == 1 and r["precision"] == 1 and np.isclose(r["hv_ratio"], 1) and r["igd_plus"] == 0
ok("perfect latency -> perfect scores")

# 5. more noise -> worse scores (averaged over 10 seeds so it isn't luck)
means = []
for s in [0.02, 0.05, 0.10, 0.20]:
    runs = []
    for seed in range(10):
        g = np.random.default_rng(100 + seed)
        bench["est"] = bench["latency"] * np.exp(g.normal(0, s, len(bench)))
        runs.append(rf.evaluate_selection(bench, "est"))
    means.append(rf.summarize_runs(runs)["mean"])
for a, b in zip(means, means[1:]):
    assert b["hv_ratio"] <= a["hv_ratio"] and b["igd_plus"] >= a["igd_plus"]
ok("scores get worse as noise grows (10 seeds per level)")

# 6. a search that only saw part of the space
ids = bench.sample(2000, random_state=0)["arch_id"]
r = rf.evaluate_selection(bench, "lat_exact", subset_ids=ids)
assert r["n_evaluated"] == 2000 and r["hv_ratio"] <= 1 + 1e-12
ok("subset search scored against the full front")

# 7. bad inputs
bad = bench.copy(); bad.loc[0, "est"] = -1.0
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    rf.evaluate_selection(bad, "est")
    assert any("<= 0" in str(x.message) for x in w)
bad.loc[1, "est"] = np.nan
try:
    rf.evaluate_selection(bad, "est"); raise AssertionError("missing value not caught")
except ValueError:
    pass
ok("negative estimates warned, missing values stopped")

print(f"\nall {passed} checks passed")
