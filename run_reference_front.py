"""Part 2: build the reference Pareto front from the team benchmark.

Run from the repository root (same folder as README.md), after Dan's setup steps:
    .venv\\Scripts\\python run_reference_front.py

Saves:
    results/reference_front.csv   the true front, one row per network
    results/reference_front.png   plot of all networks + the front
"""
from pathlib import Path

import numpy as np
import pandas as pd

import reference_front as rf


def table_from_benchmark(benchmark):
    """Turn the Part 1 arrays into the table Part 2 works with.
    Columns: arch_id, error (%), latency (ms per batch of 256)."""
    ids = np.asarray(benchmark.eligible_ids)
    obj = benchmark.objectives(ids)          # [:, 0] error %, [:, 1] latency ms
    return pd.DataFrame({"arch_id": ids, "error": obj[:, 0], "latency": obj[:, 1]})


def main():
    from ecse541_benchmark import load_benchmark
    root = Path(__file__).resolve().parent
    df = table_from_benchmark(load_benchmark(root / "data"))
    out = root / "results"
    out.mkdir(exist_ok=True)

    front = rf.pareto_front(df)
    front.to_csv(out / "reference_front.csv", index=False)
    b = rf.front_bounds(front)

    print(f"Networks: {len(df):,}")
    print(f"True front: {len(front)} networks")
    print(f"  most accurate: {front.error.min():.2f}% error at {front.latency.max():.2f} ms")
    print(f"  fastest:       {front.latency.min():.2f} ms at {front.error.max():.2f}% error")
    print(f"  broken networks (>90% error) on the front: {(front.error > 90).sum()}")
    print(f"  exact ties: {df.duplicated(['error', 'latency']).sum()}")
    print("Normalization bounds: error %.2f to %.2f %%, latency %.2f to %.2f ms" % (*b["err"], *b["lat"]))

    rf.plot_fronts(df, max_err=70, path=out / "reference_front.png",
                   title=f"NAS-Bench-201, CIFAR-100: true Pareto front ({len(front)} networks)")
    print(f"Saved {out / 'reference_front.csv'} and {out / 'reference_front.png'}")


if __name__ == "__main__":
    main()
