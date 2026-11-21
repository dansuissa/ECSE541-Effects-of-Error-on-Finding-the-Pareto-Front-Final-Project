"""
nb201_like.py -- a stand-in for NAS-Bench-201 so the code can be tested before
the real data is loaded. It copies the benchmark's STRUCTURE, not its numbers.

Same as the real benchmark:
  * one cell, 4 nodes, 6 edges, 5 operations per edge -> 5**6 = 15,625 networks
  * networks whose output is cut off from the input ("none" everywhere on the
    path) cannot learn: about 1% accuracy on CIFAR-100, i.e. ~99% error
  * 3x3 convs help accuracy most and cost the most latency

Made up (replace with real data):
  * the exact error and latency values
"""
import itertools
import numpy as np
import pandas as pd

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
# edges in NAS-Bench-201 order: (to, from)
EDGES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]

# made-up per-op costs (ms) and "learning value"
LAT_COST = {"none": 0.0, "skip_connect": 0.05, "nor_conv_1x1": 0.9,
            "nor_conv_3x3": 2.1, "avg_pool_3x3": 0.35}
LEARN = {"none": 0.0, "skip_connect": 0.0, "nor_conv_1x1": 0.6,
         "nor_conv_3x3": 1.0, "avg_pool_3x3": 0.15}


def arch_str(ops):
    """Same string format as the real benchmark, e.g. |nor_conv_3x3~0|+|none~0|skip_connect~1|+|..."""
    o = list(ops)
    return f"|{o[0]}~0|+|{o[1]}~0|{o[2]}~1|+|{o[3]}~0|{o[4]}~1|{o[5]}~2|"


def _connected(ops):
    reach = {0}
    for node in (1, 2, 3):
        for (to, frm), op in zip(EDGES, ops):
            if to == node and frm in reach and op != "none":
                reach.add(node)
                break
    return 3 in reach


def make_benchmark(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, ops in enumerate(itertools.product(OPS, repeat=6)):
        if not _connected(ops):
            err = 99.0 + rng.normal(0, 0.1)                     # broken network
        else:
            learn = sum(LEARN[o] for o in ops)
            err = 26.5 + 45 * np.exp(-learn / 1.6) + rng.normal(0, 1.5)
        lat = 6.0 + sum(LAT_COST[o] for o in ops) + rng.normal(0, 0.15)  # base + ops + jitter
        row = {"arch_id": i, "arch": arch_str(ops), "error": float(min(err, 99.5)),
               "latency": float(max(lat, 5.5))}
        row.update({f"e{k}": op for k, op in enumerate(ops)})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_benchmark()
    print(df.shape)
    print(df[["error", "latency"]].describe().round(2))
    print("broken networks:", (df.error > 98).sum())
