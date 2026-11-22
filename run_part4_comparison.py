from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ecse541_benchmark import load_benchmark
import reference_front as rf
from error_models import LatencyCorruptor


def load_team_dataframe(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load Part 1's verified benchmark into the DataFrame schema expected by Part 2."""
    bm = load_benchmark(data_dir)
    ids = np.asarray(bm.eligible_ids)
    obj = bm.objectives(ids)  # [:, 0] validation error (%), [:, 1] GPU batch latency (ms)
    ops = np.array([bm.architectures[int(i)].operations for i in ids], dtype=np.int64)

    df = pd.DataFrame({
        "arch_id": ids,
        "error": obj[:, 0],
        "latency": obj[:, 1],
    })
    for j in range(6):
        df[f"op_{j}"] = ops[:, j]
    return df, ops


def run_experiments(df: pd.DataFrame, ops: np.ndarray, out_dir: Path) -> pd.DataFrame:
    true_front = rf.pareto_front(df)
    bounds = rf.front_bounds(true_front)
    print(f"Loaded {len(df):,} eligible networks | True Pareto front: {len(true_front)} networks")

    true_lat = df["latency"].to_numpy(dtype=np.float64)
    corruptor = LatencyCorruptor(true_lat, ops)

    # Latency ranges from 7.20 ms to 26.14 ms (median 15.00 ms).
    # Sweep matched RMSE levels in milliseconds across 10 standardized seeds.
    rmse_levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    modes = ["iid", "op_bias", "neighbor_shared"]
    seeds = range(10)

    records = []
    for rmse in rmse_levels:
        for mode in modes:
            for seed in seeds:
                col = f"est_lat_{mode}_{rmse}_{seed}"
                # clip_min=0.1 ensures strictly positive latency (required by reference_front.check_inputs)
                df[col] = corruptor.corrupt(
                    mode=mode, target_rmse=rmse, seed=seed, clip_min=0.1
                )
                scores = rf.evaluate_selection(df, col, bounds=bounds)
                tau = float(kendalltau(true_lat, df[col].to_numpy()).statistic)
                records.append({
                    "mode": mode,
                    "rmse_ms": rmse,
                    "seed": seed,
                    "kendall_tau": tau,
                    **scores,
                })
                df.drop(columns=[col], inplace=True)

    results_df = pd.DataFrame(records)
    csv_path = out_dir / "part4_matched_rmse_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to {csv_path}")

    # Generate one visual example front at RMSE = 1.0 ms (seed 0) using Part 2's plot_fronts
    df["lat_op_bias_1ms"] = corruptor.corrupt(mode="op_bias", target_rmse=1.0, seed=0, clip_min=0.1)
    rf.plot_fronts(
        df,
        est_lat_col="lat_op_bias_1ms",
        max_err=70,
        path=out_dir / "part4_front_example.png",
        title="Pareto selection under operation-biased latency error (RMSE = 1.0 ms, seed 0)",
    )
    df.drop(columns=["lat_op_bias_1ms"], inplace=True)
    print(f"Saved example front plot to {out_dir / 'part4_front_example.png'}")
    return results_df


def plot_summary_curves(results_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("hv_ratio", "True Hypervolume Ratio (higher = better)"),
        ("igd_plus", "IGD+ Distance to True Front (lower = better)"),
        ("recovery", "Exact Pareto Recovery (higher = better)"),
    ]
    labels = {
        "iid": "Independent Noise (i.i.d.)",
        "op_bias": "Operation-Biased Error",
        "neighbor_shared": "Neighbor-Shared Error",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, (col, ylabel) in zip(axes, metrics):
        for mode in ["iid", "op_bias", "neighbor_shared"]:
            sub = results_df[results_df["mode"] == mode]
            grouped = sub.groupby("rmse_ms")[col]
            means = grouped.mean()
            ci95 = 1.96 * grouped.std(ddof=1) / np.sqrt(grouped.count())
            ax.plot(means.index, means.values, marker="o", lw=1.8, label=labels[mode])
            ax.fill_between(
                means.index,
                means.values - ci95.fillna(0.0).values,
                means.values + ci95.fillna(0.0).values,
                alpha=0.15,
            )
        ax.set_xlabel("Matched Latency Prediction RMSE (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Effect of Latency Error Structure at Matched RMSE (10 seeds, 95% CI)", fontsize=11)
    fig.tight_layout()
    fig_path = out_dir / "part4_matched_rmse_curves.png"
    fig.savefig(fig_path, dpi=220)
    print(f"Saved comparison curves to {fig_path}")


def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    df, ops = load_team_dataframe(root / "data")
    results_df = run_experiments(df, ops, out_dir)
    plot_summary_curves(results_df, out_dir)


if __name__ == "__main__":
    main()