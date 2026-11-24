from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import kendalltau, t
from run_part4_comparison import load_team_dataframe
from error_models import LatencyCorruptor

import reference_front as rf


RMSE_LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
SEEDS = range(10)
CLIP_MIN = 0.1


def calculate_rmse(actual, predicted):
    """Calculate the empirical latency prediction RMSE."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def sanity_checks(true_lat, predicted_lat, target_rmse,
                  corruptor, seed, ops):
    """Validate the independent-error model."""

    residuals = predicted_lat - true_lat

    actual_rmse = calculate_rmse(true_lat, predicted_lat)
    mean_error = float(np.mean(residuals))
    std_error = float(np.std(residuals))

    # Check reproducibility using the same seed.
    repeated = corruptor.corrupt(
        mode="iid",
        target_rmse=target_rmse,
        seed=seed,
        clip_min=CLIP_MIN
    )

    reproducible = np.array_equal(predicted_lat, repeated)

    # Generate unclipped predictions to detect clipping.
    unclipped = corruptor.corrupt(
        mode="iid",
        target_rmse=target_rmse,
        seed=seed,
        clip_min=None
    )

    n_clipped = int(np.sum(unclipped < CLIP_MIN))

    # Count how often each operation appears in an architecture.
    correlations = []

    for op in range(5):
        op_count = np.sum(ops == op, axis=1)

        if np.std(residuals) > 0 and np.std(op_count) > 0:
            corr = np.corrcoef(residuals, op_count)[0, 1]
            correlations.append(abs(corr))

    max_op_correlation = max(correlations, default=0.0)

    # Check correlation between true latency and prediction error.
    latency_error_corr = 0.0

    if np.std(residuals) > 0 and np.std(true_lat) > 0:
        latency_error_corr = float(
            np.corrcoef(true_lat, residuals)[0, 1]
        )

    assert np.all(np.isfinite(predicted_lat)), \
        "Non-finite latency predictions"

    assert np.all(predicted_lat > 0), \
        "Non-positive latency predictions"

    assert reproducible, \
        "Predictions are not reproducible"

    assert np.isclose(actual_rmse, target_rmse, atol=1e-8), \
        f"RMSE mismatch: {actual_rmse} vs {target_rmse}"

    assert np.isclose(mean_error, 0.0, atol=1e-8), \
        f"Unexpected mean error: {mean_error}"

    if target_rmse == 0:
        assert np.array_equal(true_lat, predicted_lat), \
            "Zero-noise predictions differ from ground truth"

    return {
        "actual_rmse_ms": actual_rmse,
        "mean_error_ms": mean_error,
        "std_error_ms": std_error,
        "n_clipped": n_clipped,
        "reproducible": reproducible,
        "max_op_correlation": max_op_correlation,
        "latency_error_correlation": latency_error_corr,
    }


def run_experiments(df, ops, out_dir):
    """Evaluate independent errors across RMSE levels and seeds."""

    df = df.copy()

    true_lat = df["latency"].to_numpy(dtype=np.float64)

    true_front = rf.pareto_front(df)
    bounds = rf.front_bounds(true_front)

    corruptor = LatencyCorruptor(true_lat, ops)

    records = []

    print(f"Architectures: {len(df)}")
    print(f"True Pareto front: {len(true_front)}")
    print(f"Latency range: {true_lat.min():.2f} - "
          f"{true_lat.max():.2f} ms")

    for rmse in RMSE_LEVELS:
        for seed in SEEDS:

            predicted_lat = corruptor.corrupt(
                mode="iid",
                target_rmse=rmse,
                seed=seed,
                clip_min=CLIP_MIN
            )

            checks = sanity_checks(
                true_lat,
                predicted_lat,
                rmse,
                corruptor,
                seed,
                ops
            )

            df["predicted_latency"] = predicted_lat

            scores = rf.evaluate_selection(
                df,
                est_lat_col="predicted_latency",
                bounds=bounds
            )

            # Measure how much latency rankings have changed.
            tau = float(
                kendalltau(true_lat, predicted_lat).statistic
            )

            if rmse == 0:
                for metric in [
                    "recovery",
                    "precision",
                    "near_recovery",
                    "hv_ratio"
                ]:
                    assert np.isclose(scores[metric], 1.0), \
                        f"Zero-noise check failed: {metric}"

                for metric in ["igd", "igd_plus"]:
                    assert np.isclose(scores[metric], 0.0), \
                        f"Zero-noise check failed: {metric}"

                assert scores["n_selected"] == len(true_front)
                assert np.isclose(tau, 1.0)

            records.append({
                "mode": "iid",
                "rmse_ms": rmse,
                "seed": seed,
                "kendall_tau": tau,
                **checks,
                **scores
            })

            print(
                f"RMSE={rmse:.2f} ms | "
                f"Seed={seed} | "
                f"Recovery={scores['recovery']:.3f} | "
                f"HV={scores['hv_ratio']:.3f} | "
                f"Tau={tau:.3f}"
            )

    results = pd.DataFrame(records)

    results.to_csv(
        out_dir / "part3_iid_results.csv",
        index=False
    )

    return results


def summarize_results(results, out_dir):
    """Calculate mean, standard deviation and 95% CI."""

    metrics = [
        "actual_rmse_ms",
        "mean_error_ms",
        "std_error_ms",
        "n_clipped",
        "max_op_correlation",
        "latency_error_correlation",
        "kendall_tau",
        "recovery",
        "precision",
        "near_recovery",
        "hv_ratio",
        "igd",
        "igd_plus",
        "n_selected"
    ]

    records = []

    for rmse, group in results.groupby("rmse_ms"):

        for metric in metrics:

            values = group[metric]
            n = len(values)

            mean = values.mean()
            std = values.std(ddof=1)

            ci95 = (
                t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
                if n > 1 else np.nan
            )

            records.append({
                "rmse_ms": rmse,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci95": ci95,
                "n_seeds": n
            })

    summary = pd.DataFrame(records)

    summary.to_csv(
        out_dir / "part3_iid_summary.csv",
        index=False
    )

    return summary


def plot_summary(summary, out_dir):
    """Plot Pareto-front degradation as noise increases."""

    metrics = [
        ("recovery", "Exact Pareto Recovery"),
        ("precision", "Pareto Precision"),
        ("near_recovery", "Near Recovery"),
        ("hv_ratio", "Hypervolume Ratio"),
        ("igd_plus", "IGD+"),
        ("kendall_tau", "Kendall Rank Correlation")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, (metric, label) in zip(axes.flat, metrics):

        sub = summary[
            summary["metric"] == metric
        ].sort_values("rmse_ms")

        x = sub["rmse_ms"].to_numpy()
        mean = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        ax.plot(x, mean, marker="o", linewidth=1.8)

        ax.fill_between(
            x,
            mean - ci,
            mean + ci,
            alpha=0.2
        )

        ax.set_xlabel("Latency Prediction RMSE (ms)")
        ax.set_ylabel(label)
        ax.set_xticks(RMSE_LEVELS)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Effect of Independent Latency Prediction Errors\n"
        "10 seeds per RMSE level, 95% confidence intervals"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir / "part3_iid_curves.png",
        dpi=220
    )

    plt.close(fig)


def plot_error_distribution(df, ops, out_dir):
    """Visualize the residual distribution at 1 ms RMSE."""

    true_lat = df["latency"].to_numpy(dtype=np.float64)

    corruptor = LatencyCorruptor(true_lat, ops)

    predicted = corruptor.corrupt(
        mode="iid",
        target_rmse=1.0,
        seed=0,
        clip_min=CLIP_MIN
    )

    residuals = predicted - true_lat

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(
        residuals,
        bins=60,
        density=True,
        alpha=0.7
    )

    # Theoretical standard Gaussian density.
    x = np.linspace(residuals.min(), residuals.max(), 300)
    gaussian = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    axes[0].plot(x, gaussian, label="N(0, 1)")
    axes[0].set_xlabel("Latency Prediction Error (ms)")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title("Independent Error Distribution")
    axes[0].legend()

    axes[1].scatter(
        true_lat,
        residuals,
        s=3,
        alpha=0.2
    )

    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("True Latency (ms)")
    axes[1].set_ylabel("Prediction Error (ms)")
    axes[1].set_title("Residuals vs. True Latency")

    fig.tight_layout()

    fig.savefig(
        out_dir / "part3_error_distribution.png",
        dpi=220
    )

    plt.close(fig)


def plot_front_example(df, ops, out_dir):
    """Show how independent errors affect Pareto selection."""

    df = df.copy()

    corruptor = LatencyCorruptor(
        df["latency"].to_numpy(dtype=np.float64),
        ops
    )

    df["predicted_latency"] = corruptor.corrupt(
        mode="iid",
        target_rmse=1.0,
        seed=0,
        clip_min=CLIP_MIN
    )

    fig = rf.plot_fronts(
        df,
        est_lat_col="predicted_latency",
        max_err=70,
        path=out_dir / "part3_front_example.png",
        title="Pareto Selection Under Independent Latency Error "
              "(RMSE = 1 ms, seed 0)"
    )

    plt.close(fig)


def main():

    root = Path(__file__).resolve().parent

    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    df, ops = load_team_dataframe(root / "data")

    results = run_experiments(df, ops, out_dir)

    summary = summarize_results(results, out_dir)

    plot_summary(summary, out_dir)
    plot_error_distribution(df, ops, out_dir)
    plot_front_example(df, ops, out_dir)

    print("\nAll experiments and sanity checks completed.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()