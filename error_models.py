from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import kendalltau, t
from run_part4_comparison import load_team_dataframe  # TODO: move to a shared data_utils module
from error_models import LatencyCorruptor, OP_NAMES

import reference_front as rf


RMSE_LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
SEEDS = range(10)
CLIP_MIN = 0.1
N_OPS = len(OP_NAMES)

MAX_IID_FEATURE_CORR = 0.05

EXAMPLE_RMSE = 1.0
EXAMPLE_SEED = 0


def calculate_rmse(actual, predicted):
    """Calculate the empirical latency prediction RMSE."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def corrupt_iid(corruptor, target_rmse, seed, clip_min=CLIP_MIN):
    """Single entry point for iid corruption (keeps call sites consistent)."""
    return corruptor.corrupt(
        mode="iid",
        target_rmse=target_rmse,
        seed=seed,
        clip_min=clip_min
    )


def max_abs_op_correlation(residuals, ops):
    """Largest |corr| between residuals and per-op counts in an architecture."""
    if np.std(residuals) == 0:
        return 0.0

    correlations = []
    for op in range(N_OPS):
        op_count = np.sum(ops == op, axis=1)
        if np.std(op_count) > 0:
            correlations.append(abs(np.corrcoef(residuals, op_count)[0, 1]))

    return float(max(correlations, default=0.0))


def sanity_checks(true_lat, predicted_lat, target_rmse,
                  corruptor, seed, ops):
    """Validate the independent-error model.

    RMSE matching happens before clipping, so exactness is checked on the
    unclipped vector. The clipped vector is checked for consistency with it,
    and its (slightly off) RMSE / mean error are recorded, not asserted.
    """

    # Nominal (unclipped) predictions: this is where the error model is exact.
    unclipped = corrupt_iid(corruptor, target_rmse, seed, clip_min=None)
    nominal_residuals = unclipped - true_lat

    nominal_rmse = calculate_rmse(true_lat, unclipped)
    nominal_mean = float(np.mean(nominal_residuals))

    residuals = predicted_lat - true_lat

    actual_rmse = calculate_rmse(true_lat, predicted_lat)
    mean_error = float(np.mean(residuals))
    std_error = float(np.std(residuals))

    n_clipped = int(np.sum(unclipped < CLIP_MIN))

    repeated = corrupt_iid(corruptor, target_rmse, seed)
    reproducible = np.array_equal(predicted_lat, repeated)


    # because clipping hits low-latency archs and induces op correlation.
    max_op_correlation = max_abs_op_correlation(nominal_residuals, ops)

    # Latency/error correlation on the effective residuals: shows clipping bias.
    latency_error_corr = 0.0
    if np.std(residuals) > 0 and np.std(true_lat) > 0:
        latency_error_corr = float(np.corrcoef(true_lat, residuals)[0, 1])

    assert np.all(np.isfinite(predicted_lat)), \
        "Non-finite latency predictions"

    assert np.all(predicted_lat > 0), \
        "Non-positive latency predictions"

    assert reproducible, \
        "Predictions are not reproducible"

    assert np.isclose(nominal_rmse, target_rmse, atol=1e-8), \
        f"Nominal RMSE mismatch: {nominal_rmse} vs {target_rmse}"

    assert np.isclose(nominal_mean, 0.0, atol=1e-8), \
        f"Unexpected nominal mean error: {nominal_mean}"

    if target_rmse == 0:
        assert np.array_equal(true_lat, predicted_lat), \
            "Zero-noise predictions differ from ground truth"
    else:
        assert np.array_equal(predicted_lat, np.maximum(unclipped, CLIP_MIN)), \
            "Clipped predictions inconsistent with unclipped ones"

    assert max_op_correlation < MAX_IID_FEATURE_CORR, \
        f"iid residuals correlate with op counts: {max_op_correlation:.3f}"

    return {
        "nominal_rmse_ms": nominal_rmse,
        "actual_rmse_ms": actual_rmse,
        "mean_error_ms": mean_error,
        "std_error_ms": std_error,
        "n_clipped": n_clipped,
        "reproducible": reproducible,
        "max_op_correlation": max_op_correlation,
        "latency_error_correlation": latency_error_corr,
    }


def check_zero_noise(scores, tau, n_true_front):
    """Perfect predictions must recover the true front."""
    for metric in ["recovery", "precision", "near_recovery", "hv_ratio"]:
        assert np.isclose(scores[metric], 1.0), \
            f"Zero-noise check failed: {metric}"

    for metric in ["igd", "igd_plus"]:
        assert np.isclose(scores[metric], 0.0), \
            f"Zero-noise check failed: {metric}"

    assert scores["n_selected"] == n_true_front
    assert np.isclose(tau, 1.0)


def run_experiments(df, ops, corruptor, out_dir):
    """Evaluate independent errors across RMSE levels and seeds."""

    df = df.copy()

    true_lat = df["latency"].to_numpy(dtype=np.float64)

    true_front = rf.pareto_front(df)
    bounds = rf.front_bounds(true_front)

    records = []

    print(f"Architectures: {len(df)}")
    print(f"True Pareto front: {len(true_front)}")
    print(f"Latency range: {true_lat.min():.2f} - "
          f"{true_lat.max():.2f} ms")

    for rmse in RMSE_LEVELS:
        for seed in SEEDS:

            predicted_lat = corrupt_iid(corruptor, rmse, seed)

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

            tau = float(kendalltau(true_lat, predicted_lat)[0])

            if rmse == 0:
                check_zero_noise(scores, tau, len(true_front))

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
                f"Clipped={checks['n_clipped']} | "
                f"Eff. RMSE={checks['actual_rmse_ms']:.3f} | "
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
        "nominal_rmse_ms",
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

        ax.set_xlabel("Nominal Latency Prediction RMSE (ms)")
        ax.set_ylabel(label)
        ax.set_xticks(RMSE_LEVELS)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Effect of Independent Latency Prediction Errors\n"
        f"{len(SEEDS)} seeds per RMSE level, 95% confidence intervals"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir / "part3_iid_curves.png",
        dpi=220
    )

    plt.close(fig)


def plot_error_distribution(true_lat, predicted, sigma, out_dir):
    """Visualize the residual distribution for one example corruption."""

    residuals = predicted - true_lat

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(
        residuals,
        bins=60,
        density=True,
        alpha=0.7
    )

    # Theoretical Gaussian density at the nominal RMSE.
    x = np.linspace(residuals.min(), residuals.max(), 300)
    gaussian = (
        np.exp(-x ** 2 / (2 * sigma ** 2))
        / (sigma * np.sqrt(2 * np.pi))
    )

    axes[0].plot(x, gaussian, label=f"N(0, {sigma:g}²)")
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


def plot_front_example(df, predicted, rmse, seed, out_dir):
    """Show how independent errors affect Pareto selection."""

    df = df.copy()
    df["predicted_latency"] = predicted

    fig = rf.plot_fronts(
        df,
        est_lat_col="predicted_latency",
        max_err=70,
        path=out_dir / "part3_front_example.png",
        title="Pareto Selection Under Independent Latency Error "
              f"(RMSE = {rmse:g} ms, seed {seed})"
    )

    plt.close(fig)


def main():

    root = Path(__file__).resolve().parent

    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    df, ops = load_team_dataframe(root / "data")

    true_lat = df["latency"].to_numpy(dtype=np.float64)
    corruptor = LatencyCorruptor(true_lat, ops)

    results = run_experiments(df, ops, corruptor, out_dir)

    summary = summarize_results(results, out_dir)
    plot_summary(summary, out_dir)

    # One example corruption shared by both example plots.
    example = corrupt_iid(corruptor, EXAMPLE_RMSE, EXAMPLE_SEED)
    plot_error_distribution(true_lat, example, EXAMPLE_RMSE, out_dir)
    plot_front_example(df, example, EXAMPLE_RMSE, EXAMPLE_SEED, out_dir)

    print("\nAll experiments and sanity checks completed.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()