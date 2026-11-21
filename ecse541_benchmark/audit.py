"""Search-space checks and measured-data diagnostics; no Pareto evaluation."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .architecture import Architecture, EDGES, FEATURE_NAMES, OPERATIONS, SPACE_SIZE, all_architectures


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def inspect_space() -> dict:
    cells = tuple(all_architectures())
    features = np.stack([cell.features() for cell in cells])
    roundtrips = all(Architecture.from_string(a.to_string()) == a for a in cells)
    ids = [a.encoding_id for a in cells]
    return {
        "evidence": "Exact enumeration of the defined search space; no measured metrics used.",
        "n_architectures": len(cells), "n_unique_encodings": len(set(ids)),
        "string_roundtrips_pass": roundtrips,
        "encoding_roundtrips_pass": ids == list(range(SPACE_SIZE)),
        "feature_shape": list(features.shape), "one_hot_row_sums_pass": bool(np.all(features[:, :30].reshape(-1, 6, 5).sum(axis=2) == 1)),
        "operation_counts_per_edge": {f"{s}->{d}": dict(zip(OPERATIONS, np.bincount([a.operations[i] for a in cells], minlength=5).tolist()))
                                      for i, (s, d) in enumerate(EDGES)},
        "input_output_path_distribution": dict(sorted(Counter(int(x) for x in features[:, -3]).items())),
        "n_disconnected_cells": int((features[:, -3] == 0).sum()),
        "n_connected_cells": int((features[:, -3] > 0).sum()),
        "official_id_alignment": "Verified only when the authors' meta_archs are loaded; encoding IDs are not assumed official IDs.",
    }


def write_space_report(directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = inspect_space()
    (directory / "architecture_checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plt = _plotting()
    from matplotlib.patches import FancyArrowPatch, Circle
    fig = plt.figure(figsize=(12, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(grid[:, 0])
    positions = {0: (0, .5), 1: (1, 1), 2: (2, 1), 3: (3, .5)}
    # Routes and labels show all six slots, including the explicitly absent edge.
    example = Architecture((3, 1, 2, 0, 4, 3))
    colors = ("#a9b2ba", "#287d8e", "#c17b35", "#24496b", "#6f648a")
    curves = (0, -.12, 0, .28, -.12, 0)
    label_positions = ((.42, .92), (.9, .36), (1.5, 1.13), (1.5, -.03), (2.1, .36), (2.65, .92))
    for i, ((src, dst), op) in enumerate(zip(EDGES, example.operations)):
        ax.add_patch(FancyArrowPatch(positions[src], positions[dst], connectionstyle=f"arc3,rad={curves[i]}",
                                    arrowstyle="-|>", mutation_scale=13, linewidth=2, color=colors[op],
                                    linestyle="--" if op == 0 else "-", shrinkA=17, shrinkB=17))
        short = {0: "none", 1: "skip", 2: "conv 1×1", 3: "conv 3×3", 4: "avg pool"}[op]
        ax.text(*label_positions[i], f"e{i}: {short}", ha="center", color=colors[op], fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    for node, position in positions.items():
        ax.add_patch(Circle(position, .12, facecolor="#eef2f5", edgecolor="#24496b", linewidth=1.5, zorder=5))
        ax.text(*position, str(node), ha="center", va="center", zorder=6, fontweight="bold")
    ax.text(0, .26, "input", ha="center", fontsize=9)
    ax.text(3, .26, "output", ha="center", fontsize=9)
    ax.text(1.5, -.32, "genes = [3, 1, 2, 0, 4, 3]\n6 edge slots × 5 categorical choices = 15,625 cells\n39 features: 30 indicators + 5 counts + 4 graph descriptors",
            ha="center", va="top", fontsize=10, linespacing=1.6)
    ax.set(xlim=(-.35, 3.35), ylim=(-.8, 1.4), title="A  |  Cell representation and feature encoding")
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 1])
    counts = summary["input_output_path_distribution"]
    ax.bar(list(counts), list(counts.values()), color="#287d8e", width=.65)
    ax.set(xticks=list(counts), xlabel="Active input-to-output paths", ylabel="Number of cells", title="B  |  Exact connectivity census")
    for k, v in counts.items():
        ax.text(k, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts.values()) * 1.2)
    ax = fig.add_subplot(grid[1, 1])
    ax.barh(["Connected", "Disconnected"], [summary["n_connected_cells"], summary["n_disconnected_cells"]], color=["#24496b", "#a9b2ba"])
    ax.set(xlabel="Number of cells", title="C  |  Both groups remain in the benchmark")
    ax.set_xlim(0, SPACE_SIZE * 1.15)
    for i, n in enumerate([summary["n_connected_cells"], summary["n_disconnected_cells"]]):
        ax.text(n + 100, i, f"{n:,}", va="center", fontsize=9)
    fig.suptitle("NAS-Bench-201 architecture audit", fontsize=16, fontweight="bold", color="#24496b")
    fig.savefig(directory / "architecture_space.png", dpi=220)
    plt.close(fig)
    return summary


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(zip(("min", "q25", "median", "q75", "max"), np.quantile(values, [0, .25, .5, .75, 1]).tolist()))


def audit_benchmark(benchmark) -> dict:
    rows = benchmark.records()
    eligible = benchmark.eligible_ids
    objectives = benchmark.objectives(eligible)
    connected = benchmark.features()[:, FEATURE_NAMES.index("input_output_paths")] > 0
    timing_records = defaultdict(list)
    for trial in benchmark.trials:
        if np.isfinite(trial.latency_s) and trial.latency_s > 0:
            timing_records[trial.arch_id].append(trial.latency_samples_s)
    repeated_timings = [values for values in timing_records.values() if len(values) > 1]
    identical_timings = sum(len(set(values)) == 1 for values in repeated_timings)
    return {
        "data_kind": benchmark.provenance["kind"], "complete": benchmark.complete,
        "n_architectures": len(benchmark), "n_trials": len(benchmark.trials),
        "n_eligible": len(eligible),
        "ineligible_arch_ids": [r["arch_id"] for r in rows if not r["eligible"]],
        "seed_count_distribution": dict(sorted(Counter(r["n_runs"] for r in rows).items())),
        "training_seeds": sorted({t.seed for t in benchmark.trials}),
        "latency_entry_count_distribution": dict(sorted(Counter(len(t.latency_samples_s) for t in benchmark.trials).items())),
        "n_cells_with_repeated_latency_records": len(repeated_timings),
        "n_repeated_cells_with_identical_latency_records": identical_timings,
        "n_repeated_cells_with_distinct_latency_records": len(repeated_timings) - identical_timings,
        "n_single_seed_cells": sum(r["n_runs"] == 1 for r in rows),
        "n_trials_without_positive_latency": sum(not np.isfinite(t.latency_s) or t.latency_s <= 0 for t in benchmark.trials),
        "n_cells_with_partial_latency": sum(0 < r["n_latency_runs"] < r["n_runs"] for r in rows),
        "n_disconnected_cells": int((~connected).sum()),
        "n_eligible_disconnected_cells": int((~connected[eligible]).sum()),
        "n_duplicate_objective_rows": len(objectives) - len(np.unique(objectives, axis=0)),
        "n_cells_with_inconsistent_cost_metadata": sum(not r["cost_metadata_consistent"] for r in rows),
        "inconsistent_cost_metadata_arch_ids": [r["arch_id"] for r in rows if not r["cost_metadata_consistent"]],
        "objective_summary": {name: _stats(objectives[:, i]) for i, name in enumerate(("validation_error_pct", "latency_ms"))},
        "between_seed_validation_std_pct": _stats([r["validation_std_pct"] for r in rows]),
        "between_seed_latency_std_ms": _stats([r["latency_std_ms"] for r in rows]),
        "source": dict(benchmark.provenance),
        "interpretation": [
            "Recorded benchmark values are the fixed experimental reference, not physical noise-free truth.",
            "Sample standard deviations use ddof=1; one observation is undefined, not zero.",
            "Latency summarizes a recorded GPU batch of 256 in milliseconds, not per-image latency.",
            "Use identical eligible architecture IDs in every comparison; do not filter by connectivity or test accuracy.",
            "Duplicate objective vectors retain their distinct architecture IDs.",
            "Parameter/FLOP metadata differences are flagged; original per-trial values are retained and auxiliary means match the authors' API.",
            "Identical timing entries across training seeds are not independent timing repetitions and do not establish physical timing uncertainty.",
        ],
    }


def write_data_report(benchmark, directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = audit_benchmark(benchmark)
    (directory / "data_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plt = _plotting()
    rows = benchmark.records()
    objectives = benchmark.objectives(benchmark.eligible_ids)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    ax = axes[0, 0]
    if len(objectives):
        ax.scatter(objectives[:, 1], objectives[:, 0], s=4, alpha=.25, color="#24496b", rasterized=True)
        ax.set_xscale("log")
    ax.set(xlabel="Recorded GPU batch latency (ms, log scale)", ylabel="CIFAR-100 validation error (percentage points)", title="A  |  Objective distribution")
    counts = summary["seed_count_distribution"]
    axes[0, 1].bar(list(counts), list(counts.values()), color="#287d8e")
    axes[0, 1].set(xticks=list(counts), xlabel="Available training seeds", ylabel="Number of architectures", title="B  |  Repeat coverage")
    ax = axes[1, 0]
    values = [r["validation_std_pct"] for r in rows if np.isfinite(r["validation_std_pct"])]
    if values:
        ax.hist(values, bins=min(40, max(1, len(values))), color="#c17b35")
    else:
        ax.text(.5, .5, "No repeated observations available", ha="center", transform=ax.transAxes)
    ax.set(title="C  |  Validation variation between seeds", xlabel="Sample standard deviation (percentage points)", ylabel="Number of architectures")
    ax = axes[1, 1]
    copies = summary["n_repeated_cells_with_identical_latency_records"]
    different = summary["n_repeated_cells_with_distinct_latency_records"]
    ax.bar([0, 1], [copies, different], color=["#6f648a", "#a9b2ba"], width=.6)
    for i, value in enumerate((copies, different)):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    ax.set(xticks=[0, 1], xticklabels=["Identical entries", "Different entries"],
           title="D  |  Latency entries across training seeds", ylabel="Architectures with repeated records")
    ax.set_ylim(0, max(1, copies, different) * 1.3)
    ax.text(.5, .95, "Identical copies are not independent timing repeats", ha="center", va="top", transform=ax.transAxes, fontsize=9)
    label = "Recorded benchmark data" if benchmark.provenance["kind"] == "official" else "SYNTHETIC TEST DATA — no experimental results"
    fig.suptitle(f"NAS-Bench-201 data audit\n{label}", fontsize=14, fontweight="bold", color="#24496b")
    fig.savefig(directory / "data_overview.png", dpi=220)
    plt.close(fig)
    return summary
