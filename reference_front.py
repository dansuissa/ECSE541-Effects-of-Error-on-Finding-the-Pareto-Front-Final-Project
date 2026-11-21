"""
reference_front.py  --  ECSE 541 final project, role #2
Reference Pareto front + evaluation metrics.   Version 2.

Both objectives are MINIMIZED:
    error   = CIFAR-100 validation error (%)
    latency = latency (ms)

How the rest of the group uses this file:
    true = pareto_front(df)                         # the reference front (ground truth)
    res  = evaluate_selection(df, "lat_noisy")      # score a front picked with corrupted latency
    summarize_runs([res_seed0, res_seed1, ...])     # mean / std / 95% CI over seeds

Changes from v1 (see the review log in the briefing doc):
    1. Ties: networks with exactly equal scores are all kept on the front.
    2. Normalization uses the TRUE FRONT's range (standard practice), so networks
       far off the front cannot change the scale. Effect was small on test data.
    3. Input checks: missing values and non-positive latency estimates are flagged.
    4. New metrics: near-recovery (with a tolerance) and IGD+.
    5. summarize_runs() for results over several seeds.
"""
import warnings
import numpy as np
import pandas as pd

ERR, LAT, ID = "error", "latency", "arch_id"
REF_MARGIN = 0.1          # reference point = front's worst corner + 10% of its range
NEAR_TOL = 0.01           # near-recovery tolerance, in normalized units (1% of the front's range)


# ------------------------------------------------------------------ checks
def check_inputs(df, cols):
    """Stop on missing values; warn on non-positive estimates (these come from
    noise models that let latency go negative, which no real device can do)."""
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"column '{c}' not found; have {list(df.columns)}")
        if df[c].isna().any():
            raise ValueError(f"column '{c}' has {int(df[c].isna().sum())} missing values")
    for c in cols:
        if c not in (ERR, ID) and (df[c] <= 0).any():
            warnings.warn(f"column '{c}' has {int((df[c] <= 0).sum())} values <= 0; "
                          "check the error model (clip or use multiplicative log-normal noise)")


# ------------------------------------------------------------- Pareto front
def pareto_mask(err, lat):
    """True where a point is non-dominated (both objectives minimized).

    Point A dominates B if A is no worse on both and strictly better on one.
    Method: sort by error (ties by latency), then sweep keeping each point whose
    latency beats everything before it. O(n log n).
    Exact duplicates of a front point are kept too (fix 1).
    """
    err, lat = np.asarray(err, float), np.asarray(lat, float)
    order = np.lexsort((lat, err))
    mask = np.zeros(len(err), dtype=bool)
    best_lat, last = np.inf, None
    for i in order:
        if lat[i] < best_lat:
            mask[i] = True
            best_lat, last = lat[i], i
        elif last is not None and lat[i] == lat[last] and err[i] == err[last]:
            mask[i] = True                      # exact twin of a front point
    return mask


def pareto_front(df, err_col=ERR, lat_col=LAT):
    """Non-dominated rows of df, sorted by error."""
    return df[pareto_mask(df[err_col], df[lat_col])].sort_values([err_col, lat_col])


# ---------------------------------------------------------- normalization
def front_bounds(true_front, err_col=ERR, lat_col=LAT):
    """Best and worst value of each objective ON THE TRUE FRONT (fix 2).
    Compute once from the real data and give everyone the same numbers."""
    return {"err": (true_front[err_col].min(), true_front[err_col].max()),
            "lat": (true_front[lat_col].min(), true_front[lat_col].max())}


def normalize(err, lat, b):
    """Map the true front onto [0, 1] x [0, 1]. Points worse than the front land above 1."""
    (e0, e1), (l0, l1) = b["err"], b["lat"]
    return ((np.asarray(err, float) - e0) / (e1 - e0),
            (np.asarray(lat, float) - l0) / (l1 - l0))


# ------------------------------------------------------------------ metrics
def hypervolume_2d(err, lat, ref=(1 + REF_MARGIN, 1 + REF_MARGIN)):
    """Exact 2D hypervolume (area dominated by the set, bounded by ref).
    Inputs normalized. Points outside ref add nothing."""
    err, lat = np.asarray(err, float), np.asarray(lat, float)
    keep = (err < ref[0]) & (lat < ref[1])
    err, lat = err[keep], lat[keep]
    if len(err) == 0:
        return 0.0
    m = pareto_mask(err, lat)
    e, l = err[m], lat[m]
    o = np.lexsort((l, e))
    e, l = e[o], l[o]
    # duplicates add no area; step widths are zero for them
    nxt = np.append(e[1:], ref[0])
    return float(np.sum((nxt - e) * (ref[1] - l)))


def recovery(found_ids, true_ids):
    """Fraction of true-front networks that were selected. Strict: a near miss counts as 0."""
    t = set(true_ids)
    return len(set(found_ids) & t) / len(t)


def precision(found_ids, true_ids):
    """Fraction of selected networks that really are on the true front."""
    f = set(found_ids)
    return len(f & set(true_ids)) / len(f) if f else 0.0


def near_recovery(found_e, found_l, true_e, true_l, tol=NEAR_TOL):
    """Fraction of true-front points that have a selected point within `tol`
    (normalized, max over the two objectives). Softer than recovery: it credits
    picks that are practically as good. Inputs normalized."""
    F = np.column_stack([found_e, found_l])
    T = np.column_stack([true_e, true_l])
    if len(F) == 0:
        return 0.0
    d = np.abs(T[:, None, :] - F[None, :, :]).max(-1)
    return float((d.min(1) <= tol).mean())


def igd(found_e, found_l, true_e, true_l):
    """Inverted generational distance: mean distance from each true-front point
    to the nearest selected point. Lower = better. Inputs normalized."""
    F = np.column_stack([found_e, found_l])
    T = np.column_stack([true_e, true_l])
    d = np.sqrt(((T[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    return float(d.min(1).mean())


def igd_plus(found_e, found_l, true_e, true_l):
    """IGD+ (Ishibuchi et al. 2015): like IGD but only counts the amount by which
    a selected point is WORSE than a true point. Unlike plain IGD it never rewards
    a worse set, so it is the safer distance metric to report."""
    F = np.column_stack([found_e, found_l])
    T = np.column_stack([true_e, true_l])
    diff = np.maximum(F[None, :, :] - T[:, None, :], 0.0)
    return float(np.sqrt((diff ** 2).sum(-1)).min(1).mean())


# --------------------------------------------- the function the group calls
def evaluate_selection(df, est_lat_col, bounds=None, subset_ids=None,
                       err_col=ERR, lat_col=LAT, id_col=ID, tol=NEAR_TOL):
    """Pick a front using ESTIMATED latency, then score it with TRUE values.

    df          : whole benchmark: id, true error, true latency, estimated latency
    est_lat_col : the corrupted latency column (roles #3 and #4)
    bounds      : front_bounds() of the true front; computed here if not given
    subset_ids  : only the networks a search actually evaluated (role #5).
                  The selection is then limited to those, but it is still scored
                  against the FULL true front.
    """
    check_inputs(df, [id_col, err_col, lat_col, est_lat_col])
    true = pareto_front(df, err_col, lat_col)
    b = bounds or front_bounds(true, err_col, lat_col)

    pool = df if subset_ids is None else df[df[id_col].isin(set(subset_ids))]
    picked = pool[pareto_mask(pool[err_col], pool[est_lat_col])]

    te, tl = normalize(true[err_col], true[lat_col], b)
    pe, pl = normalize(picked[err_col], picked[lat_col], b)        # TRUE values
    hv_t, hv_p = hypervolume_2d(te, tl), hypervolume_2d(pe, pl)

    return {
        "n_true_front": len(true),
        "n_selected": len(picked),
        "n_evaluated": len(pool),
        "recovery": recovery(picked[id_col], true[id_col]),
        "precision": precision(picked[id_col], true[id_col]),
        "near_recovery": near_recovery(pe, pl, te, tl, tol),
        "hv_ratio": hv_p / hv_t,
        "igd": igd(pe, pl, te, tl),
        "igd_plus": igd_plus(pe, pl, te, tl),
    }


def summarize_runs(results):
    """Mean, std and 95% confidence interval for each metric over seeds."""
    r = pd.DataFrame(results)
    n = len(r)
    out = pd.DataFrame({"mean": r.mean(), "std": r.std(ddof=1)})
    out["ci95"] = 1.96 * out["std"] / np.sqrt(n) if n > 1 else np.nan
    return out


# ------------------------------------------------------------------ plotting
def plot_fronts(df, est_lat_col=None, err_col=ERR, lat_col=LAT, id_col=ID,
                title=None, path=None, max_err=None):
    """True front (blue), picks made with corrupted latency (red rings),
    true-front networks that were missed (grey x)."""
    import matplotlib.pyplot as plt
    view = df if max_err is None else df[df[err_col] <= max_err]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(view[lat_col], view[err_col], s=3, c="#cfcfcf", label="All networks", rasterized=True)
    t = pareto_front(df, err_col, lat_col)
    ax.step(t[lat_col], t[err_col], where="post", color="#1f4e8c", lw=1.6)
    ax.scatter(t[lat_col], t[err_col], s=22, color="#1f4e8c", label=f"True front ({len(t)})", zorder=3)
    if est_lat_col:
        p = df[pareto_mask(df[err_col], df[est_lat_col])]
        miss = t[~t[id_col].isin(p[id_col])]
        ax.scatter(p[lat_col], p[err_col], s=34, facecolors="none", edgecolors="#c0392b",
                   lw=1.2, label=f"Picked with corrupted latency ({len(p)})", zorder=4)
        ax.scatter(miss[lat_col], miss[err_col], s=30, marker="x", color="#555555",
                   lw=1.2, label=f"Missed front networks ({len(miss)})", zorder=5)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CIFAR-100 validation error (%)")
    ax.set_title(title or "Reference Pareto front")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# ------------------------------------------------- loading the real benchmark
def load_nb201(path, dataset="cifar100", hp="200"):
    """Build the benchmark table from the NATS-Bench topology file (= NAS-Bench-201).

    NOT TESTED HERE: the data file can't be downloaded in this workspace.
    Needs `pip install nats_bench` and the 'NATS-tss-v1_0-3ffb9-simple' folder.
    Check the field names on one network first (print(info), print(cost)).
    If role #1 already produces this table, use theirs instead.
    """
    from nats_bench import create
    api = create(path, "tss", fast_mode=True, verbose=False)
    rows = []
    for i in range(len(api)):
        info = api.get_more_info(i, dataset, hp=hp, is_random=False)  # mean over trained seeds
        cost = api.get_cost_info(i, dataset, hp=hp)
        rows.append({ID: i, "arch": api.arch(i),
                     ERR: 100.0 - info["valid-accuracy"],
                     LAT: cost["latency"] * 1000.0})                      # s -> ms
    return pd.DataFrame(rows)
