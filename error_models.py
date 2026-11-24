import itertools
import numpy as np
from scipy import sparse

OP_NAMES = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]


def generate_all_nb201_ops() -> np.ndarray:
    """Return the exact (15625, 6) integer encoding for NAS-Bench-201."""
    return np.array(list(itertools.product(range(5), repeat=6)), dtype=np.int64)


def match_rmse(raw_err: np.ndarray, target_rmse: float) -> np.ndarray:
    """Center raw_err and rescale it so its empirical RMSE equals target_rmse."""
    if target_rmse == 0.0:
        return np.zeros_like(raw_err, dtype=np.float64)
    centered = raw_err - np.mean(raw_err)
    rms = np.sqrt(np.mean(centered ** 2))
    if rms < 1e-12:
        raise ValueError("Raw error variance is near zero; cannot rescale.")
    return centered * (target_rmse / rms)


def build_neighbor_graph(ops: np.ndarray) -> sparse.csr_matrix:
    """Build sparse 1-hop Hamming adjacency matrix (differing by 1 edge op)."""
    n, n_edges = ops.shape
    powers = 5 ** np.arange(n_edges - 1, -1, -1)
    codes = ops @ powers
    code_to_idx = {int(c): i for i, c in enumerate(codes)}

    rows, cols = [], []
    for i in range(n):
        for e in range(n_edges):
            orig = ops[i, e]
            for alt in range(5):
                if alt != orig:
                    j = code_to_idx.get(int(codes[i] + (alt - orig) * powers[e]))
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


class LatencyCorruptor:
    """
    Permanent Role #4 module: generates corrupted latency estimates with
    identical empirical RMSE across independent and structured error models.
    """

    def __init__(self, latency: np.ndarray, ops: np.ndarray):
        self.latency = np.asarray(latency, dtype=np.float64)
        self.ops = np.asarray(ops, dtype=np.int64)
        self._adj: sparse.csr_matrix | None = None

    def corrupt(
        self,
        mode: str,
        target_rmse: float,
        seed: int = 0,
        op_weights: np.ndarray | None = None,
        jitter_ratio: float = 0.2,
        smooth_steps: int = 2,
        smooth_alpha: float = 0.5,
        clip_min: float = 0.0,
    ) -> np.ndarray:
        """
        Modes:
          - 'iid': independent random Gaussian noise (Role #3 baseline)
          - 'op_bias': systematic bias tied to specific operation types (Role #4)
          - 'neighbor_shared': smooth residuals shared across 1-hop neighbors (Role #4)
        """
        if target_rmse == 0.0:
            return self.latency.copy()

        rng = np.random.default_rng(seed)
        n = len(self.latency)

        if mode == "iid":
            raw = rng.standard_normal(n)

        elif mode == "op_bias":
            if op_weights is None:
                op_weights = rng.standard_normal(5)
            op_bias = np.asarray(op_weights, dtype=np.float64)[self.ops].sum(axis=1)
            op_bias = (op_bias - op_bias.mean()) / (op_bias.std() + 1e-12)
            raw = op_bias + jitter_ratio * rng.standard_normal(n)

        elif mode == "neighbor_shared":
            if self._adj is None:
                self._adj = build_neighbor_graph(self.ops)
            raw = rng.standard_normal(n)
            deg = np.maximum(np.asarray(self._adj.sum(axis=1)).ravel(), 1.0)
            for _ in range(smooth_steps):
                raw = (1.0 - smooth_alpha) * raw + smooth_alpha * ((self._adj @ raw) / deg)

        else:
            raise ValueError(f"Unknown corruption mode: {mode}")

        matched = match_rmse(raw, target_rmse)
        corrupted = self.latency + matched
        if clip_min is not None:
            corrupted = np.maximum(corrupted, clip_min)
        return corrupted
