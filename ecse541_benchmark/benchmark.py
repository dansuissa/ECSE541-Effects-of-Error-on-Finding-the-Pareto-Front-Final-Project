"""Validated trial records, deterministic aggregation, and a portable team contract."""

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .architecture import Architecture, EDGES, FEATURE_NAMES, OPERATIONS, SPACE_SIZE

SCHEMA_VERSION = 2
OBJECTIVE_NAMES = ("validation_error_pct", "latency_ms")
PROTOCOL = {
    "dataset": "cifar100",
    "split": "x-valid",
    "training_epochs": 200,
    "epoch_index": 199,
    "network_config": {"channel": 16, "num_cells": 5, "class_num": 100},
    "aggregation": "arithmetic mean over all available training seeds",
    "latency_source": "NAS-Bench-201 recorded GPU latency",
    "latency_unit": "ms",
    "latency_batch_size": 256,
    "latency_policy": "mean of positive per-trial means; missing trials retained",
    "objective_names": list(OBJECTIVE_NAMES),
    "objective_directions": ["min", "min"],
    "accuracy_unit": "percent",
    "error_unit": "percentage points",
    "auxiliary_cost_policy": "retain per-trial values; flag cross-seed differences; aggregate by arithmetic mean",
}
TRIAL_FIELDS = ("arch_id", "seed", "validation_accuracy_pct", "latency_samples_s", "params_million", "flops_million")
ROW_FIELDS = ("arch_id", "arch_str", "encoding_id") + tuple(f"op_{i}" for i in range(6)) + (
    "validation_accuracy_pct", "validation_error_pct", "latency_ms", "validation_std_pct",
    "latency_std_ms", "n_runs", "n_latency_runs", "params_million", "flops_million", "eligible", "cost_metadata_consistent",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    return int(value)


@dataclass(frozen=True)
class Trial:
    arch_id: int
    seed: int
    validation_accuracy_pct: float
    latency_samples_s: tuple[float, ...]
    params_million: float
    flops_million: float

    def __post_init__(self):
        object.__setattr__(self, "arch_id", _integer(self.arch_id, "arch_id"))
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        for name in ("validation_accuracy_pct", "params_million", "flops_million"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0 or (name == "validation_accuracy_pct" and value > 100):
                raise ValueError(f"Invalid {name} for architecture {self.arch_id}, seed {self.seed}: {value}")
            object.__setattr__(self, name, value)
        samples = tuple(float(x) for x in self.latency_samples_s)
        if any(not np.isfinite(x) for x in samples):
            raise ValueError(f"Nonfinite latency sample for architecture {self.arch_id}, seed {self.seed}.")
        object.__setattr__(self, "latency_samples_s", samples)

    @property
    def latency_s(self) -> float:
        # The official API first averages repetitions, then excludes nonpositive trial means.
        return float(np.mean(self.latency_samples_s)) if self.latency_samples_s else float("nan")


@dataclass(frozen=True)
class Benchmark:
    architectures: tuple[Architecture, ...]
    trials: tuple[Trial, ...]
    provenance_json: str
    complete: bool = True

    def __post_init__(self):
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "trials", tuple(sorted(self.trials, key=lambda x: (x.arch_id, x.seed))))
        if not self.architectures or any(not isinstance(x, Architecture) for x in self.architectures):
            raise ValueError("Expected a nonempty sequence of validated architectures in official ID order.")
        if self.complete and len(self.architectures) != SPACE_SIZE:
            raise ValueError(f"Expected all {SPACE_SIZE} architectures, got {len(self.architectures)}.")
        if len({a.encoding_id for a in self.architectures}) != len(self.architectures):
            raise ValueError("Duplicate architecture strings/encodings.")
        seen = set()
        for trial in self.trials:
            key = (trial.arch_id, trial.seed)
            if key in seen:
                raise ValueError(f"Duplicate trial (arch_id, seed)={key}.")
            if trial.arch_id >= len(self.architectures):
                raise ValueError(f"Trial references an unknown architecture: {trial.arch_id}")
            seen.add(key)
        if {x.arch_id for x in self.trials} != set(range(len(self.architectures))):
            raise ValueError("Every architecture must have at least one validation trial.")
        provenance = json.loads(self.provenance_json)
        if not isinstance(provenance, dict) or provenance.get("kind") not in {"official", "synthetic_test"}:
            raise ValueError("Provenance must identify 'official' or 'synthetic_test' data.")

    def __len__(self) -> int:
        return len(self.architectures)

    @property
    def provenance(self):
        return MappingProxyType(json.loads(self.provenance_json))

    @cached_property
    def _grouped(self):
        grouped = defaultdict(list)
        for trial in self.trials:
            grouped[trial.arch_id].append(trial)
        return grouped

    def records(self) -> list[dict]:
        rows = []
        for arch_id, architecture in enumerate(self.architectures):
            trials = self._grouped[arch_id]
            accuracies = [t.validation_accuracy_pct for t in trials]
            latencies = [t.latency_s * 1000 for t in trials if np.isfinite(t.latency_s) and t.latency_s > 0]
            mean_accuracy = float(np.mean(accuracies))
            row = dict(arch_id=arch_id, arch_str=architecture.to_string(), encoding_id=architecture.encoding_id)
            row.update({f"op_{i}": op for i, op in enumerate(architecture.operations)})
            row.update(
                validation_accuracy_pct=mean_accuracy,
                validation_error_pct=100.0 - mean_accuracy,
                latency_ms=float(np.mean(latencies)) if latencies else float("nan"),
                validation_std_pct=float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else float("nan"),
                latency_std_ms=float(np.std(latencies, ddof=1)) if len(latencies) > 1 else float("nan"),
                n_runs=len(trials), n_latency_runs=len(latencies),
                params_million=float(np.mean([t.params_million for t in trials])),
                flops_million=float(np.mean([t.flops_million for t in trials])),
                eligible=int(bool(latencies)),
                cost_metadata_consistent=int(all(
                    np.allclose([getattr(t, field) for t in trials], getattr(trials[0], field), rtol=1e-8, atol=1e-10)
                    for field in ("params_million", "flops_million")
                )),
            )
            rows.append(row)
        return rows

    @cached_property
    def _objective_array(self):
        return np.array([[r[k] for k in OBJECTIVE_NAMES] for r in self.records()], dtype=float)

    @property
    def eligible_ids(self) -> np.ndarray:
        """Official IDs with both usable objectives; no renumbering or connectivity filter."""
        return np.flatnonzero(np.isfinite(self._objective_array).all(axis=1))

    def _ids(self, ids) -> np.ndarray:
        if ids is None:
            return np.arange(len(self))
        values = np.asarray(ids)
        if values.ndim != 1 or (values.size and (values.dtype.kind not in "iu" or values.dtype.kind == "b")):
            raise ValueError("IDs must be a one-dimensional sequence of integers.")
        if values.size and ((values < 0).any() or (values >= len(self)).any()):
            raise ValueError("Architecture ID outside the benchmark.")
        return values.astype(int)

    def objectives(self, ids=None) -> np.ndarray:
        """Return a copy of [validation error %, latency ms], both minimized."""
        selected = self._ids(ids)
        values = self._objective_array[selected].copy()
        if not np.isfinite(values).all():
            raise ValueError("Missing latency: pass benchmark.eligible_ids to use the audited common universe.")
        return values

    def features(self, ids=None) -> np.ndarray:
        """Return 39 target-free features aligned to requested official IDs."""
        selected = self._ids(ids)
        return self._features[selected].copy()

    @cached_property
    def _features(self):
        return np.stack([a.features() for a in self.architectures])

    def id_for(self, architecture: Architecture | str) -> int:
        cell = Architecture.from_string(architecture) if isinstance(architecture, str) else architecture
        try:
            return self._encoding_to_id[cell.encoding_id]
        except KeyError as exc:
            raise ValueError("Architecture is absent from this benchmark.") from exc

    @cached_property
    def _encoding_to_id(self):
        return {a.encoding_id: i for i, a in enumerate(self.architectures)}

    def save(self, directory: str | Path) -> Path:
        """Create a new bundle; existing outputs are never silently overwritten."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Output directory must be empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        trials = [dict(
            arch_id=t.arch_id, seed=t.seed, validation_accuracy_pct=t.validation_accuracy_pct,
            latency_samples_s=json.dumps(t.latency_samples_s, allow_nan=False),
            params_million=t.params_million, flops_million=t.flops_million,
        ) for t in self.trials]
        _write_csv(directory / "trials.csv", TRIAL_FIELDS, trials)
        _write_csv(directory / "benchmark.csv", ROW_FIELDS, self.records())
        manifest = {
            "schema_version": SCHEMA_VERSION, "complete": self.complete,
            "n_architectures": len(self), "n_trials": len(self.trials),
            "protocol": PROTOCOL, "provenance": dict(self.provenance),
            "operations": OPERATIONS, "edges": EDGES, "feature_names": FEATURE_NAMES,
            "files": {name: sha256_file(directory / name) for name in ("benchmark.csv", "trials.csv")},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return directory


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, float) and np.isnan(v) else v for k, v in row.items()})


def build_benchmark(architectures, trials, *, provenance: dict, complete=True) -> Benchmark:
    """Architectures must be in official meta_archs order; this order defines arch_id."""
    cells = tuple(Architecture.from_string(a) if isinstance(a, str) else a for a in architectures)
    return Benchmark(cells, tuple(trials), json.dumps(provenance, allow_nan=False), complete)


def load_benchmark(directory: str | Path, *, allow_synthetic=False, allow_partial=False) -> Benchmark:
    """Verify file hashes and recompute aggregates before exposing any objectives."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["protocol"] != PROTOCOL:
        raise ValueError("Unsupported schema or mismatched experimental protocol.")
    if manifest["operations"] != list(OPERATIONS) or manifest["edges"] != [list(e) for e in EDGES] or manifest["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Architecture/feature schema mismatch.")
    if manifest["provenance"]["kind"] != "official" and not allow_synthetic:
        raise ValueError("Synthetic test data cannot be used as experimental evidence.")
    if not manifest["complete"] and not allow_partial:
        raise ValueError("Partial benchmarks are disabled for experiments.")
    for name in ("benchmark.csv", "trials.csv"):
        if sha256_file(directory / name) != manifest["files"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
    with (directory / "benchmark.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(ROW_FIELDS):
            raise ValueError("Unexpected benchmark.csv columns.")
        rows = list(reader)
    if [int(r["arch_id"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Official architecture IDs must be unique, contiguous, and sorted.")
    with (directory / "trials.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(TRIAL_FIELDS):
            raise ValueError("Unexpected trials.csv columns.")
        trials = [Trial(int(r["arch_id"]), int(r["seed"]), float(r["validation_accuracy_pct"]),
                        tuple(json.loads(r["latency_samples_s"])), float(r["params_million"]),
                        float(r["flops_million"])) for r in reader]
    benchmark = build_benchmark([r["arch_str"] for r in rows], trials, provenance=manifest["provenance"], complete=manifest["complete"])
    if len(benchmark) != manifest["n_architectures"] or len(trials) != manifest["n_trials"]:
        raise ValueError("Manifest counts differ from the tables.")
    computed_rows = benchmark.records()
    for key in ROW_FIELDS:
        if key == "arch_str":
            continue
        stored_values = np.array([float(row[key]) if row[key] else float("nan") for row in rows])
        computed_values = np.array([row[key] for row in computed_rows])
        matches = np.isclose(stored_values, computed_values, rtol=1e-12, atol=1e-12, equal_nan=True)
        if not matches.all():
            arch_id = int(np.flatnonzero(~matches)[0])
            raise ValueError(f"Derived value mismatch at architecture {arch_id}, column {key}.")
    return benchmark
