"""Extract the original NAS-Bench-201 v1.1 state without loading network weights.

The serialized schema is checked against the authors' nas_201_api/api_201.py
and api_utils.py. Keeping the raw state avoids a second full API deep copy.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .architecture import Architecture, SPACE_SIZE
from .benchmark import Trial, build_benchmark, sha256_file

SOURCE_NAME = "NAS-Bench-201-v1_1-096897.pth"
SOURCE_ID = "16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_"
SOURCE_URL = f"https://drive.google.com/file/d/{SOURCE_ID}/view"
AUTHORS_URL = "https://github.com/D-X-Y/NAS-Bench-201"


def extract_state(state: dict, *, provenance: dict, complete=True):
    """Read final CIFAR-100 validation results, preserving original IDs and seeds."""
    if not isinstance(state, dict) or not {"meta_archs", "arch2infos", "evaluated_indexes"} <= state.keys():
        raise ValueError("Expected an original NAS-Bench-201 state dictionary.")
    architectures = state["meta_archs"]
    if complete and len(architectures) != SPACE_SIZE:
        raise ValueError(f"Expected {SPACE_SIZE} official meta_archs, got {len(architectures)}.")
    expected = set(range(len(architectures)))
    evaluated = list(state["evaluated_indexes"])
    if set(evaluated) != expected or len(evaluated) != len(expected) or set(state["arch2infos"]) != expected:
        raise ValueError("Evaluated indices and arch2infos must cover meta_archs exactly.")
    trials = []
    for arch_id, arch_str in enumerate(architectures):
        Architecture.from_string(arch_str)
        try:
            info = state["arch2infos"][arch_id]["full"]
            if info["arch_index"] != arch_id or info["arch_str"] != arch_str:
                raise ValueError(f"Official ID/string mismatch at architecture {arch_id}.")
            seeds = info["dataset_seed"]["cifar100"]
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError(f"Missing or repeated CIFAR-100 seeds at architecture {arch_id}.")
            recorded = {seed for dataset, seed in info["all_results"] if dataset == "cifar100"}
            if set(seeds) != recorded:
                raise ValueError(f"Seed list differs from trial records at architecture {arch_id}.")
            for seed in sorted(seeds):
                run = info["all_results"][("cifar100", seed)]
                stored_seed = run["seed"]
                if isinstance(stored_seed, str) and stored_seed.isascii() and stored_seed.isdecimal():
                    stored_seed = int(stored_seed)
                if stored_seed != seed or run["epochs"] != 200 or run["arch_config"]["arch_str"] != arch_str:
                    raise ValueError(f"Trial identity/budget mismatch at architecture {arch_id}, seed {seed}.")
                expected_config = {"channel": 16, "num_cells": 5, "class_num": 100}
                if any(run["arch_config"].get(key) != value for key, value in expected_config.items()):
                    raise ValueError(f"Network configuration mismatch at architecture {arch_id}, seed {seed}.")
                if "x-valid" not in run["eval_names"]:
                    raise ValueError(f"Missing validation split at architecture {arch_id}, seed {seed}.")
                latency = run["latency"]
                trials.append(Trial(
                    arch_id=arch_id, seed=seed,
                    validation_accuracy_pct=run["eval_acc1es"]["x-valid@199"],
                    latency_samples_s=tuple(latency) if latency is not None else (),
                    params_million=run["params"], flops_million=run["flop"],
                ))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Incomplete or incompatible source record at architecture {arch_id}: {exc}") from exc
    return build_benchmark(architectures, trials, provenance=provenance, complete=complete)


def extract_file(path: str | Path, *, trust_source=False, expected_sha256=None, progress=None, verify_api=False):
    """Deserialize only a trusted original file; use hashes for reproducibility."""
    path = Path(path)
    if not trust_source:
        raise ValueError("The original .pth uses pickle. Pass --trust-source only for a trusted official download.")
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark source is missing: {path}. Run the download command first.")
    fingerprint = sha256_file(path)
    if expected_sha256 is not None and fingerprint.lower() != expected_sha256.lower():
        raise ValueError("Source SHA-256 does not match --expected-sha256.")
    from ._stream import load_projected_state
    state = load_projected_state(path, progress=progress)
    benchmark = extract_state(state, provenance={
        "kind": "official", "source_name": path.name, "source_sha256": fingerprint,
        "source_url": SOURCE_URL, "authors_url": AUTHORS_URL,
        "reader": "projected legacy PyTorch pickle; unused learning curves omitted",
        "extractor_version": "1.1.0",
        "hardware_note": "Original recorded GPU timing, batch 256; no new hardware measurements.",
    })
    if verify_api:
        checked = compare_with_authors_api(state, benchmark)
        provenance = dict(benchmark.provenance, api_parity_architectures=checked)
        benchmark = replace(benchmark, provenance_json=json.dumps(provenance))
    return benchmark


def compare_with_authors_api(state, benchmark) -> int:
    """Check every exported mean against the authors' unmodified aggregation API."""
    try:
        from nas_201_api.api_utils import ArchResults
    except ImportError as exc:
        raise ImportError("API comparison requires the optional nas-bench-201 package.") from exc
    for row in benchmark.records():
        arch_id = row["arch_id"]
        info = ArchResults.create_from_state_dict(state["arch2infos"][arch_id]["full"])
        accuracy = info.get_metrics("cifar100", "x-valid", iepoch=199, is_random=False)["accuracy"]
        costs = info.get_compute_costs("cifar100")
        latency = costs["latency"]
        expected_latency = float("nan") if latency is None else latency * 1000
        if not np.isclose(row["validation_accuracy_pct"], accuracy, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Authors' validation-accuracy API differs at architecture {arch_id}.")
        if not np.isclose(row["latency_ms"], expected_latency, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"Authors' latency API differs at architecture {arch_id}.")
        for column, source_key in (("params_million", "params"), ("flops_million", "flops")):
            if not np.isclose(row[column], costs[source_key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Authors' auxiliary-cost API differs at architecture {arch_id}: {column}.")
    return len(benchmark)


def download_source(path: str | Path) -> Path:
    """Download the authors' file to an explicit path, leaving existing files intact."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Choose a fresh destination; source or partial download already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install downloader: python -m pip install gdown") from exc
    result = gdown.download(id=SOURCE_ID, output=str(temporary), quiet=False)
    if not result or not temporary.is_file():
        raise OSError("Download did not complete. The official download link is in README.md.")
    with temporary.open("rb") as stream:
        prefix = stream.read(512).lower()
    if b"<html" in prefix or b"<!doctype html" in prefix:
        raise OSError("Received a web page instead of the benchmark; use the official download link.")
    temporary.rename(path)
    return path
