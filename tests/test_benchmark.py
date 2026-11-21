import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from ecse541_benchmark import Trial, build_benchmark, load_benchmark
from ecse541_benchmark.audit import audit_benchmark, write_data_report
from ecse541_benchmark.benchmark import sha256_file


def test_units_aggregation_id_alignment_and_objective_order(benchmark):
    # Trial means are 3ms and 9ms: averaging all repetitions together would give 5ms, incorrectly.
    assert np.allclose(benchmark.objectives([0]), [[39., 6.]])
    assert benchmark.records()[0]["n_runs"] == 2
    assert benchmark.id_for(benchmark.architectures[0].to_string()) == 0
    assert benchmark.architectures[0].encoding_id != 0
    assert np.array_equal(benchmark.features([3, 0])[1], benchmark.architectures[0].features())
    assert np.array_equal(benchmark.eligible_ids, [0, 2, 3])
    assert benchmark.objectives([]).shape == (0, 2)
    assert benchmark.features([]).shape == (0, 39)


def test_missing_latency_is_retained_and_explicit(benchmark):
    with pytest.raises(ValueError, match="Missing latency"):
        benchmark.objectives()
    rows = benchmark.records()
    assert len(rows) == 4 and np.isnan(rows[1]["latency_ms"])
    assert np.isnan(rows[1]["validation_std_pct"])
    assert rows[2]["n_latency_runs"] == 2 and rows[2]["n_runs"] == 3
    audit = audit_benchmark(benchmark)
    assert audit["ineligible_arch_ids"] == [1]
    assert audit["n_cells_with_partial_latency"] == 1


def test_duplicate_timing_entries_are_exposed(benchmark):
    original = benchmark.trials[0]
    other = benchmark.trials[1]
    copied = Trial(other.arch_id, other.seed, other.validation_accuracy_pct,
                   original.latency_samples_s, other.params_million, other.flops_million)
    shared = build_benchmark(benchmark.architectures, (original, copied) + benchmark.trials[2:],
                             provenance=dict(benchmark.provenance), complete=False)
    report = audit_benchmark(shared)
    assert report["n_repeated_cells_with_identical_latency_records"] == 1
    assert report["n_repeated_cells_with_distinct_latency_records"] == 1
    assert report["n_cells_with_repeated_latency_records"] == 2


def test_objectives_and_features_cannot_mutate_reference(benchmark):
    objectives = benchmark.objectives([0])
    objectives[:] = -100
    features = benchmark.features([0])
    features[:] = 999
    assert np.allclose(benchmark.objectives([0]), [[39, 6]])
    assert benchmark.features([0]).max() < 999


@pytest.mark.parametrize("ids", [[-1], [4], [1.1], [True], [[0]], "0"])
def test_query_rejects_bad_ids(benchmark, ids):
    with pytest.raises(ValueError):
        benchmark.objectives(ids)


def test_bundle_roundtrip_preserves_raw_trials(benchmark, tmp_path):
    directory = benchmark.save(tmp_path / "bundle")
    restored = load_benchmark(directory, allow_synthetic=True, allow_partial=True)
    assert restored.trials == benchmark.trials
    assert restored.architectures == benchmark.architectures
    assert np.array_equal(restored.objectives(restored.eligible_ids), benchmark.objectives(benchmark.eligible_ids))
    with pytest.raises(ValueError, match="Synthetic"):
        load_benchmark(directory)
    with pytest.raises(ValueError, match="Partial"):
        load_benchmark(directory, allow_synthetic=True)
    with pytest.raises(FileExistsError):
        benchmark.save(directory)


@pytest.mark.parametrize("filename", ["benchmark.csv", "trials.csv"])
def test_corrupted_files_rejected(benchmark, tmp_path, filename):
    directory = benchmark.save(tmp_path / "bundle")
    with (directory / filename).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Checksum"):
        load_benchmark(directory, allow_synthetic=True, allow_partial=True)


def test_recomputed_aggregates_detect_semantic_corruption(benchmark, tmp_path):
    directory = benchmark.save(tmp_path / "bundle")
    table = directory / "benchmark.csv"
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["validation_error_pct"] = "12.0"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["benchmark.csv"] = sha256_file(table)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Derived value mismatch"):
        load_benchmark(directory, allow_synthetic=True, allow_partial=True)


def test_duplicate_trials_rejected_and_variable_cost_metadata_flagged(benchmark):
    with pytest.raises(ValueError, match="Duplicate trial"):
        build_benchmark(benchmark.architectures, benchmark.trials + (benchmark.trials[0],), provenance=dict(benchmark.provenance), complete=False)
    t = benchmark.trials[0]
    changed = Trial(t.arch_id, t.seed, t.validation_accuracy_pct, t.latency_samples_s, 999., t.flops_million)
    varied = build_benchmark(benchmark.architectures, (changed,) + benchmark.trials[1:], provenance=dict(benchmark.provenance), complete=False)
    row = varied.records()[0]
    assert row["cost_metadata_consistent"] == 0
    assert np.isclose(row["params_million"], (999. + benchmark.trials[1].params_million) / 2)
    assert varied.trials[0].params_million == 999.
    assert audit_benchmark(varied)["inconsistent_cost_metadata_arch_ids"] == [0]
    assert np.array_equal(varied.objectives(varied.eligible_ids), benchmark.objectives(benchmark.eligible_ids))


@pytest.mark.parametrize("changes", [
    {"validation_accuracy_pct": -1}, {"validation_accuracy_pct": 101},
    {"validation_accuracy_pct": float("nan")}, {"latency_samples_s": [float("inf")]},
    {"arch_id": 1.5}, {"seed": True}, {"params_million": -1},
])
def test_invalid_measurements(changes):
    fields = dict(arch_id=0, seed=777, validation_accuracy_pct=60., latency_samples_s=(.01,), params_million=.5, flops_million=10.)
    fields.update(changes)
    with pytest.raises(ValueError):
        Trial(**fields)


def test_data_figure_and_json_are_generated_with_synthetic_label(benchmark, tmp_path):
    result = write_data_report(benchmark, tmp_path)
    assert result["data_kind"] == "synthetic_test"
    assert (tmp_path / "data_overview.png").stat().st_size > 10_000
    assert json.loads((tmp_path / "data_audit.json").read_text())["n_architectures"] == 4


def test_cli_reports_missing_source_and_trust_gate(tmp_path):
    result = subprocess.run([sys.executable, "-m", "ecse541_benchmark", "build", "--source", str(tmp_path / "missing.pth"),
                             "--out", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode == 2 and "--trust-source" in result.stderr
