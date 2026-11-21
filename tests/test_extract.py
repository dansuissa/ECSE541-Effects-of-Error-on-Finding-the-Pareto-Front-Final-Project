import numpy as np
import pytest

from ecse541_benchmark import Architecture, SPACE_SIZE
from ecse541_benchmark.architecture import all_architectures
from ecse541_benchmark.extract import extract_file, extract_state


@pytest.mark.parametrize("corruption", ["arch_id", "arch_str", "epochs", "seed", "split", "accuracy", "coverage", "seeds"])
def test_source_checks_reject_misaligned_records(source_state, corruption):
    state = source_state
    info = state["arch2infos"][0]["full"]
    run = info["all_results"][("cifar100", 777)]
    if corruption == "arch_id":
        info["arch_index"] = 3
    elif corruption == "arch_str":
        info["arch_str"] = state["meta_archs"][1]
    elif corruption == "epochs":
        run["epochs"] = 12
    elif corruption == "seed":
        run["seed"] = 999
    elif corruption == "split":
        run["eval_names"].remove("x-valid")
    elif corruption == "accuracy":
        del run["eval_acc1es"]["x-valid@199"]
    elif corruption == "coverage":
        state["evaluated_indexes"].pop()
    elif corruption == "seeds":
        info["dataset_seed"]["cifar100"].append(999)
    with pytest.raises(ValueError):
        extract_state(state, provenance={"kind": "synthetic_test"}, complete=False)


def test_partial_source_cannot_be_accidentally_marked_complete(source_state):
    with pytest.raises(ValueError, match="15625"):
        extract_state(source_state, provenance={"kind": "synthetic_test"})


def test_zero_padded_source_seed_labels(source_state, benchmark):
    for entry in source_state["arch2infos"].values():
        for info in entry.values():
            for (_, seed), run in info["all_results"].items():
                run["seed"] = f"{seed:04d}"
    normalized = extract_state(source_state, provenance={"kind": "synthetic_test"}, complete=False)
    assert normalized.trials == benchmark.trials


def test_nondigit_seed_label_is_not_coerced(source_state):
    source_state["arch2infos"][0]["full"]["all_results"][("cifar100", 777)]["seed"] = "777.0"
    with pytest.raises(ValueError, match="identity"):
        extract_state(source_state, provenance={"kind": "synthetic_test"}, complete=False)


@pytest.mark.parametrize("key, value", [("channel", 8), ("num_cells", 3), ("class_num", 10)])
def test_mixed_network_configurations_rejected(source_state, key, value):
    source_state["arch2infos"][0]["full"]["all_results"][("cifar100", 777)]["arch_config"][key] = value
    with pytest.raises(ValueError, match="Network configuration"):
        extract_state(source_state, provenance={"kind": "synthetic_test"}, complete=False)


def test_actual_torch_serialization_and_sha256_guard(source_state, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "fixture.pth"
    torch.save(source_state, path)
    with pytest.raises(ValueError, match="pickle"):
        extract_file(path)
    with pytest.raises(ValueError, match="SHA-256"):
        extract_file(path, trust_source=True, expected_sha256="0" * 64)
    restored = torch.load(path, weights_only=False, map_location="cpu")
    benchmark = extract_state(restored, provenance={"kind": "synthetic_test"}, complete=False)
    assert np.allclose(benchmark.objectives([0]), [[39., 6.]])


def test_against_authors_actual_api(source_state):
    api_module = pytest.importorskip("nas_201_api", reason="Optional parity check: install nas-bench-201 to run.")
    api = api_module.NASBench201API(source_state, verbose=False)
    benchmark = extract_state(source_state, provenance={"kind": "synthetic_test"}, complete=False)
    for arch_id in range(len(benchmark)):
        info = api.query_meta_info_by_index(arch_id, hp="200")
        accuracy = info.get_metrics("cifar100", "x-valid", iepoch=199, is_random=False)["accuracy"]
        latency = info.get_compute_costs("cifar100")["latency"]
        row = benchmark.records()[arch_id]
        assert np.isclose(row["validation_accuracy_pct"], accuracy)
        if latency is None:
            assert np.isnan(row["latency_ms"])
        else:
            assert np.isclose(row["latency_ms"], latency * 1000)


def test_full_space_build_and_load_with_synthetic_metrics(tmp_path):
    from ecse541_benchmark import build_benchmark, load_benchmark, Trial
    cells = tuple(all_architectures())
    # All cells exercise row/feature alignment; these measurements have no scientific meaning.
    trials = [Trial(i, 777, 50. + i % 30, (.001 + (i % 100) * .0001,), .5, 10.) for i in range(SPACE_SIZE)]
    benchmark = build_benchmark(cells[::-1], trials, provenance={"kind": "synthetic_test"})
    benchmark.save(tmp_path / "full")
    restored = load_benchmark(tmp_path / "full", allow_synthetic=True)
    assert len(restored) == SPACE_SIZE
    assert restored.features().shape == (SPACE_SIZE, 39)
    assert np.array_equal(restored.objectives(), benchmark.objectives())
    assert restored.id_for(Architecture.from_encoding_id(0)) == SPACE_SIZE - 1


@pytest.mark.parametrize("ordered", [False, True])
def test_projected_reader_matches_full_torch_load(source_state, tmp_path, ordered):
    from collections import OrderedDict
    from ecse541_benchmark._stream import load_projected_state
    torch = pytest.importorskip("torch")
    for entry in source_state["arch2infos"].values():
        for info in entry.values():
            for run in info["all_results"].values():
                mapping = OrderedDict if ordered else dict
                run["train_times"] = mapping((i, .1 + i) for i in range(200))
                run["eval_acc1es"] = mapping(run["eval_acc1es"])
                run["eval_losses"] = mapping(run["eval_losses"])
                run["params"] = np.float64(run["params"])
    path = tmp_path / "legacy.pth"
    torch.save(source_state, path, _use_new_zipfile_serialization=False)
    projected = load_projected_state(path)
    full = torch.load(path, weights_only=False, map_location="cpu")
    a = extract_state(projected, provenance={"kind": "synthetic_test"}, complete=False)
    b = extract_state(full, provenance={"kind": "synthetic_test"}, complete=False)
    assert a.trials == b.trials
    for entry in projected["arch2infos"].values():
        for info in entry.values():
            for run in info["all_results"].values():
                assert not run["train_acc1es"]
                assert run["train_times"] is None
                assert set(run["eval_acc1es"]) == {"x-valid@199"}


def test_projected_reader_rejects_unexpected_tensor_storages(tmp_path):
    from ecse541_benchmark._stream import load_projected_state
    torch = pytest.importorskip("torch")
    path = tmp_path / "tensor.pth"
    torch.save({"tensor": torch.zeros(3)}, path, _use_new_zipfile_serialization=False)
    with pytest.raises(Exception, match="tensor/storage"):
        load_projected_state(path)
