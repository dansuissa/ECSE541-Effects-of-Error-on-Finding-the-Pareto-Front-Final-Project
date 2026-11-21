"""Small synthetic records in the authors' serialized format; never experimental data."""

from copy import deepcopy

import pytest

from ecse541_benchmark import Architecture
from ecse541_benchmark.extract import extract_state


@pytest.fixture
def source_state():
    # Deliberately not encoding order: official IDs must follow meta_archs.
    cells = [Architecture((3, 1, 2, 0, 4, 3)), Architecture((0,) * 6),
             Architecture((1,) * 6), Architecture((4,) * 6)]
    state = {"meta_archs": [a.to_string() for a in cells], "arch2infos": {}, "evaluated_indexes": [3, 0, 2, 1]}
    seeds = [(777, 888), (777,), (777, 888, 999), (777,)]
    for arch_id, arch_str in enumerate(state["meta_archs"]):
        runs = {}
        for j, seed in enumerate(seeds[arch_id]):
            latency = [(.002 + arch_id * .001), (.004 + arch_id * .001)] if j == 0 else [.009]
            if arch_id == 1 or (arch_id == 2 and j == 2):
                latency = None
            run = {
                "name": "synthetic", "net_state_dict": None,
                "arch_config": {"arch_str": arch_str, "channel": 16, "num_cells": 5, "class_num": 100},
                "seed": seed, "epochs": 200, "params": .5 + arch_id, "flop": 10. + arch_id,
                "latency": latency, "eval_names": ["x-valid", "x-test", "ori-test"],
                "eval_acc1es": {"x-valid@199": 60. + arch_id + j * 2, "x-test@199": 92., "ori-test@199": 91.},
                "eval_losses": {"x-valid@199": 1., "x-test@199": .2, "ori-test@199": .3},
                "eval_times": {}, "train_times": None,
                "train_acc1es": {199: 99.}, "train_acc5es": None, "train_losses": {199: .1},
            }
            runs[("cifar100", seed)] = run
        full = {"arch_index": arch_id, "arch_str": arch_str, "dataset_seed": {"cifar100": list(seeds[arch_id])},
                "all_results": runs, "clear_net_done": False}
        state["arch2infos"][arch_id] = {"full": full, "less": deepcopy(full)}
    return state


@pytest.fixture
def benchmark(source_state):
    return extract_state(source_state, provenance={"kind": "synthetic_test", "description": "unit-test fixture"}, complete=False)
