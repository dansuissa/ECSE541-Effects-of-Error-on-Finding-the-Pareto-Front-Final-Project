import itertools

import numpy as np
import pytest

from ecse541_benchmark import Architecture, EDGES, FEATURE_NAMES, SPACE_SIZE
from ecse541_benchmark.audit import inspect_space


def test_exhaustive_encodings_and_independent_graph_oracle():
    signatures = set()
    explicit_paths = ((0, 3), (0, 1, 3), (0, 2, 3), (0, 1, 2, 3))
    for encoding_id, genes in enumerate(itertools.product(range(5), repeat=6)):
        cell = Architecture(genes)
        assert cell.encoding_id == encoding_id
        assert Architecture.from_encoding_id(encoding_id) == cell
        assert Architecture.from_string(cell.to_string()) == cell
        signatures.add(cell.to_string())
        active = {edge for edge, op in zip(EDGES, genes) if op}
        paths = [path for path in explicit_paths if all(edge in active for edge in zip(path[:-1], path[1:]))]
        useful = set().union(*(set(zip(p[:-1], p[1:])) for p in paths))
        features = cell.features()
        assert features.shape == (len(FEATURE_NAMES),)
        assert features[FEATURE_NAMES.index("input_output_paths")] == len(paths)
        assert features[FEATURE_NAMES.index("useful_edges")] == len(useful)
        assert features[30:35].sum() == 6
        assert np.array_equal(features[:30].reshape(6, 5).argmax(axis=1), genes)
    assert len(signatures) == SPACE_SIZE


@pytest.mark.parametrize("genes", [(), (0,) * 5, (0,) * 7, (5,) * 6, (-1,) * 6, (1.0,) * 6, (True,) * 6])
def test_invalid_genes(genes):
    with pytest.raises(ValueError):
        Architecture(genes)


@pytest.mark.parametrize("value", [-1, SPACE_SIZE, 1.0, True, "1"])
def test_invalid_encoding_id(value):
    with pytest.raises(ValueError):
        Architecture.from_encoding_id(value)


@pytest.mark.parametrize("edit", [
    lambda s: s.replace("~0", "~1", 1),
    lambda s: s.replace("none", "conv", 1),
    lambda s: s + "|", lambda s: " " + s,
    lambda s: s.replace("+", "", 1),
    lambda s: s.replace("~0", "~00", 1),
    lambda s: s.replace("|none~1|", "|none~0|", 1),
])
def test_malformed_strings_rejected(edit):
    with pytest.raises(ValueError):
        Architecture.from_string(edit(Architecture((0,) * 6).to_string()))


def test_adjacency_orientation_and_no_input_path():
    cell = Architecture((0, 0, 3, 0, 0, 1))
    assert cell.adjacency()[1, 2] and cell.adjacency()[2, 3]
    assert not cell.adjacency()[2, 1]
    assert cell.features()[FEATURE_NAMES.index("input_output_paths")] == 0
    assert cell.features()[FEATURE_NAMES.index("useful_edges")] == 0


def test_space_audit_has_balanced_operation_slots():
    result = inspect_space()
    assert result["string_roundtrips_pass"] and result["encoding_roundtrips_pass"]
    assert result["n_connected_cells"] + result["n_disconnected_cells"] == SPACE_SIZE
    assert all(set(counts.values()) == {3125} for counts in result["operation_counts_per_edge"].values())
