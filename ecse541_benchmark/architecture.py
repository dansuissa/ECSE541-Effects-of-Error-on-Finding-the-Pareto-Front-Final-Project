"""Lossless cell encodings. Encoding IDs are separate from official benchmark IDs."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

OPERATIONS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
EDGES = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))
SPACE_SIZE = len(OPERATIONS) ** len(EDGES)
FEATURE_NAMES = tuple(
    f"edge_{src}_{dst}:{op}" for src, dst in EDGES for op in OPERATIONS
) + tuple(f"count:{op}" for op in OPERATIONS) + (
    "active_edges", "input_output_paths", "reachable_nodes", "useful_edges",
)


@dataclass(frozen=True)
class Architecture:
    operations: tuple[int, ...]

    def __post_init__(self):
        values = tuple(self.operations)
        if len(values) != 6 or any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or not 0 <= x < 5
            for x in values
        ):
            raise ValueError("A cell requires six integer operation codes in [0, 4].")
        object.__setattr__(self, "operations", tuple(int(x) for x in values))

    @classmethod
    def from_string(cls, text: str) -> "Architecture":
        if not isinstance(text, str):
            raise ValueError("Architecture must be a canonical NAS-Bench-201 string.")
        groups = text.split("+")
        if len(groups) != 3:
            raise ValueError(f"Expected three node groups: {text!r}")
        operations = []
        for dst, group in enumerate(groups, start=1):
            tokens = group.split("|")
            if tokens[0] or tokens[-1] or len(tokens) != dst + 2:
                raise ValueError(f"Malformed incoming edges for node {dst}: {group!r}")
            for src, token in enumerate(tokens[1:-1]):
                fields = token.split("~")
                if len(fields) != 2 or fields[1] != str(src) or fields[0] not in OPERATIONS:
                    raise ValueError(f"Invalid operation or source node: {token!r}")
                operations.append(OPERATIONS.index(fields[0]))
        return cls(tuple(operations))

    def to_string(self) -> str:
        groups = []
        start = 0
        for dst in range(1, 4):
            groups.append("|" + "|".join(
                f"{OPERATIONS[self.operations[start + src]]}~{src}" for src in range(dst)
            ) + "|")
            start += dst
        return "+".join(groups)

    @property
    def encoding_id(self) -> int:
        """Base-five key, with the first edge most significant; never an official ID."""
        value = 0
        for op in self.operations:
            value = 5 * value + op
        return value

    @classmethod
    def from_encoding_id(cls, value: int) -> "Architecture":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < SPACE_SIZE:
            raise ValueError(f"Encoding ID must be an integer in [0, {SPACE_SIZE - 1}].")
        operations = [0] * 6
        for i in range(5, -1, -1):
            value, operations[i] = divmod(int(value), 5)
        return cls(tuple(operations))

    def adjacency(self) -> np.ndarray:
        """Boolean adjacency[src, dst]; a 'none' edge is absent."""
        matrix = np.zeros((4, 4), dtype=bool)
        for (src, dst), op in zip(EDGES, self.operations):
            matrix[src, dst] = op != 0
        return matrix

    def features(self) -> np.ndarray:
        """30 categorical indicators, five operation counts, four graph descriptors."""
        one_hot = np.eye(5, dtype=float)[list(self.operations)].ravel()
        counts = np.bincount(self.operations, minlength=5)
        adjacency = self.adjacency()
        paths = np.zeros(4, dtype=int)
        paths[0] = 1
        for dst in range(1, 4):
            paths[dst] = sum(paths[src] for src in range(dst) if adjacency[src, dst])
        reaches_output = np.zeros(4, dtype=bool)
        reaches_output[3] = True
        for src in range(2, -1, -1):
            reaches_output[src] = any(adjacency[src, dst] and reaches_output[dst] for dst in range(src + 1, 4))
        useful_edges = sum(
            adjacency[src, dst] and paths[src] > 0 and reaches_output[dst] for src, dst in EDGES
        )
        return np.concatenate((one_hot, counts, [adjacency.sum(), paths[3], (paths > 0).sum(), useful_edges]))


def all_architectures():
    """Enumerate encodings without assigning official architecture IDs."""
    return (Architecture.from_encoding_id(i) for i in range(SPACE_SIZE))

