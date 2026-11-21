"""Memory-efficient projection of the original legacy PyTorch benchmark pickle.

Only learning curves unrelated to the final validation objective are discarded.
All pickle references remain valid; repeated string objects share storage, and
the dense memo uses a list instead of a large integer-keyed dictionary.
"""

import pickle
import sys
from collections import OrderedDict
from pathlib import Path


class _DiscardCurve(dict):
    def __setitem__(self, key, value):
        pass


class _FinalValidation(dict):
    def __setitem__(self, key, value):
        if key == "x-valid@199":
            super().__setitem__(key, value)


class _DenseMemo:
    def __init__(self):
        self.values = []

    def __len__(self):
        return len(self.values)

    def __setitem__(self, index, value):
        if isinstance(value, str):
            value = sys.intern(value)
        if index == len(self.values):
            self.values.append(value)
        elif 0 <= index < len(self.values):
            self.values[index] = value
        else:
            raise pickle.UnpicklingError("Expected a dense memo in the original benchmark pickle.")

    def __getitem__(self, index):
        if not 0 <= index < len(self.values):
            raise pickle.UnpicklingError(f"Invalid pickle memo reference {index}.")
        return self.values[index]


class _ProjectedUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()
    discarded_fields = {"train_acc1es", "train_acc5es", "train_losses", "train_times", "eval_times"}

    def __init__(self, stream, *, progress=None):
        super().__init__(stream, encoding="utf-8")
        self.memo = _DenseMemo()
        self.progress = progress
        self.records_seen = 0

    def load_empty_dictionary(self):
        field = self.stack[-1] if self.stack and isinstance(self.stack[-1], str) else None
        if field in self.discarded_fields:
            value = _DiscardCurve()
        elif field in {"eval_acc1es", "eval_losses"}:
            value = _FinalValidation()
        else:
            value = {}
        self.append(value)

    dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary

    def load_setitems(self):
        super().load_setitems()
        value = self.stack[-1]
        if type(value) is dict and "epochs" in value and isinstance(value.get("train_times"), _DiscardCurve):
            value["train_times"] = None
        if type(value) is dict and "full" in value and "less" in value:
            self.records_seen += 1
            if self.progress and (self.records_seen == 1 or self.records_seen % 500 == 0):
                self.progress(self.records_seen)

    dispatch[pickle.SETITEMS[0]] = load_setitems

    def load_reduce(self):
        args = self.stack.pop()
        function = self.stack[-1]
        field = self.stack[-2] if len(self.stack) > 1 and isinstance(self.stack[-2], str) else None
        if function is OrderedDict and not args and field in self.discarded_fields:
            value = _DiscardCurve()
        elif function is OrderedDict and not args and field in {"eval_acc1es", "eval_losses"}:
            value = _FinalValidation()
        else:
            value = function(*args)
        self.stack[-1] = value

    dispatch[pickle.REDUCE[0]] = load_reduce

    def persistent_load(self, identity):
        raise pickle.UnpicklingError("Unexpected tensor/storage in the weight-free benchmark metadata.")


def load_projected_state(path: str | Path, *, progress=None):
    """Load original protocol-2 metadata without retaining unused learning curves.

Like the original torch.load path, this deserializes pickle and therefore must
only be called after the caller has established trust in the source file.
"""
    with Path(path).open("rb", buffering=1024 * 1024) as stream:
        magic = pickle.load(stream)
        version = pickle.load(stream)
        system = pickle.load(stream)
        if magic != 119547037146038801333356 or version != 1001 or not isinstance(system, dict):
            raise ValueError("Expected the original legacy NAS-Bench-201 PyTorch file format.")
        state = _ProjectedUnpickler(stream, progress=progress).load()
        storage_keys = pickle.load(stream)
        if storage_keys:
            raise ValueError("Unexpected tensor weights in benchmark metadata.")
        if progress and len(state.get("arch2infos", {})) % 500:
            progress(len(state["arch2infos"]))
    return state
