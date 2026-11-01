"""Recursive bitwise comparison retained from the streaming-prefix validation."""
from dataclasses import fields, is_dataclass
import numpy as np

def assert_same(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    elif is_dataclass(a):
        for f in fields(a):
            assert_same(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b
