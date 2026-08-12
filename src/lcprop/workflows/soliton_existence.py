"""Compatibility alias for ``lcprop.lc.workflows.soliton_existence``."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("lcprop.lc.workflows.soliton_existence")
_sys.modules[__name__] = _canonical
