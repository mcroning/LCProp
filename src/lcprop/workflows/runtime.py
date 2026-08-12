"""Compatibility alias for ``lcprop.lc.workflows.runtime``."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("lcprop.lc.workflows.runtime")
_sys.modules[__name__] = _canonical
