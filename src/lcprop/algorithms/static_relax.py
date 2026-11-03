"""Compatibility alias for :mod:`lcprop.lc.static_relax`."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("lcprop.lc.static_relax")
_sys.modules[__name__] = _canonical
