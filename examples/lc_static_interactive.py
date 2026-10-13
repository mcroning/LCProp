"""Compatibility import for the packaged interactive LC controller."""
import sys
from lcprop.notebook import interactive_lc

# Preserve module identity for callers and callback monkeypatches.
sys.modules[__name__] = interactive_lc
