"""Optional notebook interfaces, loaded only when explicitly requested."""

__all__ = ["InteractiveLC", "InteractiveBatch"]


def __getattr__(name):
    if name == "InteractiveLC":
        from .interactive_lc import InteractiveLC

        globals()[name] = InteractiveLC
        return InteractiveLC
    if name == "InteractiveBatch":
        from .interactive_batch import InteractiveBatch

        globals()[name] = InteractiveBatch
        return InteractiveBatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
