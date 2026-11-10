"""Optional notebook interfaces, loaded only when explicitly requested."""

__all__ = ["InteractiveLC", "InteractiveBatch", "LCNotebook", "PRNotebook"]


def __getattr__(name):
    if name == "InteractiveLC":
        from .interactive_lc import InteractiveLC

        globals()[name] = InteractiveLC
        return InteractiveLC
    if name in {"InteractiveBatch", "LCNotebook", "PRNotebook"}:
        from . import interactive_batch

        controller = getattr(interactive_batch, name)
        globals()[name] = controller
        return controller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
