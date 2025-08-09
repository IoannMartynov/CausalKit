"""
Backward-compatible shim for t-test API.

Historically, tests import `causalkit.inference.ttest` as a module.
The implementation lives under `causalkit.inference.att.ttest`.
This module re-exports the `ttest` function to preserve the import path.
"""
import sys
import types
from types import ModuleType
from causalkit.inference.att.ttest import ttest as _ttest

# Re-export name for direct use from this module

def ttest(*args, **kwargs):
    return _ttest(*args, **kwargs)

# Make this module callable so that `callable(causalkit.inference.ttest)` is True
class _CallableModule(ModuleType):
    def __call__(self, *args, **kwargs):
        return _ttest(*args, **kwargs)

_current = sys.modules.get(__name__)
if _current is not None and not isinstance(_current, _CallableModule):
    _new = _CallableModule(__name__, _current.__doc__)
    _new.__dict__.update(_current.__dict__)
    sys.modules[__name__] = _new
    # Also ensure parent package attribute refers to this callable module
    try:
        parent_pkg_name = __name__.rsplit('.', 1)[0]
        parent_pkg = sys.modules.get(parent_pkg_name)
        if parent_pkg is not None:
            setattr(parent_pkg, 'ttest', _new)
    except Exception:
        pass

__all__ = ["ttest"]
