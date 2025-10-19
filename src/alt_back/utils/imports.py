from __future__ import annotations

import importlib
from typing import Any


def import_from_string(dotted_path: str) -> Any:
    """Import an attribute from a module path string."""
    try:
        module_path, attr_name = dotted_path.rsplit(".", 1)
    except ValueError as exc:
        raise ImportError(f"Invalid dotted path '{dotted_path}'") from exc

    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_path}' has no attribute '{attr_name}'") from exc

