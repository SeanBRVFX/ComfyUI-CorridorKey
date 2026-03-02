"""
@author: local
@title: ComfyUI-CorridorKey
@nickname: CorridorKey
@description: ComfyUI node for CorridorKey inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .corridor_key import schedule_upstream_check
    from .nodes import CorridorKey
except ImportError:
    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    from corridor_key import schedule_upstream_check
    from nodes import CorridorKey

NODE_CLASS_MAPPINGS = {
    "CorridorKey": CorridorKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CorridorKey": "CorridorKey",
}

schedule_upstream_check()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
