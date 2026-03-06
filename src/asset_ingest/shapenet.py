"""Backward-compatible ShapeNet adapter exports."""

from .shapenet_core import ShapeNetCoreAdapter
from .shapenet_sem import ShapeNetSemAdapter

__all__ = ["ShapeNetCoreAdapter", "ShapeNetSemAdapter"]

