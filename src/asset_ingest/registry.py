"""Registry for all supported object-source ingest adapters."""

from __future__ import annotations

from .base import BaseIngestAdapter
from .dexnet import DexNetAdapter
from .graspnet import GraspNetAdapter
from .hope import HOPEAdapter
from .kit import KITAdapter
from .mso import MSOAdapter
from .objaverse import ObjaverseAdapter
from .realdex import RealDexAdapter
from .shapenet import ShapeNetCoreAdapter, ShapeNetSemAdapter
from .ycb import YCBAdapter


ADAPTERS: dict[str, type[BaseIngestAdapter]] = {
    "YCB": YCBAdapter,
    "RealDex": RealDexAdapter,
    "KIT": KITAdapter,
    "DexNet": DexNetAdapter,
    "GraspNet": GraspNetAdapter,
    "HOPE": HOPEAdapter,
    "MSO": MSOAdapter,
    "Objaverse": ObjaverseAdapter,
    "ShapeNetCore": ShapeNetCoreAdapter,
    "ShapeNetSem": ShapeNetSemAdapter,
}


def get_adapter(source_name: str) -> BaseIngestAdapter:
    if source_name not in ADAPTERS:
        raise KeyError(
            f"Unknown source '{source_name}'. Available: {sorted(ADAPTERS.keys())}"
        )
    return ADAPTERS[source_name]()
