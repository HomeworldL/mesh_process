"""Object source ingest package (download, organize, manifest, verify)."""

from .base import (
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
)
from .manifest import IngestManifest, ObjectRecord, validate_manifest_dict
from .registry import ADAPTERS, get_adapter

__all__ = [
    "DEFAULT_MASS_KG",
    "BaseIngestAdapter",
    "DownloadReport",
    "IngestConfig",
    "OrganizeReport",
    "IngestManifest",
    "ObjectRecord",
    "validate_manifest_dict",
    "ADAPTERS",
    "get_adapter",
]
