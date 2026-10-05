"""Storage backend abstraction (local filesystem OR S3).

Routes write photos via the same `Storage` interface regardless of where
they end up. Selected via SNAPPY_STORAGE=local|s3.
"""
from .base    import Storage, StorageObject  # noqa: F401
from .factory import get_storage             # noqa: F401
