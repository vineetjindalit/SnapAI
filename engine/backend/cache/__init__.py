"""Cache abstraction — in-memory dict for dev, Redis for production."""
from .client import Cache, get_cache  # noqa: F401
