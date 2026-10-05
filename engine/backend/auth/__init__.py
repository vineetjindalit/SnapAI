"""Phase 3 starter: JWT auth + user accounts.

Disabled by default. Enable with `SNAPPY_AUTH=1` to require Bearer tokens
on /sessions endpoints. Schema lives in utils.persistence (`users` table).
"""
from .jwt_auth import JWTAuth, hash_password, verify_password  # noqa: F401
