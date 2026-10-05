"""Snappy HTTP + WebSocket layer.

Modules:
    globals       — singletons + SESSIONS dict (the DI container)
    session       — Session dataclass
    ws_protocol   — RFC 6455 WS frame helpers
    pipeline      — per-frame ML orchestration + worker pool
    routes        — pure-function HTTP handlers
    server        — asyncio.Protocol + dispatch
"""
