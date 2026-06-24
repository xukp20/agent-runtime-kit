"""Shared runtime service containers and contexts."""

from .contexts import RuntimeContext
from .services import ARKServices, AppServices, RuntimePausedError, RuntimePauseController

_MCP_GATEWAY_EXPORTS = {
    "RuntimeMcpContextResolver",
    "RuntimeMcpToolGateway",
    "RuntimeToolContext",
    "RuntimeToolContextError",
    "RuntimeToolIdentity",
    "runtime_context_from_fastmcp_context",
}


def __getattr__(name: str):
    if name in _MCP_GATEWAY_EXPORTS:
        from . import mcp_tool_gateway

        return getattr(mcp_tool_gateway, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARKServices",
    "AppServices",
    "RuntimeMcpContextResolver",
    "RuntimeMcpToolGateway",
    "RuntimePauseController",
    "RuntimePausedError",
    "RuntimeContext",
    "RuntimeToolContext",
    "RuntimeToolContextError",
    "RuntimeToolIdentity",
    "runtime_context_from_fastmcp_context",
]
