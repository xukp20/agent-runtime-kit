from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "ark-pi-http-test",
    host="127.0.0.1",
    port=int(os.environ["ARK_PI_MCP_HTTP_PORT"]),
    log_level="ERROR",
    stateless_http=True,
)


@mcp.tool()
def echo(marker: str) -> str:
    return f"ARK_MCP_SERVER:{marker}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
