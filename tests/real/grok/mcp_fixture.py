import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

server = FastMCP(
    "ark_probe", host="127.0.0.1", port=int(os.environ.get("ARK_MCP_PORT", "18976"))
)


@server.tool()
def probe_echo(token: str) -> str:
    """Return the given token prefixed by ARK_MCP_OK."""
    log = os.environ.get("ARK_MCP_CALL_LOG")
    if log:
        with Path(log).open("a") as f:
            f.write(token + "\n")
    return "ARK_MCP_OK:" + token


if __name__ == "__main__":
    server.run(transport=os.environ.get("ARK_MCP_TRANSPORT", "stdio"))
