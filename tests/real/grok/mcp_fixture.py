import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context

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


@server.tool()
def probe_identity(ctx: Context) -> dict[str, str]:
    """Return the current server-observed Step, round and result-profile identity."""
    request = ctx.request_context.request
    if request is not None:
        return {"step": request.headers.get("x-step", ""), "round": request.headers.get("x-round", ""),
                "profile": request.headers.get("x-ark-mcp-result-profile", "")}
    return {"step": os.environ.get("ARK_STEP_ID", ""), "round": os.environ.get("ARK_ROUND", ""),
            "profile": os.environ.get("ARK_MCP_RESULT_PROFILE", "")}


if __name__ == "__main__":
    server.run(transport=os.environ.get("ARK_MCP_TRANSPORT", "stdio"))
