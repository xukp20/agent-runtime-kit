from mcp.server.fastmcp import FastMCP


server = FastMCP("ark-openai-agents-test")


@server.tool()
def echo_marker(value: str) -> str:
    """Return a deterministic MCP marker."""

    return f"ARK_MCP:{value}"


if __name__ == "__main__":
    server.run(transport="stdio")
