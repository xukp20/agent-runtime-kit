from mcp.server.fastmcp import FastMCP


mcp = FastMCP("ark-pi-test", log_level="ERROR")


@mcp.tool()
def echo(marker: str) -> str:
    return f"ARK_MCP_SERVER:{marker}"


if __name__ == "__main__":
    mcp.run(transport="stdio")

