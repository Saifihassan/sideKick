"""Filesystem tools for the sidekick agent provided via MCP."""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/home/hassan",
            ],
        }
    }
)
browser_client = MultiServerMCPClient({
    "playwright": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
    }
})

fs_tools = asyncio.run(client.get_tools())
browser_tools = asyncio.run(browser_client.get_tools())

tools = fs_tools + browser_tools

def _make_sync(coro):
    def _sync_wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return _sync_wrapper

# Enable synchronous invocation support for all MCP tools
for tool in tools:
    if tool.coroutine and not tool.func:
        tool.func = _make_sync(tool.coroutine)

__all__ = ["fs_tools", "tools", "client", "browser_tools"]
