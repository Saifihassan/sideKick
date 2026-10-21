"""Filesystem and web-search tools for the sidekick agent."""

import asyncio
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)

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

def _make_sync(coro):
    def _sync_wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return _sync_wrapper

# Enable synchronous invocation support for all MCP tools
for tool_ in fs_tools + browser_tools:
    if tool_.coroutine and not tool_.func:
        tool_.func = _make_sync(tool_.coroutine)


# ============================================================================
# Web Search Tools
# ============================================================================

@tool
def searxng_search(query: str, max_results: int = 5) -> str:
    """Search the web using SearXNG.

    Args:
        query: Search query string.
        max_results: Number of results to return.
    """
    searxng_url = os.getenv("SEARXNG_URL")
    if not searxng_url:
        return "Error: SEARXNG_URL environment variable is not set."
    try:
        response = requests.get(
            f"{searxng_url.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])[:max_results]
        if not results:
            return f"No results found for: {query}"
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('content', '')}")
        return "\n\n".join(output)
    except requests.RequestException as e:
        return f"SearXNG search error: {e}"


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily.

    Args:
        query: Search query string.
        max_results: Number of results to return.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])[:max_results]
        if not results:
            return f"No results found for: {query}"
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('content', '')}")
        return "\n\n".join(output)
    except requests.RequestException as e:
        return f"Tavily search error: {e}"


@tool
def delete_file(path: str) -> str:
    """Delete a file from the filesystem.

    Args:
        path: Path to the file to delete.
    """
    try:
        resolved_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(resolved_path):
            return f"Error: File '{path}' does not exist."
        if os.path.isdir(resolved_path):
            return f"Error: '{path}' is a directory. delete_file only operates on files."
        os.remove(resolved_path)
        return f"Successfully deleted '{path}'."
    except Exception as e:
        return f"Error deleting file '{path}': {e}"


tools = fs_tools + browser_tools + [searxng_search, tavily_search, delete_file]

__all__ = [
    "fs_tools",
    "browser_tools",
    "tools",
    "client",
    "searxng_search",
    "tavily_search",
    "delete_file",
]
