import os
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load environment variables
load_dotenv(override=True)


# ============================================================================
# File Handling Tools
# ============================================================================

@tool
def read_file(file_path: str) -> str:
    """Read and return the text contents of a file.

    Args:
        file_path: The relative or absolute path to the file to read.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is a directory, not a file."
        
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except PermissionError:
        return f"Error: Permission denied when attempting to read '{file_path}'."
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file with the provided text content.

    Args:
        file_path: The relative or absolute path to the file to write.
        content: The text content to write into the file.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{file_path}'."
    except PermissionError:
        return f"Error: Permission denied when attempting to write to '{file_path}'."
    except Exception as e:
        return f"Error writing to file '{file_path}': {str(e)}"


@tool
def append_file(file_path: str, content: str) -> str:
    """Append text content to an existing file, or create it if it doesn't exist.

    Args:
        file_path: The relative or absolute path to the file.
        content: The text content to append to the file.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully appended {len(content)} characters to '{file_path}'."
    except PermissionError:
        return f"Error: Permission denied when attempting to append to '{file_path}'."
    except Exception as e:
        return f"Error appending to file '{file_path}': {str(e)}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """List the contents of a directory, including files and subdirectories.

    Args:
        directory_path: The path to the directory to list (defaults to current directory '.').
    """
    try:
        path = Path(directory_path).expanduser().resolve()
        if not path.exists():
            return f"Error: Directory '{directory_path}' does not exist."
        if not path.is_dir():
            return f"Error: '{directory_path}' is not a directory."

        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return f"Directory '{directory_path}' is empty."

        lines = [f"Contents of '{directory_path}':"]
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  [DIR]  {entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                    lines.append(f"  [FILE] {entry.name} ({size} bytes)")
                except Exception:
                    lines.append(f"  [FILE] {entry.name}")
        return "\n".join(lines)
    except PermissionError:
        return f"Error: Permission denied accessing directory '{directory_path}'."
    except Exception as e:
        return f"Error listing directory '{directory_path}': {str(e)}"


@tool
def delete_file(file_path: str) -> str:
    """Delete a specified file.

    Args:
        file_path: The relative or absolute path to the file to delete.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if path.is_dir():
            return f"Error: '{file_path}' is a directory. This tool only deletes files."
        
        path.unlink()
        return f"Successfully deleted file '{file_path}'."
    except PermissionError:
        return f"Error: Permission denied when attempting to delete '{file_path}'."
    except Exception as e:
        return f"Error deleting file '{file_path}': {str(e)}"


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
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            output.append(f"{i}. {title}\nURL: {url}\nSnippet: {content}")
        return "\n\n".join(output)
    except Exception as e:
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
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            output.append(f"{i}. {title}\nURL: {url}\nSnippet: {content}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Tavily search error: {e}"


# Convenient tool list for agent integration
tools = [
    read_file,
    write_file,
    append_file,
    list_directory,
    delete_file,
    searxng_search,
    tavily_search,
]

__all__ = [
    "read_file",
    "write_file",
    "append_file",
    "list_directory",
    "delete_file",
    "searxng_search",
    "tavily_search",
    "tools",
]

