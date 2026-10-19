# sidekick

A small tool-using LLM agent with a self-critiquing evaluator loop.

The agent answers your requests using a set of filesystem and web-search
tools, while a second "evaluator" model critiques each answer and re-prompts
the worker (up to 3 times) until the request is satisfied.

## Features

- **Tool-using worker agent** built with `langchain` / `langgraph`.
- **Tools**: read/write/append/list/delete files, plus SearXNG and Tavily web
  search.
- **Evaluator loop**: a second model checks the answer against your request
  and requests revisions when needed.
- **Persistent memory**: conversations are checkpointed to a local SQLite
  database (`sidekick.db`).
- **Todo-list middleware** so the agent can plan multi-step tasks.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (recommended)

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file in the project root with your credentials:

   ```dotenv
   VELONA_BASE_URL=...
   VELONA_API_KEY=...
   GROQ_BASE_URL=...
   GROQ_API_KEY=...
   # Optional, for the web-search tools:
   SEARXNG_URL=...
   TAVILY_API_KEY=...
   ```

## Usage

Run the interactive agent:

```bash
uv run sidekick
```

or, without installing the entry point:

```bash
uv run python -m sidekick.agent
```

Type your request at the `You:` prompt. Enter `exit` or `quit` to stop.

## Project layout

```
src/sidekick/
├── __init__.py   # exposes main() for the `sidekick` console script
├── agent.py      # worker + evaluator agent loop
└── tools.py      # LangChain tools used by the agent
```

## Development

Scan the project for common problems and auto-fix the safe ones:

```bash
python project_doctor.py . --dry-run   # preview
python project_doctor.py .             # apply
```

Run the tests:

```bash
python test_project_doctor.py
```
