# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management. Run all commands from the repo root.

```bash
# Install dependencies
uv sync

# Run the backend server (from backend/)
cd backend && uv run uvicorn app:app --reload --port 8000

# Run all tests
uv run pytest backend/tests/

# Run a single test file
uv run pytest backend/tests/test_ai_generator.py

# Run a single test by name
uv run pytest backend/tests/test_ai_generator.py::TestToolUsePath::test_executes_tool_when_stop_reason_is_tool_use
```

**Environment**: Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. The backend reads it via `python-dotenv` on startup.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for course materials with a FastAPI backend and a plain HTML/JS/CSS frontend.

### Request flow

1. Frontend (`frontend/`) sends a `POST /api/query` with `{query, session_id}`.
2. `app.py` passes the query to `RAGSystem.query()`.
3. `RAGSystem` builds a prompt and calls `AIGenerator.generate_response()` with two registered tools.
4. `AIGenerator` runs a tool-use loop (max `MAX_TOOL_ROUNDS = 2`). Claude decides whether to invoke tools before giving a final answer.
5. Tools (`search_tools.py`) call into `VectorStore` (ChromaDB) and return formatted text to Claude.
6. After the loop, `ToolManager.get_last_sources()` collects citation metadata for the UI.
7. `SessionManager` appends the exchange to in-memory conversation history (capped at `MAX_HISTORY`).

### Key components

| File | Role |
|------|------|
| `backend/config.py` | Single `Config` dataclass; reads `.env`. All tunable parameters live here. |
| `backend/rag_system.py` | Top-level orchestrator. Wires together all subsystems. |
| `backend/ai_generator.py` | Anthropic API client. Handles the tool-use loop; `MAX_TOOL_ROUNDS` caps sequential calls. |
| `backend/vector_store.py` | ChromaDB wrapper. Two collections: `course_catalog` (titles/outlines) and `course_content` (chunked text). Course title is the ChromaDB document ID. |
| `backend/search_tools.py` | `Tool` ABC + `CourseSearchTool` / `CourseOutlineTool` implementations + `ToolManager`. |
| `backend/document_processor.py` | Parses `.txt` course files into `Course`/`Lesson`/`CourseChunk` models, then sentence-chunks the content. |
| `backend/session_manager.py` | In-memory session store; conversation history is serialised as a string injected into the system prompt. |

### Course document format

The document processor expects `.txt` files with this header:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <lesson title>
...
```

Files are loaded from `../docs` (relative to `backend/`) at startup. Re-ingestion is idempotent — existing course titles are skipped.

### ChromaDB persistence

ChromaDB data is written to `backend/chroma_db/` (gitignored). Delete this directory to force a full re-index on next startup.

### Tool-use loop invariant

`AIGenerator.generate_response` strips `tools`/`tool_choice` from the **final** synthesis API call so Claude always produces a plain-text answer at the end, even after tool rounds or tool failures.

### Model IDs

Claude 4 model IDs use the format `claude-{family}-{major}-{minor}` (no date suffix). `test_ai_generator.py::TestModelIDValidity` will fail if `config.ANTHROPIC_MODEL` uses a Claude 3-style date-suffixed ID.
