"""
Tests for RAGSystem.query() in rag_system.py.

All external components (VectorStore, AIGenerator, SessionManager,
DocumentProcessor) are mocked.  These tests verify that query() correctly
wires its components together and that exceptions from AIGenerator propagate
to the caller (which is what triggers "query failed" in the frontend).
"""
import pytest
from unittest.mock import MagicMock, patch, call

from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rag():
    """
    RAGSystem with every constructor dependency mocked out.
    All patches target the names as imported inside rag_system.py.
    """
    with (
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.VectorStore"),
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.SessionManager"),
    ):
        cfg = MagicMock()
        cfg.ANTHROPIC_API_KEY = "test_key"
        cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"
        cfg.CHUNK_SIZE = 800
        cfg.CHUNK_OVERLAP = 100
        cfg.MAX_RESULTS = 5
        cfg.MAX_HISTORY = 2
        cfg.CHROMA_PATH = "/tmp/test_chroma"
        cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"

        system = RAGSystem(cfg)
        # Default: AI returns a string, no sources
        system.ai_generator.generate_response.return_value = "Mocked answer."
        system.tool_manager = MagicMock()
        system.tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        system.tool_manager.get_last_sources.return_value = []
        yield system


# ---------------------------------------------------------------------------
# Return-value contract
# ---------------------------------------------------------------------------

class TestQueryReturnValues:
    def test_returns_tuple_of_str_and_list(self, rag):
        answer, sources = rag.query("What is RAG?")

        assert isinstance(answer, str)
        assert isinstance(sources, list)

    def test_answer_comes_from_ai_generator(self, rag):
        rag.ai_generator.generate_response.return_value = "Deep answer about RAG."

        answer, _ = rag.query("What is RAG?")

        assert answer == "Deep answer about RAG."

    def test_sources_come_from_tool_manager(self, rag):
        rag.tool_manager.get_last_sources.return_value = [
            {"label": "MCP Course - Lesson 1", "url": "http://example.com"}
        ]

        _, sources = rag.query("What is MCP?")

        assert sources == [{"label": "MCP Course - Lesson 1", "url": "http://example.com"}]

    def test_returns_empty_sources_when_no_tool_was_called(self, rag):
        rag.tool_manager.get_last_sources.return_value = []

        _, sources = rag.query("What is 2+2?")

        assert sources == []


# ---------------------------------------------------------------------------
# AI generator is called correctly
# ---------------------------------------------------------------------------

class TestQueryCallsAIGenerator:
    def test_calls_generate_response_once(self, rag):
        rag.query("question")

        rag.ai_generator.generate_response.assert_called_once()

    def test_passes_tools_to_ai_generator(self, rag):
        rag.query("question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in kwargs
        assert kwargs["tools"] == [{"name": "search_course_content"}]

    def test_passes_tool_manager_to_ai_generator(self, rag):
        rag.query("question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tool_manager" in kwargs
        assert kwargs["tool_manager"] is rag.tool_manager

    def test_prompt_wraps_original_query(self, rag):
        rag.query("what are embeddings?")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "what are embeddings?" in kwargs["query"]


# ---------------------------------------------------------------------------
# Conversation history / session management
# ---------------------------------------------------------------------------

class TestQuerySessionHandling:
    def test_fetches_history_when_session_id_provided(self, rag):
        rag.session_manager.get_conversation_history.return_value = "User: hi"

        rag.query("follow-up", session_id="sess_1")

        rag.session_manager.get_conversation_history.assert_called_once_with("sess_1")

    def test_history_passed_to_ai_generator(self, rag):
        rag.session_manager.get_conversation_history.return_value = "User: hi\nAssistant: hello"

        rag.query("follow-up", session_id="sess_1")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] == "User: hi\nAssistant: hello"

    def test_history_is_none_without_session_id(self, rag):
        rag.query("standalone question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] is None

    def test_adds_exchange_to_session_after_response(self, rag):
        rag.ai_generator.generate_response.return_value = "The answer."

        rag.query("what is RAG?", session_id="sess_2")

        rag.session_manager.add_exchange.assert_called_once_with(
            "sess_2", "what is RAG?", "The answer."
        )

    def test_does_not_update_session_without_session_id(self, rag):
        rag.query("question")

        rag.session_manager.add_exchange.assert_not_called()


# ---------------------------------------------------------------------------
# Sources lifecycle
# ---------------------------------------------------------------------------

class TestQuerySourcesLifecycle:
    def test_resets_sources_after_retrieving_them(self, rag):
        rag.query("question")

        rag.tool_manager.reset_sources.assert_called_once()

    def test_retrieves_sources_before_reset(self, rag):
        """get_last_sources must be called before reset_sources."""
        call_order = []
        rag.tool_manager.get_last_sources.side_effect = lambda: call_order.append("get") or []
        rag.tool_manager.reset_sources.side_effect = lambda: call_order.append("reset")

        rag.query("question")

        assert call_order == ["get", "reset"], (
            "get_last_sources() must be called before reset_sources() — "
            "otherwise sources are wiped before the caller can read them."
        )


# ---------------------------------------------------------------------------
# Error propagation — this is what causes "query failed" in the frontend
# ---------------------------------------------------------------------------

class TestQueryErrorPropagation:
    def test_exception_from_ai_generator_propagates(self, rag):
        """
        When AIGenerator raises (e.g. because of an invalid model ID),
        the exception must bubble up from query() so the FastAPI handler
        can return a 500 and the frontend shows 'query failed'.
        """
        rag.ai_generator.generate_response.side_effect = Exception(
            "model `claude-sonnet-4-20250514` does not exist"
        )

        with pytest.raises(Exception, match="does not exist"):
            rag.query("What is RAG?")

    def test_exception_message_is_preserved(self, rag):
        rag.ai_generator.generate_response.side_effect = Exception("API error: bad request")

        with pytest.raises(Exception) as exc_info:
            rag.query("question")

        assert "API error" in str(exc_info.value)
