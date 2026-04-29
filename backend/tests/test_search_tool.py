"""
Tests for CourseSearchTool.execute() and ToolManager in search_tools.py.

Strategy: unit tests use a MagicMock VectorStore so we can control every
return value precisely.  Integration tests (marked with @pytest.mark.integration)
hit the real ChromaDB that was seeded at server startup.
"""

from unittest.mock import MagicMock

import pytest
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


def _make_results(docs, metas, distances=None):
    """Helper: build a SearchResults with sensible defaults."""
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances or [0.5] * len(docs),
    )


# ---------------------------------------------------------------------------
# CourseSearchTool.execute() – argument forwarding
# ---------------------------------------------------------------------------


class TestExecuteForwardsArguments:
    def test_forwards_query_only(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool.execute(query="what is RAG?")

        mock_store.search.assert_called_once_with(
            query="what is RAG?",
            course_name=None,
            lesson_number=None,
        )

    def test_forwards_course_name_filter(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool.execute(query="embeddings", course_name="Advanced Retrieval")

        mock_store.search.assert_called_once_with(
            query="embeddings",
            course_name="Advanced Retrieval",
            lesson_number=None,
        )

    def test_forwards_lesson_number_filter(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool.execute(query="embeddings", lesson_number=3)

        mock_store.search.assert_called_once_with(
            query="embeddings",
            course_name=None,
            lesson_number=3,
        )

    def test_forwards_both_filters(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool.execute(query="embeddings", course_name="MCP", lesson_number=2)

        mock_store.search.assert_called_once_with(
            query="embeddings",
            course_name="MCP",
            lesson_number=2,
        )


# ---------------------------------------------------------------------------
# CourseSearchTool.execute() – return-value formatting
# ---------------------------------------------------------------------------


class TestExecuteReturnValues:
    def test_returns_formatted_course_and_lesson_header(self, tool, mock_store):
        mock_store.search.return_value = _make_results(
            docs=["Lesson text here"],
            metas=[{"course_title": "MCP Course", "lesson_number": 2}],
        )
        mock_store.get_lesson_link.return_value = "http://example.com/lesson2"

        result = tool.execute(query="MCP tools")

        assert "[MCP Course - Lesson 2]" in result
        assert "Lesson text here" in result

    def test_header_omits_lesson_when_lesson_number_is_none(self, tool, mock_store):
        mock_store.search.return_value = _make_results(
            docs=["General content"],
            metas=[{"course_title": "RAG Course", "lesson_number": None}],
        )

        result = tool.execute(query="overview")

        assert "[RAG Course]" in result
        assert "Lesson" not in result

    def test_returns_no_content_message_when_empty(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_no_content_message_includes_course_filter(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool.execute(query="topic", course_name="GhostCourse")

        assert "GhostCourse" in result

    def test_no_content_message_includes_lesson_filter(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool.execute(query="topic", lesson_number=99)

        assert "99" in result

    def test_returns_error_string_on_search_failure(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: connection failed",
        )

        result = tool.execute(query="anything")

        assert "Search error" in result

    def test_multiple_results_are_separated(self, tool, mock_store):
        mock_store.search.return_value = _make_results(
            docs=["First chunk", "Second chunk"],
            metas=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
            ],
        )
        mock_store.get_lesson_link.return_value = None

        result = tool.execute(query="topic")

        assert "First chunk" in result
        assert "Second chunk" in result


# ---------------------------------------------------------------------------
# CourseSearchTool – sources tracking
# ---------------------------------------------------------------------------


class TestSourcesTracking:
    def test_populates_last_sources_after_successful_search(self, tool, mock_store):
        mock_store.search.return_value = _make_results(
            docs=["content"],
            metas=[{"course_title": "RAG Course", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "http://example.com/l1"

        tool.execute(query="test")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "RAG Course - Lesson 1"
        assert tool.last_sources[0]["url"] == "http://example.com/l1"

    def test_source_label_has_no_lesson_suffix_when_lesson_number_is_none(
        self, tool, mock_store
    ):
        mock_store.search.return_value = _make_results(
            docs=["content"],
            metas=[{"course_title": "RAG Course", "lesson_number": None}],
        )

        tool.execute(query="test")

        assert tool.last_sources[0]["label"] == "RAG Course"

    def test_last_sources_empty_before_first_execute(self, mock_store):
        fresh_tool = CourseSearchTool(mock_store)
        assert fresh_tool.last_sources == []

    def test_last_sources_empty_when_no_results(self, tool, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool.execute(query="nothing")

        assert tool.last_sources == []


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------


class TestToolManager:
    def test_register_and_execute_tool(self):
        manager = ToolManager()
        mock_store = MagicMock()
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="hello")

        assert "No relevant content found" in result

    def test_execute_unknown_tool_returns_error_string(self):
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="hello")

        assert "not found" in result.lower()

    def test_get_last_sources_returns_sources_from_search_tool(self):
        manager = ToolManager()
        mock_store = MagicMock()
        tool = CourseSearchTool(mock_store)
        tool.last_sources = [{"label": "Test Course", "url": "http://example.com"}]
        manager.register_tool(tool)

        sources = manager.get_last_sources()

        assert sources == [{"label": "Test Course", "url": "http://example.com"}]

    def test_get_last_sources_returns_empty_when_no_sources(self):
        manager = ToolManager()
        mock_store = MagicMock()
        manager.register_tool(CourseSearchTool(mock_store))

        assert manager.get_last_sources() == []

    def test_reset_sources_clears_all_tools(self):
        manager = ToolManager()
        mock_store = MagicMock()
        tool = CourseSearchTool(mock_store)
        tool.last_sources = [{"label": "X", "url": None}]
        manager.register_tool(tool)

        manager.reset_sources()

        assert tool.last_sources == []

    def test_get_tool_definitions_returns_list(self):
        manager = ToolManager()
        mock_store = MagicMock()
        manager.register_tool(CourseSearchTool(mock_store))

        defs = manager.get_tool_definitions()

        assert isinstance(defs, list)
        assert defs[0]["name"] == "search_course_content"
