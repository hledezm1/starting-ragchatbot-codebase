"""
Tests for the FastAPI application endpoints in app.py.

app.py mounts StaticFiles at module load time and initialises RAGSystem
(ChromaDB) at import time.  Both are patched before the import so tests run
without a real filesystem or database.
"""

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Stub that replaces StaticFiles to avoid "directory does not exist" errors.
# DevStaticFiles in app.py subclasses StaticFiles, so the stub must be a
# real class (not a MagicMock instance).
# ---------------------------------------------------------------------------


class _StubStaticFiles:
    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, scope, receive, send):
        pass


# ---------------------------------------------------------------------------
# Module-level mock wired into the app before import.
# Using a single instance allows per-test configuration via the `rag` fixture.
# ---------------------------------------------------------------------------

_DEFAULT_ANSWER = "Test answer about RAG."
_DEFAULT_COURSES = {"total_courses": 2, "course_titles": ["Course A", "Course B"]}

_mock_rag = MagicMock()
_mock_rag.query.return_value = (_DEFAULT_ANSWER, [])
_mock_rag.get_course_analytics.return_value = _DEFAULT_COURSES
_mock_rag.session_manager.create_session.return_value = "test-session-id"
_mock_rag.session_manager.clear_session.return_value = None
_mock_rag.add_course_folder.return_value = (0, 0)

with (
    patch("rag_system.RAGSystem", return_value=_mock_rag),
    patch("fastapi.staticfiles.StaticFiles", _StubStaticFiles),
):
    import app as _app_module

    _app_module.rag_system = _mock_rag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(_app_module.app)


@pytest.fixture(autouse=True)
def _reset():
    """Reset all mock state before each test so tests are fully independent."""
    _mock_rag.reset_mock()
    _mock_rag.query.return_value = (_DEFAULT_ANSWER, [])
    _mock_rag.query.side_effect = None
    _mock_rag.get_course_analytics.return_value = _DEFAULT_COURSES.copy()
    _mock_rag.get_course_analytics.side_effect = None
    _mock_rag.session_manager.create_session.return_value = "test-session-id"
    _mock_rag.session_manager.clear_session.return_value = None
    _mock_rag.add_course_folder.return_value = (0, 0)


@pytest.fixture
def rag():
    """Return the shared mock RAGSystem for per-test customisation."""
    return _mock_rag


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_returns_200_with_valid_query(self, client):
        resp = client.post("/api/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/api/query", json={"query": "What is RAG?"})
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_answer_comes_from_rag_system(self, client, rag):
        rag.query.return_value = ("Custom answer text.", [])
        resp = client.post("/api/query", json={"query": "anything"})
        assert resp.json()["answer"] == "Custom answer text."

    def test_sources_serialised_in_response(self, client, rag):
        rag.query.return_value = (
            "Answer",
            [{"label": "Course A - Lesson 1", "url": "http://example.com"}],
        )
        resp = client.post("/api/query", json={"query": "test"})
        sources = resp.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["label"] == "Course A - Lesson 1"
        assert sources[0]["url"] == "http://example.com"

    def test_generates_session_id_when_none_provided(self, client):
        resp = client.post("/api/query", json={"query": "test"})
        assert resp.json()["session_id"] == "test-session-id"

    def test_echoes_provided_session_id(self, client):
        resp = client.post(
            "/api/query", json={"query": "test", "session_id": "my-session"}
        )
        assert resp.json()["session_id"] == "my-session"

    def test_returns_500_when_rag_system_raises(self, client, rag):
        rag.query.side_effect = Exception("internal failure")
        resp = client.post("/api/query", json={"query": "bad query"})
        assert resp.status_code == 500

    def test_error_detail_present_on_500(self, client, rag):
        rag.query.side_effect = Exception("internal failure")
        resp = client.post("/api/query", json={"query": "bad query"})
        assert "internal failure" in resp.json()["detail"]

    def test_returns_422_for_missing_query_field(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------


class TestCoursesEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/courses")
        assert resp.status_code == 200

    def test_response_has_total_courses(self, client):
        resp = client.get("/api/courses")
        assert "total_courses" in resp.json()

    def test_response_has_course_titles(self, client):
        resp = client.get("/api/courses")
        assert "course_titles" in resp.json()

    def test_total_courses_reflects_analytics(self, client, rag):
        rag.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        resp = client.get("/api/courses")
        assert resp.json()["total_courses"] == 5

    def test_course_titles_list_in_response(self, client, rag):
        rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Intro to RAG"],
        }
        resp = client.get("/api/courses")
        assert "Intro to RAG" in resp.json()["course_titles"]

    def test_returns_500_when_analytics_raises(self, client, rag):
        rag.get_course_analytics.side_effect = Exception("db error")
        resp = client.get("/api/courses")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /api/session/new
# ---------------------------------------------------------------------------


class TestNewSessionEndpoint:
    def test_returns_200(self, client):
        resp = client.post("/api/session/new", json={})
        assert resp.status_code == 200

    def test_response_has_session_id(self, client):
        resp = client.post("/api/session/new", json={})
        assert "session_id" in resp.json()

    def test_returns_new_session_id(self, client):
        resp = client.post("/api/session/new", json={})
        assert resp.json()["session_id"] == "test-session-id"

    def test_clears_old_session_when_provided(self, client, rag):
        client.post("/api/session/new", json={"session_id": "old-session"})
        rag.session_manager.clear_session.assert_called_once_with("old-session")

    def test_does_not_clear_session_when_none_provided(self, client, rag):
        client.post("/api/session/new", json={})
        rag.session_manager.clear_session.assert_not_called()

    def test_always_creates_new_session(self, client, rag):
        client.post("/api/session/new", json={})
        rag.session_manager.create_session.assert_called_once()
