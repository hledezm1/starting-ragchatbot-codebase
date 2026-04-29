import os
import sys
from unittest.mock import MagicMock

import pytest

# Make backend/ importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_QUERY = "What is retrieval-augmented generation?"
SAMPLE_ANSWER = "RAG combines retrieval with language generation for accurate answers."
SAMPLE_SOURCES = [
    {"label": "Intro to AI - Lesson 1", "url": "http://example.com/lesson1"}
]
SAMPLE_COURSE_ANALYTICS = {
    "total_courses": 2,
    "course_titles": ["Intro to AI", "Advanced RAG"],
}

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_rag():
    """Generic mock RAGSystem for unit tests that need a pre-configured stand-in."""
    m = MagicMock()
    m.query.return_value = (SAMPLE_ANSWER, SAMPLE_SOURCES)
    m.get_course_analytics.return_value = SAMPLE_COURSE_ANALYTICS
    m.session_manager.create_session.return_value = "test-session-id"
    m.session_manager.clear_session.return_value = None
    m.add_course_folder.return_value = (0, 0)
    return m
