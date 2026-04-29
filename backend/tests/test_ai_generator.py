"""
Tests for AIGenerator in ai_generator.py.

All tests mock the Anthropic client — no real API calls are made.
The model-ID validity tests compare config.py's model string against the
known-valid Claude 4 IDs; a failure here is the most likely root cause of
"query failed" in production.
"""

from unittest.mock import MagicMock, patch

import pytest
from ai_generator import AIGenerator
from config import config as app_config

# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------


def _text_response(text="Answer text"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _tool_use_response(
    tool_name="search_course_content", tool_input=None, tool_id="tool_abc123"
):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input or {"query": "what is RAG?"}
    block.id = tool_id
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gen():
    """AIGenerator with a fully mocked Anthropic client."""
    with patch("anthropic.Anthropic"):
        g = AIGenerator(api_key="test_key", model="claude-sonnet-4-6")
    return g


# ---------------------------------------------------------------------------
# Model ID validity — most likely root cause of "query failed"
# ---------------------------------------------------------------------------

KNOWN_VALID_MODELS = {
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
}


class TestModelIDValidity:
    def test_configured_model_is_known_valid(self):
        """
        Fails when config.py has an invalid model string such as
        'claude-sonnet-4-20250514'.  Claude 4 model IDs use the format
        'claude-{family}-{major}-{minor}' — no date suffix.
        """
        assert app_config.ANTHROPIC_MODEL in KNOWN_VALID_MODELS, (
            f"config.ANTHROPIC_MODEL='{app_config.ANTHROPIC_MODEL}' is not a "
            f"recognised Claude 4 model ID.\n"
            f"Known valid IDs: {sorted(KNOWN_VALID_MODELS)}\n"
            "Claude 4 models do NOT use date suffixes (unlike Claude 3.x)."
        )

    def test_model_id_does_not_use_date_suffix(self):
        """
        Claude 3.5 used IDs like 'claude-3-5-sonnet-20241022'.
        Claude 4 dropped the date: 'claude-sonnet-4-6'.
        A date-suffixed Claude 4 ID is invalid and causes every API call to fail.
        """
        last_segment = app_config.ANTHROPIC_MODEL.split("-")[-1]
        is_8_digit_date = len(last_segment) == 8 and last_segment.isdigit()

        assert not is_8_digit_date, (
            f"config.ANTHROPIC_MODEL='{app_config.ANTHROPIC_MODEL}' ends with an "
            "8-digit date string — this format is only valid for Claude 3.x models. "
            "Use a Claude 4 ID like 'claude-sonnet-4-6'."
        )


# ---------------------------------------------------------------------------
# Direct (no-tool) response path
# ---------------------------------------------------------------------------


class TestDirectResponsePath:
    def test_returns_text_from_response(self, gen):
        gen.client.messages.create.return_value = _text_response("Paris.")

        result = gen.generate_response("Capital of France?")

        assert result == "Paris."

    def test_api_called_with_user_message(self, gen):
        gen.client.messages.create.return_value = _text_response()

        gen.generate_response("My question")

        kwargs = gen.client.messages.create.call_args[1]
        assert kwargs["messages"] == [{"role": "user", "content": "My question"}]

    def test_system_prompt_included(self, gen):
        gen.client.messages.create.return_value = _text_response()

        gen.generate_response("question")

        kwargs = gen.client.messages.create.call_args[1]
        assert "system" in kwargs
        assert len(kwargs["system"]) > 0

    def test_conversation_history_appended_to_system(self, gen):
        gen.client.messages.create.return_value = _text_response()

        gen.generate_response(
            "follow-up", conversation_history="User: hi\nAssistant: hello"
        )

        kwargs = gen.client.messages.create.call_args[1]
        assert "Previous conversation" in kwargs["system"]
        assert "User: hi" in kwargs["system"]

    def test_no_tools_key_when_tools_not_provided(self, gen):
        gen.client.messages.create.return_value = _text_response()

        gen.generate_response("question", tools=None)

        kwargs = gen.client.messages.create.call_args[1]
        assert "tools" not in kwargs

    def test_tools_passed_to_api_when_provided(self, gen):
        gen.client.messages.create.return_value = _text_response()
        tool_defs = [
            {"name": "search_course_content", "description": "...", "input_schema": {}}
        ]

        gen.generate_response("question", tools=tool_defs)

        kwargs = gen.client.messages.create.call_args[1]
        assert kwargs["tools"] == tool_defs
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_does_not_call_tool_manager_on_end_turn(self, gen):
        gen.client.messages.create.return_value = _text_response()
        mock_tm = MagicMock()

        gen.generate_response("question", tool_manager=mock_tm)

        mock_tm.execute_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Tool-use execution path
# ---------------------------------------------------------------------------


class TestToolUsePath:
    def test_executes_tool_when_stop_reason_is_tool_use(self, gen):
        gen.client.messages.create.side_effect = [
            _tool_use_response(tool_input={"query": "what is RAG?"}),
            _text_response("RAG stands for Retrieval Augmented Generation."),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "RAG search result text"

        result = gen.generate_response(
            "What is RAG?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        mock_tm.execute_tool.assert_called_once_with(
            "search_course_content", query="what is RAG?"
        )
        assert result == "RAG stands for Retrieval Augmented Generation."

    def test_tool_result_content_present_in_second_api_call(self, gen):
        gen.client.messages.create.side_effect = [
            _tool_use_response(tool_id="call_xyz", tool_input={"query": "embeddings"}),
            _text_response("Embeddings answer"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Embeddings are dense vectors..."

        gen.generate_response(
            "explain embeddings",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_xyz"
        assert (
            "Embeddings are dense vectors" in tool_result_msg["content"][0]["content"]
        )

    def test_assistant_tool_use_block_in_second_call_messages(self, gen):
        """The assistant's tool-use content block must appear in the second call's history."""
        first_resp = _tool_use_response(tool_input={"query": "MCP"})
        gen.client.messages.create.side_effect = [first_resp, _text_response()]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        gen.generate_response("question", tools=[{}], tool_manager=mock_tm)

        second_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        assistant_msg = second_msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == first_resp.content

    def test_raises_when_api_call_fails(self, gen):
        """An API error (e.g. invalid model) must propagate — not be silently swallowed."""
        gen.client.messages.create.side_effect = Exception(
            "model `claude-sonnet-4-20250514` does not exist"
        )

        with pytest.raises(Exception, match="does not exist"):
            gen.generate_response("anything")

    # -----------------------------------------------------------------------
    # Sequential tool-calling (up to MAX_TOOL_ROUNDS rounds)
    # -----------------------------------------------------------------------

    def test_single_round_makes_two_api_calls(self, gen):
        """1 tool_use round followed by end_turn → 2 total API calls."""
        gen.client.messages.create.side_effect = [
            _tool_use_response(),
            _text_response("Answer"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        result = gen.generate_response("q", tools=[{}], tool_manager=mock_tm)

        assert gen.client.messages.create.call_count == 2
        assert result == "Answer"

    def test_two_round_tool_use_makes_three_api_calls(self, gen):
        """2 tool_use rounds → 3 total API calls (2 rounds + final synthesis)."""
        gen.client.messages.create.side_effect = [
            _tool_use_response(tool_id="t1"),
            _tool_use_response(tool_id="t2"),
            _text_response("Final answer"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        result = gen.generate_response("q", tools=[{}], tool_manager=mock_tm)

        assert gen.client.messages.create.call_count == 3
        assert result == "Final answer"

    def test_two_rounds_both_tools_executed(self, gen):
        """Both tools in a 2-round sequence are each executed once."""
        gen.client.messages.create.side_effect = [
            _tool_use_response(
                tool_name="search_course_content",
                tool_id="t1",
                tool_input={"query": "X"},
            ),
            _tool_use_response(
                tool_name="get_course_outline",
                tool_id="t2",
                tool_input={"course_name": "Y"},
            ),
            _text_response("Done"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        gen.generate_response("q", tools=[{}], tool_manager=mock_tm)

        assert mock_tm.execute_tool.call_count == 2
        mock_tm.execute_tool.assert_any_call("search_course_content", query="X")
        mock_tm.execute_tool.assert_any_call("get_course_outline", course_name="Y")

    def test_final_call_has_no_tools(self, gen):
        """The final synthesis call after all tool rounds must not include the tools key."""
        gen.client.messages.create.side_effect = [
            _tool_use_response(tool_id="t1"),
            _tool_use_response(tool_id="t2"),
            _text_response("Final answer"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        gen.generate_response(
            "q", tools=[{"name": "search_course_content"}], tool_manager=mock_tm
        )

        final_kwargs = gen.client.messages.create.call_args_list[2][1]
        assert "tools" not in final_kwargs

    def test_tools_present_in_tool_rounds(self, gen):
        """API calls during tool rounds include the tools parameter."""
        tool_defs = [{"name": "search_course_content"}]
        gen.client.messages.create.side_effect = [
            _tool_use_response(tool_id="t1"),
            _tool_use_response(tool_id="t2"),
            _text_response("Final"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        gen.generate_response("q", tools=tool_defs, tool_manager=mock_tm)

        assert "tools" in gen.client.messages.create.call_args_list[0][1]
        assert "tools" in gen.client.messages.create.call_args_list[1][1]

    def test_tool_failure_still_makes_final_synthesis_call(self, gen):
        """Tool raises exception → loop breaks but final synthesis call is still made."""
        gen.client.messages.create.side_effect = [
            _tool_use_response(),
            _text_response("Synthesis despite error"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = Exception("search failed")

        result = gen.generate_response("q", tools=[{}], tool_manager=mock_tm)

        assert gen.client.messages.create.call_count == 2
        assert result == "Synthesis despite error"

    def test_context_preserved_across_rounds(self, gen):
        """Messages in round 2 include the assistant tool-use block and tool result from round 1."""
        first_resp = _tool_use_response(tool_id="t1", tool_input={"query": "first"})
        gen.client.messages.create.side_effect = [
            first_resp,
            _tool_use_response(tool_id="t2", tool_input={"query": "second"}),
            _text_response("Final"),
        ]
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "round one result"

        gen.generate_response("q", tools=[{}], tool_manager=mock_tm)

        round2_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        roles = [m["role"] for m in round2_msgs]
        assert roles == ["user", "assistant", "user"]
        assert round2_msgs[1]["content"] == first_resp.content
        tool_result_content = round2_msgs[2]["content"]
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "t1"
