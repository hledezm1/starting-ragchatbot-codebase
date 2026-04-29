import anthropic
from typing import List, Optional, Dict, Any

MAX_TOOL_ROUNDS = 2

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- Use the outline tool for questions about a course's structure, syllabus, lesson list, or what topics are covered
- Use up to two sequential tool calls when a query requires chaining results (e.g., look up a lesson title, then search for related content)
- Each tool call should build on previous results when applicable
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific content questions**: Use the search tool first, then answer
- **Course outline / syllabus questions**: Use the outline tool and return the course title, course link, and every lesson with its number and title
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional sequential tool usage (up to MAX_TOOL_ROUNDS).

        Terminates when:
          (a) MAX_TOOL_ROUNDS rounds completed — final synthesis call made without tools
          (b) Claude returns no tool_use blocks — that response is returned directly
          (c) A tool call raises an exception — loop breaks, final synthesis call made
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        current_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        if tools:
            current_params["tools"] = tools
            current_params["tool_choice"] = {"type": "auto"}

        # Without a tool_manager there is nothing to execute; make a single call
        if not tool_manager:
            response = self.client.messages.create(**current_params)
            return response.content[0].text

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.messages.create(**current_params)

            # Termination (b): Claude did not request a tool
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_use_blocks:
                return response.content[0].text

            # Execute tools; stop executing further tools on first failure
            tool_results = []
            tool_failed = False
            for block in tool_use_blocks:
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    result = f"Tool execution error: {e}"
                    tool_failed = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
                if tool_failed:
                    break

            # Preserve context: assistant turn + tool results
            current_params["messages"] = current_params["messages"] + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

            if tool_failed:
                break  # fall through to final synthesis (c)

        # Termination (a)/(c): final synthesis without tools
        final_params = {k: v for k, v in current_params.items() if k not in ("tools", "tool_choice")}
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
