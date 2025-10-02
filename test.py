from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

import json
import logfire
from supabase import Client, create_client
from openai import AsyncOpenAI
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
db_password = os.getenv("DB_PASSWORD")
supabase_url = os.getenv("SUPABASE_URL")
supabase_secret = os.getenv("SUPABASE_SECRET")

openai_client = AsyncOpenAI(api_key=openai_api_key)
supabase: Client = create_client(
    supabase_url,
    supabase_secret
)

import asyncio
from typing import Any, List, Optional

# Assumes `pydantic_ai_expert` is your Agent instance (e.g., PydanticAI Agent)
# and that it supports `run_stream(...)` returning an object with `stream_text(...)`

async def _run_agent_once(
    agent: Any,
    user_input: str,
    deps: Optional[dict] = None,
    message_history: Optional[List[Any]] = None,
) -> str:
    """Run the agent once and return the full response text."""
    deps = deps or {}
    message_history = message_history or []

    async with agent.run_stream(
        user_input,
        deps=deps,
        message_history=message_history,
    ) as result:
        chunks = []
        async for chunk in result.stream_text(delta=True):
            chunks.append(chunk)
        return "".join(chunks)

def run_and_print_response(
    agent: Any,
    user_input: str,
    deps: Optional[dict] = None,
    message_history: Optional[List[Any]] = None,
) -> str:
    """
    Synchronous wrapper: runs the agent and prints the full response text.
    Returns the text as well.
    """
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )
    full_text = asyncio.run(_run_agent_once(agent, user_input, deps, message_history))
    print(full_text)
    return full_text


# --- Example usage ---
full_text = run_and_print_response(pydantic_ai_expert, "Tell me about seadatanet dataset.")
