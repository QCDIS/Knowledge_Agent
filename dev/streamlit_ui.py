from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client, create_client
from openai import AsyncOpenAI

####### HF Transformers ######
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from dataclasses import dataclass
# from pydantic_ai import Agent

# # Load a Hugging Face model
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# # Create a text generation pipeline
# hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


# ####### HF Transformers ######



# Import all the message part classes
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
from pydantic_ai_expert_dev import pydantic_ai_expert, PydanticAIDeps

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


# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)


# async def run_agent_with_streaming(user_input: str):
#     """
#     Run the agent with streaming text for the user_input prompt,
#     while maintaining the entire conversation in `st.session_state.messages`.
#     """

#     print("Printing user_input", user_input)
#     # Prepare dependencies
#     deps = PydanticAIDeps(
#         supabase=supabase,
#         openai_client=openai_client
#     )

#     # deps = PydanticAIDeps(
#     #     supabase=supabase,
#     #     hf_pipeline=hf_pipeline  # Replacing OpenAI client with Hugging Face
#     # )


#     # Run the agent in a stream
#     async with pydantic_ai_expert.run_stream(
#         user_input,
#         deps=deps,
#         message_history= st.session_state.messages[:-1],  # pass entire conversation so far
#     ) as result:
#         # We'll gather partial text to show incrementally
#         partial_text = ""
#         message_placeholder = st.empty()

#         # Render partial text as it arrives
#         async for chunk in result.stream_text(delta=True):
#             partial_text += chunk
#             message_placeholder.markdown(partial_text)

#         # Now that the stream is finished, we have a final result.
#         # Add new messages from this run, excluding user-prompt messages
#         filtered_messages = [msg for msg in result.new_messages()
#                             if not (hasattr(msg, 'parts') and
#                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]
#         st.session_state.messages.extend(filtered_messages)

#         # Add the final response to the messages
#         st.session_state.messages.append(
#             ModelResponse(parts=[TextPart(content=partial_text)])
#         )

async def run_agent_with_streaming(user_input: str):
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # history up to (but not incl.) this turn's user msg
    ) as result:
        # Live stream to UI (not persisted)
        partial_text = ""
        message_placeholder = st.empty()
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # --- Build a single persisted assistant message ---
        # 1) Gather messages created in this run (tool/system included).
        run_msgs = result.new_messages()

        # 2) Keep only non-user messages
        run_msgs = [
            m for m in run_msgs
            if not (hasattr(m, "parts") and any(p.part_kind == "user-prompt" for p in m.parts))
        ]

        # 3) From those, extract *assistant* responses (ModelResponse with text parts)
        def is_assistant_response(m):
            return isinstance(m, ModelResponse) and any(
                getattr(p, "part_kind", "") == "text" for p in getattr(m, "parts", [])
            )

        assistant_msgs = [m for m in run_msgs if is_assistant_response(m)]

        # 4) Choose exactly one assistant message to persist:
        chosen_assistant = None
        if assistant_msgs:
            # keep the last (final) assistant response from the agent
            chosen_assistant = assistant_msgs[-1]
        elif partial_text.strip():
            # fallback: create one from the streamed text
            chosen_assistant = ModelResponse(parts=[TextPart(content=partial_text)])

        # 5) Persist: first any non-assistant messages you *do* want to keep (optional)
        non_assistant = [m for m in run_msgs if not is_assistant_response(m)]
        # If you don't want to keep system/tool messages in history, set non_assistant = []

        # 6) De-dup guard: avoid storing same assistant text twice
        def msg_text(m):
            if hasattr(m, "parts"):
                return "\n".join(
                    getattr(p, "content", "")
                    for p in m.parts
                    if getattr(p, "part_kind", "") == "text"
                ).strip()
            return ""

        if chosen_assistant:
            new_text = msg_text(chosen_assistant)
            # If last stored message is the same assistant text, skip
            if st.session_state.messages:
                last = st.session_state.messages[-1]
                if isinstance(last, ModelResponse) and msg_text(last) == new_text:
                    chosen_assistant = None  # skip adding duplicate

        # 7) Finally extend session_state with exactly one assistant message
        st.session_state.messages.extend(non_assistant)
        if chosen_assistant:
            st.session_state.messages.append(chosen_assistant)

async def main():
    st.title("Environmental Expert...")
    st.write("Ask any question about the Environment and Earth Science")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about Environment?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())