from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client, create_client
from openai import AsyncOpenAI


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


def inject_privacy_popup():
    if "privacy_popup_loaded" not in st.session_state:
        st.session_state.privacy_popup_loaded = True

        st.components.v1.html("""
        <style>
                    .privacy-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.45);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 99999;
            }

            .privacy-modal {
            width: 90%;
            max-width: 520px;
            background: #fff;
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
            text-align: center;
            }

            .privacy-modal h3 {
            margin: 0 0 12px 0;
            }

            .privacy-text {
            margin: 0 0 16px 0;
            color: #444;
            line-height: 1.5;
            }

            .privacy-link-wrap {
            margin: 0 0 20px 0;
            }

            .privacy-link {
            display: inline-block;
            padding: 10px 16px;
            border-radius: 8px;
            text-decoration: none;
            background: #f5f5f5;
            border: 1px solid #ddd;
            color: #222;
            font-weight: 600;
            }

            .privacy-link:hover {
            background: #ececec;
            }

            .privacy-actions {
            display: flex;
            justify-content: center;
            gap: 12px;
            }

            .privacy-accept-btn,
            .privacy-decline-btn {
            padding: 10px 22px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 15px;
            }
        </style>

        <div class="privacy-overlay" id="privacy-popup">
        <div class="privacy-modal">
            <h3>Privacy Statement</h3>

            <p class="privacy-text">
            Please read our privacy statement before continuing.
            </p>

            <p class="privacy-link-wrap">
                <a
                    href="https://search-envri.qcdis.org/search/static/privacy/privacy_statement.pdf"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="privacy-link">
                    Open Privacy Statement (PDF)
                </a>
            </p>

            <div class="privacy-actions">
            <button type="button" id="accept-privacy-btn" class="privacy-accept-btn">Yes</button>
            <button type="button" id="decline-privacy-btn" class="privacy-decline-btn">No</button>
            </div>
        </div>
        </div>
<script>
  document.addEventListener("DOMContentLoaded", function () {
    const popup = document.getElementById("privacy-popup");
    const acceptBtn = document.getElementById("accept-privacy-btn");
    const declineBtn = document.getElementById("decline-privacy-btn");

    if (localStorage.getItem("privacyAccepted") === "true") {
      popup.style.display = "none";
      document.body.style.overflow = "";
      return;
    }

    popup.style.display = "flex";
    document.body.style.overflow = "hidden";

    acceptBtn.addEventListener("click", function () {
      popup.style.display = "none";
      document.body.style.overflow = "";
      localStorage.setItem("privacyAccepted", "true");
    });

    declineBtn.addEventListener("click", function () {
      popup.style.display = "none";
      document.body.style.overflow = "";
      localStorage.setItem("privacyAccepted", "true");
    });
  });
</script>

<script>
  declineBtn.addEventListener("click", function () {
    window.close();
    setTimeout(function () {
      window.location.href = "https://search-envri.qcdis.org/search";
    }, 200);
  });
</script>
                              
        
        </script>
        """, height=700, width=900)

def inject_matomo():
    if "matomo_loaded" not in st.session_state:
        st.session_state.matomo_loaded = True

        st.components.v1.html("""
        <!-- Matomo -->
            <script>
            var _paq = window._paq = window._paq || [];
            /* tracker methods like "setCustomDimension" should be called before "trackPageView" */
            _paq.push(['trackPageView']);
            _paq.push(['enableLinkTracking']);
            (function() {
                var u="//analytics.envri.eu/";
                _paq.push(['setTrackerUrl', u+'matomo.php']);
                _paq.push(['setSiteId', '6']);
                var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
                g.async=true; g.src=u+'matomo.js'; s.parentNode.insertBefore(g,s);
            })();
            </script>
            <!-- End Matomo Code -->
        """,
            height=0,
            width=0,)

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



async def handle_user_query(user_input: str):
    """
    Handles both:
    1. manually typed chat input
    2. query forwarded from another search site
    """

    st.session_state.messages.append(
        ModelRequest(parts=[UserPromptPart(content=user_input)])
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        await run_agent_with_streaming(user_input)


async def handle_user_input(user_input: str):
    st.session_state.messages.append(
        ModelRequest(parts=[UserPromptPart(content=user_input)])
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        await run_agent_with_streaming(user_input)


async def main():

    st.title("Environmental Expert...")
    st.write("Ask any question about the Environment and Earth Science")

    inject_matomo()
    # inject_privacy_popup()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "handled_forwarded_queries" not in st.session_state:
        st.session_state.handled_forwarded_queries = set()

    forwarded_query = st.query_params.get("q", "").strip()

    print("Printing forwarded query:", forwarded_query)

    # Display previous conversation
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Case 1: query came from external website
    if forwarded_query:
        if forwarded_query not in st.session_state.handled_forwarded_queries:
            print("Handling forwarded query:", forwarded_query)

            st.session_state.handled_forwarded_queries.add(forwarded_query)

            await handle_user_input(forwarded_query)

            # Optional but recommended:
            # remove q from URL so refresh does not resubmit
            st.query_params.clear()
            st.rerun()

    # Case 2: normal Streamlit chat input
    user_input = st.chat_input("What questions do you have about Environment?")

    if user_input:
        await handle_user_input(user_input)

# Original main function
# async def main():

#     st.title("Environmental Expert...")
#     st.write("Ask any question about the Environment and Earth Science")

#     inject_matomo()
#     #inject_privacy_popup()

    

#     # Initialize chat history in session state if not present
#     if "messages" not in st.session_state:
#         st.session_state.messages = []


#     forwarded_query = st.query_params.get("q", "").strip()
#     print("Printing forwarded query", forwarded_query)

#     # Display all messages from the conversation so far
#     # Each message is either a ModelRequest or ModelResponse.
#     # We iterate over their parts to decide how to display them.
#     for msg in st.session_state.messages:
#         if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
#             for part in msg.parts:
#                 display_message_part(part)

#     # Chat input for the user
#     forwarded = False
#     if forwarded_query != "":
#         forwarded = True
#         #user_input = st.chat_input(forwarded_query)
#         print("You are here ... inside forwarded query")
#         st.session_state.chat_input = forwarded_query
#         #st.chat_input(key="chat_input")
#         user_input = forwarded_query

#     else:
#         user_input = st.chat_input("What questions do you have about Environment?")

# # st.session_state.chat_input = "Hello, world!"
# # st.chat_input(key="chat_input")

#     if user_input:
#         # We append a new request to the conversation explicitly
#         st.session_state.messages.append(
#             ModelRequest(parts=[UserPromptPart(content=user_input)])
#         )

#         # Display user prompt in the UI
#         with st.chat_message("user"):
#             st.markdown(user_input)


#         # Display the assistant's partial response while streaming
#         with st.chat_message("assistant"):
#             # Actually run the agent now, streaming the text
#             await run_agent_with_streaming(user_input)

#         if forwarded:
#             user_input = st.chat_input("What questions do you have about Environment?")
#             forwarded = False



if __name__ == "__main__":
    asyncio.run(main())

# https://chat-envri.qcdis.org/?q=air%20pollution%20in%20Europe