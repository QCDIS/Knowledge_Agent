from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import json
import os
import sys
import textwrap
import time

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
#from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client, create_client
from typing import Any, List, Optional
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIResponsesModel

from mcp import ClientSession

# Reuse the transport helpers from mcp_client.py (same directory). The second form
# covers running from the parent folder, e.g. `python -m dev.<this_file>` or a UI there.
try:
    from mcp_client import connect, attr, DEFAULT_URL as MCP_DEFAULT_URL
except ModuleNotFoundError:
    from dev.mcp_client import connect, attr, DEFAULT_URL as MCP_DEFAULT_URL

load_dotenv()

# ---------------------------------------------------------------------------
# Knowledge-base backend selection
#   KB_BACKEND=vector -> only the Supabase vector database (original behaviour)
#   KB_BACKEND=mcp    -> only the Ifremer MCP server (Argo floats + OSO ontology)
#   KB_BACKEND=auto   -> both; the agent picks per question, and falls back to the
#                        vector database when the MCP server is unreachable
# Can be overridden per request via PydanticAIDeps(backend=...).
# ---------------------------------------------------------------------------
KB_BACKENDS = ("vector", "mcp", "auto")
KB_BACKEND = os.getenv("KB_BACKEND", "auto").lower()
if KB_BACKEND not in KB_BACKENDS:
    raise ValueError(f"KB_BACKEND must be one of {KB_BACKENDS}, got {KB_BACKEND!r}")

MCP_URL = os.getenv("MCP_URL", MCP_DEFAULT_URL)
MCP_TOKEN = os.getenv("KB_TOKEN")          # one of the server's KB_READ_TOKENS
MCP_MAX_CHARS = int(os.getenv("MCP_MAX_CHARS", "15000"))  # cap on text returned to the LLM

# Debug printing of the combined context (auto mode):
#   KB_DEBUG=true            -> print the content from BOTH sources in the terminal
#   KB_DEBUG_MAX_CHARS=600   -> max characters per printed field/chunk (0 = no limit)
KB_DEBUG = os.getenv("KB_DEBUG", "false").strip().lower() in ("1", "true", "yes", "on")
KB_DEBUG_MAX_CHARS = int(os.getenv("KB_DEBUG_MAX_CHARS", "600"))

SOURCE_MCP = "MCP (Ifremer knowledge base)"
SOURCE_VECTOR = "Knowledge base (Supabase vector DB)"


def log_fetch(source: str, detail: str = "") -> None:
    """Print, at the moment a fetch starts, which source is being queried."""
    print(f"[Fetching from] {source}" + (f" | {detail}" if detail else ""), flush=True)


def log_source(source: str, detail: str = "") -> None:
    """Print which source a tool call used (after the fetch returns)."""
    print(f"[Response source] {source}" + (f" | {detail}" if detail else ""), flush=True)


openai_api_key = os.getenv("OPENAI_API_KEY")
db_password = os.getenv("DB_PASSWORD")
supabase_url = os.getenv("SUPABASE_URL")
supabase_secret = os.getenv("SUPABASE_SECRET")

openai_client = AsyncOpenAI(api_key=openai_api_key)
supabase: Client = create_client(
    supabase_url,
    supabase_secret
)

LLM_PROVIDERS = {
    "openai": {
        "label": "OpenAI (gpt-4.1)",
        "model_name": "gpt-4.1",
        "base_url": None,               # default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai_gpt56_sol": {
        "label": "OpenAI (GPT-5.6 Sol)",
        "model_name": "gpt-5.6-sol",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "api": "responses",
    },
    "openai_gpt6_astra": {
        "label": "OpenAI (GPT-6 Astra)",
        "model_name": "gpt-6-astra",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "api": "responses",
    },
    "egi": {
        "label": "EGI (gpt-oss-120b)",
        "model_name": "gpt-oss-120b",
        "base_url": "https://llm.ai.egi.eu/v1",
        "api_key_env": "EGI_LLM_TOKEN",
    },
    
}



# Original
# gpt-5-mini
# gpt-4.1-mini
llm = os.getenv('LLM_MODEL', 'gpt-4.1')
#model = OpenAIModel("gpt-4.1", api_key=openai_api_key)
model = OpenAIChatModel("gpt-4.1")
#######################

# Caude start
def build_model(provider_key: str = "openai"):
    """Build a model for the selected provider (Chat Completions or Responses API)."""
    cfg = LLM_PROVIDERS.get(provider_key, LLM_PROVIDERS["openai"])
    api_key = os.getenv(cfg["api_key_env"])
    provider = OpenAIProvider(base_url=cfg["base_url"], api_key=api_key)
    if cfg.get("api") == "responses":
        return OpenAIResponsesModel(cfg["model_name"], provider=provider)
    return OpenAIChatModel(cfg["model_name"], provider=provider)

# Keep your existing default model for the agent's fallback:
model = build_model("openai")
# Claude done

# print("YOU ARE HERE ......")

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    backend: str = KB_BACKEND   # "vector" | "mcp" | "auto"

system_prompt = """
You are an expert at Environmental earth science and you have access to all the documentation to,
including examples, an API reference, and other resources.


Always let the user know when you didn't find the answer in the documentation or the right URL - DO NOT MAKE YOUR OWN URLs. Only Use URLS from the 'url' key.
Always generate your result with an appropriate URL that you get from the following context, URLS, Summary and Title.

Ask follow-up questions if you do not have enough context. If the context is not clear ask try to answer with your own knowledge and also ask follow up questions.
*** ALWAYS BE CONSISTENT WITH YOUR ANSWERS ***
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)


@pydantic_ai_expert.system_prompt
def knowledge_source_instructions(ctx: RunContext[PydanticAIDeps]) -> str:
    """Tell the model which knowledge sources are available for this request."""
    backend = ctx.deps.backend
    if backend == "vector":
        return "Use the documentation tools (retrieve_relevant_documentation, list_documentation_pages, get_page_content) to answer."
    mcp_part = (
        "The Ifremer knowledge base (MCP tools: search_argo_knowledge_base, get_argo_float, "
        "list_argo_filter_values) holds Euro-Argo float metadata (WMO ids, positions, sensors such as "
        "DOXY for dissolved oxygen, status, deployment dates, seas/ocean basins, operators) and the OSO "
        "ontology. Use it for questions about Argo floats, ocean observing platforms, sensors and the OSO "
        "ontology. When the user names a region, sensor or status, pass it as a filter; if unsure of the "
        "exact value, call list_argo_filter_values first. Cite WMO ids and any URLs present in the results."
    )
    if backend == "mcp":
        return mcp_part
    # auto: context from BOTH sources is fetched in code before the model runs
    # (see auto_combined_context below) and appended to these instructions.
    return (
        "AUTO MODE: for every question, relevant content has already been retrieved from BOTH sources "
        "and is included below under 'RETRIEVED CONTEXT': (A) the environmental documentation knowledge "
        "base (Supabase vector DB) and (B) the Ifremer MCP knowledge base (Euro-Argo float metadata + OSO "
        "ontology).\n"
        "1. Judge which retrieved items are actually relevant to the question; ignore the rest. One source "
        "may be entirely irrelevant, unavailable or empty - that is normal.\n"
        "2. Answer from the relevant items, combining both sources when both help.\n"
        "3. Only call tools if the retrieved context is insufficient: get_argo_float for one float's full "
        "record, search_argo_knowledge_base with filters (sea_area, ocean_basin, sensor such as DOXY, status, "
        "dates; check values with list_argo_filter_values), get_page_content for a full documentation page. "
        "Don't repeat a search that was already done with the same query.\n"
        "4. End the answer with a line 'Sources used:' naming the source(s) (Knowledge base / MCP) the answer "
        "is based on, with URLs or WMO ids from the context. Never invent URLs."
    )


# --- Tool gating: hide tools whose backend is disabled for this request -----
async def only_vector(ctx: RunContext[PydanticAIDeps], tool_def: ToolDefinition) -> Optional[ToolDefinition]:
    return tool_def if ctx.deps.backend in ("vector", "auto") else None


async def only_mcp(ctx: RunContext[PydanticAIDeps], tool_def: ToolDefinition) -> Optional[ToolDefinition]:
    return tool_def if ctx.deps.backend in ("mcp", "auto") else None

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        # print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error
    

    

@pydantic_ai_expert.tool(prepare=only_vector)
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        log_fetch(SOURCE_VECTOR, f"retrieve_relevant_documentation: {user_query!r}")
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        #print("User query", user_query, query_embedding)
        # Query Supabase for relevant documents
        # print("before matching ...")
        result = ctx.deps.supabase.rpc(
            'match_site_pages_6',
            {
                'query_embedding': query_embedding,
                'match_count': 10
            }
        ).execute()
        # print("Length of returned data ", len(result.data))
        # print("Data ", result.data)
        
        if not result.data:
            # print("Returning from here ...")
            log_source(SOURCE_VECTOR, "retrieve_relevant_documentation: no matching documents")
            return "No relevant documentation found."

        log_source(SOURCE_VECTOR, f"retrieve_relevant_documentation: {len(result.data)} chunks")
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            #print("Printing DOC", doc)
            chunk_text = f"""
                # {doc['title']}
                # {doc['url']}
                # {doc['summary']}
                {doc['content']}
                """
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        # print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool(prepare=only_vector)
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        log_fetch(SOURCE_VECTOR, "list_documentation_pages")
        result = ctx.deps.supabase.from_('site_pages_dev') \
            .select('url') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        log_source(SOURCE_VECTOR, f"list_documentation_pages: {len(urls)} pages")
        return urls
        
    except Exception as e:
        # print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool(prepare=only_vector)
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # print("WE ARE EXECUTING HERE")
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        log_fetch(SOURCE_VECTOR, f"get_page_content: {url}")
        result = ctx.deps.supabase.from_('site_pages_dev') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'environmental_docs') \
            .order('chunk_number') \
            .execute()
        
        # result = ctx.deps.supabase.from_('site_pages') \
        #     .select('title, content, chunk_number') \
        #     .eq('url', url) \
        #     .order('chunk_number') \
        #     .execute()
        
        if not result.data:
            # print("No content found ...")
            return f"No content found for URL: {url}"
            
        log_source(SOURCE_VECTOR, f"get_page_content: {url}")

        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        # print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"


# ---------------------------------------------------------------------------
# MCP knowledge base (Ifremer: Euro-Argo floats + OSO ontology)
# ---------------------------------------------------------------------------
_search_schema_cache: Optional[dict] = None


def _unwrap(exc: BaseException) -> BaseException:
    """anyio wraps transport errors in ExceptionGroups; get the real one."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


def _strip_heavy(obj: Any) -> Any:
    """Drop embedding vectors etc. so they don't eat the LLM context."""
    if isinstance(obj, dict):
        return {k: _strip_heavy(v) for k, v in obj.items()
                if not any(t in k.lower() for t in ("vector", "embedding"))}
    if isinstance(obj, list):
        return [_strip_heavy(v) for v in obj]
    return obj


async def _mcp_call(tool: str, arguments: dict) -> Any:
    """Open a session, call one MCP tool, return its (structured) result."""
    global _search_schema_cache
    headers = {"Authorization": f"Bearer {MCP_TOKEN}"} if MCP_TOKEN else {}
    async with connect(MCP_URL, headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Map our argument names onto whatever the server's search tool expects
            # (the deployed build uses `k`, newer builds `limit`).
            if tool == "search_knowledge_base":
                if _search_schema_cache is None:
                    tools = (await session.list_tools()).tools
                    t = next((t for t in tools if t.name == tool), None)
                    schema = attr(t, "input_schema", "inputSchema") if t else None
                    _search_schema_cache = (schema or {}).get("properties", {})
                props = _search_schema_cache
                if "limit" in arguments and "limit" not in props:
                    n = arguments.pop("limit")
                    for alt in ("k", "top_k"):
                        if alt in props:
                            arguments[alt] = n
                            break
                arguments = {k: v for k, v in arguments.items() if not props or k in props}

            result = await session.call_tool(tool, arguments)

    texts = [b.text for b in result.content if getattr(b, "text", None)]
    if attr(result, "is_error", "isError"):
        raise RuntimeError(" ".join(texts) or f"{tool} failed on the MCP server")
    structured = attr(result, "structured_content", "structuredContent")
    if structured is not None:
        return structured
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return joined


async def _mcp_tool(ctx: RunContext[PydanticAIDeps], tool: str, arguments: dict) -> str:
    """Call an MCP tool and format the result (or a useful error) for the LLM."""
    arguments = {k: v for k, v in arguments.items() if v is not None}
    # print(f"MCP -> {tool} {arguments}")
    log_fetch(SOURCE_MCP, f"{tool} {arguments}")
    try:
        data = await _mcp_call(tool, arguments)
    except Exception as e:  # noqa: BLE001 - surface any transport/tool error to the agent
        err = _unwrap(e)
        # print(f"MCP error ({tool}): {type(err).__name__}: {err}")
        msg = f"The Ifremer knowledge base (MCP) failed: {type(err).__name__}: {err}."
        if ctx.deps.backend == "auto":
            msg += " Rely on the knowledge base (vector DB) content instead."
            log_source(SOURCE_MCP, f"{tool} FAILED -> LLM told to rely on the knowledge base")
        else:
            log_source(SOURCE_MCP, f"{tool} FAILED")
        return msg

    log_source(SOURCE_MCP, tool)

    text = json.dumps(_strip_heavy(data), ensure_ascii=False, indent=1, default=str)
    if len(text) > MCP_MAX_CHARS:
        text = text[:MCP_MAX_CHARS] + "\n... [truncated; narrow the query or add filters]"
    return text


@pydantic_ai_expert.tool(prepare=only_mcp)
async def search_argo_knowledge_base(
    ctx: RunContext[PydanticAIDeps],
    query: str,
    limit: int = 10,
    sea_area: Optional[str] = None,
    ocean_basin: Optional[str] = None,
    sensor: Optional[str] = None,
    status: Optional[str] = None,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> str:
    """
    Hybrid semantic + keyword search over the Ifremer knowledge base:
    Euro-Argo float metadata and the OSO ontology.

    Args:
        query: Natural-language search, e.g. "Argo float measuring dissolved oxygen in the Southern Ocean".
        limit: Number of results (default 10).
        sea_area: Strict filter on named sea, e.g. "Labrador Sea" (see list_argo_filter_values("seas")).
        ocean_basin: Strict filter on ocean basin (see list_argo_filter_values("ocean_regions")).
        sensor: Strict filter on sensor code, e.g. "DOXY" (dissolved oxygen).
        status: Float status, e.g. "active" or "inactive".
        date_min: Last cycle date on/after, YYYY-MM-DD.
        date_max: Last cycle date on/before, YYYY-MM-DD.

    Filters are hard constraints and exclude OSO records; omit them for ontology questions.
    """
    return await _mcp_tool(ctx, "search_knowledge_base", {
        "query": query, "limit": limit, "sea_area": sea_area, "ocean_basin": ocean_basin,
        "sensor": sensor, "status": status, "date_min": date_min, "date_max": date_max,
    })


@pydantic_ai_expert.tool(prepare=only_mcp)
async def get_argo_float(ctx: RunContext[PydanticAIDeps], wmo: str) -> str:
    """
    Get the full metadata record of one Argo float by its WMO id (e.g. "6903240").
    """
    return await _mcp_tool(ctx, "get_argo_float", {"wmo": wmo})


_FILTER_VALUE_TOOLS = {
    "seas": "list_seas",
    "ocean_regions": "list_ocean_regions",
    "statuses": "list_argo_statuses",
    "oso_entity_types": "list_oso_entity_types",
}


@pydantic_ai_expert.tool(prepare=only_mcp)
async def list_argo_filter_values(ctx: RunContext[PydanticAIDeps], kind: str) -> str:
    """
    List the valid values for a search filter in the Ifremer knowledge base.

    Args:
        kind: One of "seas", "ocean_regions", "statuses", "oso_entity_types".
    """
    tool = _FILTER_VALUE_TOOLS.get(kind)
    if tool is None:
        return f"Unknown kind {kind!r}. Use one of: {', '.join(_FILTER_VALUE_TOOLS)}."
    return await _mcp_tool(ctx, tool, {})


# ---------------------------------------------------------------------------
# AUTO mode: fetch from BOTH sources before the model runs
# ---------------------------------------------------------------------------
# Instructions are re-evaluated on every model request of a run (but never stored in
# the chat history), so results are cached per run_id: one fetch per user question.
AUTO_MCP_LIMIT = int(os.getenv("AUTO_MCP_LIMIT", "10"))
_auto_cache: dict = {}            # run_id -> {"text": str, "sources": {source: summary}}
_AUTO_CACHE_MAX = 50


def _useful(text: str) -> bool:
    return bool(text) and not text.startswith(_NO_RESULT_PREFIXES)


# --- Debug printing ----------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_W = 100  # print width


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _clip(text: str) -> str:
    text = " ".join(str(text).split()) if len(str(text)) > 200 else str(text).strip()
    if KB_DEBUG_MAX_CHARS and len(text) > KB_DEBUG_MAX_CHARS:
        return text[:KB_DEBUG_MAX_CHARS] + _c(f" ... [+{len(text) - KB_DEBUG_MAX_CHARS} chars]", "2")
    return text


def _wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(line, width=_W, initial_indent=indent, subsequent_indent=indent) or indent
        for line in text.splitlines() or [""]
    )


def _debug_vector(text: str) -> None:
    if not _useful(text):
        print(_c(f"  (no usable content) {text}", "33"))
        return
    chunks = [c for c in text.split("\n\n---\n\n") if c.strip()]
    print(_c(f"  {len(chunks)} chunk(s)", "2"))
    for i, chunk in enumerate(chunks, 1):
        lines = textwrap.dedent(chunk).strip().splitlines()
        head = [l.strip().lstrip("#").strip() for l in lines[:3]]
        head += [""] * (3 - len(head))
        content = "\n".join(lines[3:]).strip()
        print(_c(f"\n  [{i}] {head[0]}", "1"))
        print(f"      url:     {head[1]}")
        print(_wrap(f"summary: {_clip(head[2])}", "      "))
        print(_wrap(f"content: {_clip(content)}", "      "))


_MCP_KEY_ORDER = ("_id", "wmo", "source", "title", "name", "status", "status_code", "sea_area",
                  "ocean_region", "sensor_codes", "last_cycle_date", "url", "score")


def _debug_mcp(text: str) -> None:
    if not _useful(text):
        print(_c(f"  (no usable content) {text}", "33"))
        return
    try:
        data = json.loads(text.split("\n... [truncated")[0])
    except json.JSONDecodeError:
        print(_c("  (result was truncated or not JSON; raw text follows)", "2"))
        print(_wrap(_clip(text)))
        return
    items = data.get("result", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        print(_wrap(_clip(json.dumps(items, ensure_ascii=False))))
        return
    print(_c(f"  {len(items)} result(s)", "2"))
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            print(f"\n  [{i}] {_clip(item)}")
            continue
        label = item.get("_id") or item.get("wmo") or item.get("title") or item.get("name") or ""
        print(_c(f"\n  [{i}] {label}", "1"))
        keys = [k for k in _MCP_KEY_ORDER if k in item] + [k for k in item if k not in _MCP_KEY_ORDER]
        for k in keys:
            v = item[k]
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            print(_wrap(f"{k + ':':<16} {_clip(v)}", "      "))


def debug_print_combined(query: str, vector_txt: str, mcp_txt: str, timings: dict) -> None:
    """Pretty-print the content from both sources that is passed to the LLM."""
    bar = "=" * _W
    print("\n" + _c(bar, "36"))
    print(_c(f" KB_DEBUG | combined context for: {query!r}", "1;36"))
    print(_c(f" total chars sent to the LLM: A={len(vector_txt):,}  B={len(mcp_txt):,}", "2"))
    print(_c(bar, "36"))

    print("\n" + _c(f"--- SOURCE A: {SOURCE_VECTOR}  ({timings.get('vector', 0):.2f}s) ", "1;32").ljust(_W, "-"))
    _debug_vector(vector_txt)

    print("\n" + _c(f"--- SOURCE B: {SOURCE_MCP}  ({timings.get('mcp', 0):.2f}s) ", "1;35").ljust(_W, "-"))
    _debug_mcp(mcp_txt)

    print("\n" + _c(bar, "36") + "\n", flush=True)


async def _timed(coro, timings: dict, key: str):
    t0 = time.perf_counter()
    try:
        return await coro
    finally:
        timings[key] = time.perf_counter() - t0


async def fetch_both_sources(ctx: RunContext[PydanticAIDeps], query: str) -> dict:
    """Query the vector DB and the MCP server in parallel; return combined, labelled context."""
    print(f"[Auto mode] Fetching from BOTH sources for: {query!r}", flush=True)
    timings: dict = {}
    vector_res, mcp_res = await asyncio.gather(
        _timed(retrieve_relevant_documentation(ctx, query), timings, "vector"),
        _timed(_mcp_tool(ctx, "search_knowledge_base", {"query": query, "limit": AUTO_MCP_LIMIT}), timings, "mcp"),
        return_exceptions=True,
    )
    vector_txt = vector_res if isinstance(vector_res, str) else f"Error retrieving documentation: {vector_res}"
    mcp_txt = mcp_res if isinstance(mcp_res, str) else f"The Ifremer knowledge base (MCP) failed: {mcp_res}"

    sources = {}
    if _useful(vector_txt):
        sources[SOURCE_VECTOR] = "documentation chunks"
    if _useful(mcp_txt):
        sources[SOURCE_MCP] = "Argo/OSO records"
    print("[Auto mode] Context passed to the LLM from: "
          + (" + ".join(sources) if sources else "neither source (nothing found)"), flush=True)

    if KB_DEBUG:
        debug_print_combined(query, vector_txt, mcp_txt, timings)

    text = (
        "RETRIEVED CONTEXT (auto mode; both sources were searched with the user's question)\n\n"
        f"=== SOURCE A: {SOURCE_VECTOR} ===\n{vector_txt}\n\n"
        f"=== SOURCE B: {SOURCE_MCP} ===\n{mcp_txt}\n"
    )
    return {"text": text, "sources": sources}


@pydantic_ai_expert.instructions
async def auto_combined_context(ctx: RunContext[PydanticAIDeps]) -> str:
    """In auto mode, attach content from both sources to the model's instructions."""
    if ctx.deps.backend != "auto" or not isinstance(ctx.prompt, str) or not ctx.prompt.strip():
        return ""
    entry = _auto_cache.get(ctx.run_id)
    if entry is None:
        entry = await fetch_both_sources(ctx, ctx.prompt)
        if len(_auto_cache) >= _AUTO_CACHE_MAX:
            _auto_cache.pop(next(iter(_auto_cache)))
        _auto_cache[ctx.run_id] = entry
    return entry["text"]


# ---------------------------------------------------------------------------
# After a run: report which source(s) the final answer was built from
# ---------------------------------------------------------------------------
_TOOL_SOURCES = {
    "retrieve_relevant_documentation": SOURCE_VECTOR,
    "list_documentation_pages": SOURCE_VECTOR,
    "get_page_content": SOURCE_VECTOR,
    "search_argo_knowledge_base": SOURCE_MCP,
    "get_argo_float": SOURCE_MCP,
    "list_argo_filter_values": SOURCE_MCP,
}


_NO_RESULT_PREFIXES = (
    "The Ifremer knowledge base (MCP) failed",
    "Unknown kind",
    "No relevant documentation found",
    "No content found",
    "Error retrieving",
)


def response_sources(result) -> List[str]:
    """Sources used in a finished agent run (works for run() and run_stream() results)."""
    from pydantic_ai.messages import ToolReturnPart
    used = []
    auto_entry = _auto_cache.get(getattr(result, "run_id", None))
    if auto_entry:
        used.extend(auto_entry["sources"])
    for msg in result.new_messages():
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                src = _TOOL_SOURCES.get(part.tool_name)
                content = part.content if isinstance(part.content, str) else ""
                if content.startswith(_NO_RESULT_PREFIXES) or part.content in ([], None):
                    continue  # failed or empty call - didn't contribute to the answer
                if src and src not in used:
                    used.append(src)
    return used


def print_response_source(result) -> None:
    """Print which sources supplied content to the LLM for this answer.

    In auto mode both sources are always searched; which items the LLM judged relevant
    is stated by the LLM itself in the 'Sources used:' line of its answer.
    """
    used = response_sources(result)
    if not used:
        print("[Response source] None - answered without retrieving anything (model knowledge)", flush=True)
    else:
        print("[Response source] Content supplied to the LLM from: " + " + ".join(used), flush=True)