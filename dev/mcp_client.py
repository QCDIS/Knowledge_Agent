"""Query the Ifremer KB MCP server (streamable HTTP) with a bearer token.

Setup:
    pip install -U mcp
    export KB_TOKEN=<one of your KB_READ_TOKENS values, e.g. the chatbot token>

Usage:
    python kb_mcp_client.py                      # default Southern Ocean DOXY query
    python kb_mcp_client.py --list-tools         # show tool names + input schemas
    python kb_mcp_client.py --list-seas          # valid values for sea_area
    python kb_mcp_client.py -q "Argo float Labrador Sea oxygen" --sea-area "Labrador Sea"
"""
import argparse
import asyncio
import json
import os
import sys

from contextlib import asynccontextmanager

from mcp import ClientSession

try:  # mcp >= 2.0
    from mcp.client.streamable_http import streamable_http_client
    import httpx2 as _httpx

    @asynccontextmanager
    async def connect(url, headers):
        async with _httpx.AsyncClient(headers=headers, timeout=60) as http:
            async with streamable_http_client(url, http_client=http) as streams:
                yield streams[0], streams[1]

except ImportError:  # mcp 1.x
    from mcp.client.streamable_http import streamablehttp_client

    @asynccontextmanager
    async def connect(url, headers):
        async with streamablehttp_client(url, headers=headers) as streams:
            yield streams[0], streams[1]


DEFAULT_URL = "http://145.100.135.157:8765/mcp"
SEARCH_TOOL = "search_knowledge_base"


def attr(obj, snake, camel):
    """mcp 2.x uses snake_case attributes, 1.x uses camelCase."""
    return getattr(obj, snake, None) or getattr(obj, camel, None)


def show(result):
    """Print structured output if the server provides it, else the text blocks."""
    structured = attr(result, "structured_content", "structuredContent")
    if structured:
        print(json.dumps(structured, indent=2, ensure_ascii=False))
        return
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            print(block)
            continue
        try:
            print(json.dumps(json.loads(text), indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(text)


async def main(args):
    token = os.environ.get("OPENAI_API_KEY")
    if not token:
        sys.exit("Set KB_TOKEN to one of your KB_READ_TOKENS values.")

    headers = {"Authorization": f"Bearer {token}"}

    async with connect(args.url, headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = {t.name: t for t in (await session.list_tools()).tools}

            if args.list_tools:
                for t in tools.values():
                    props = list((attr(t, "input_schema", "inputSchema") or {}).get("properties", {}))
                    print(f"{t.name}: {props}")
                return

            if args.list_seas:
                show(await session.call_tool("list_seas", {}))
                return

            if SEARCH_TOOL not in tools:
                sys.exit(f"{SEARCH_TOOL!r} not found. Available: {sorted(tools)}")

            # Build arguments from the tool's own schema so renamed fields don't break us.
            props = (attr(tools[SEARCH_TOOL], "input_schema", "inputSchema") or {}).get("properties", {})
            arguments = {"query": args.query}

            limit_name = next((n for n in ("limit", "k", "top_k") if n in props), None)
            if limit_name:
                arguments[limit_name] = args.limit

            filters = {
                "sea_area": args.sea_area,
                "ocean_basin": args.ocean_basin,
                "sensor": args.sensor,
                "status": args.status,
                "date_min": args.date_min,
                "date_max": args.date_max,
            }
            for name, value in filters.items():
                if value is None:
                    continue
                if name not in props:
                    print(f"warning: server has no filter {name!r}, skipping", file=sys.stderr)
                    continue
                arguments[name] = value

            print(f"Calling {SEARCH_TOOL} with {arguments}\n", file=sys.stderr)
            result = await session.call_tool(SEARCH_TOOL, arguments)
            if attr(result, "is_error", "isError"):
                print("Tool returned an error:", file=sys.stderr)
            show(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("-q", "--query",
                   default="Argo float measuring dissolved oxygen in the Southern Ocean")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("--sensor", default="DOXY", help="sensor code; pass '' via --sensor= to disable")
    p.add_argument("--sea-area")
    p.add_argument("--ocean-basin")
    p.add_argument("--status")
    p.add_argument("--date-min", help="YYYY-MM-DD")
    p.add_argument("--date-max", help="YYYY-MM-DD")
    p.add_argument("--list-tools", action="store_true")
    p.add_argument("--list-seas", action="store_true")
    a = p.parse_args()
    if a.sensor == "":
        a.sensor = None
    try:
        asyncio.run(main(a))
    except BaseException as exc:  # unwrap anyio ExceptionGroups into one readable line
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        while getattr(exc, "exceptions", None):
            exc = exc.exceptions[0]
        msg = str(exc)
        hint = ""
        if "401" in msg or "Server returned an error" in msg:
            hint = " (check KB_TOKEN is one of KB_READ_TOKENS)"
        elif "403" in msg or "421" in msg:
            hint = " (add this address to MCP_ALLOWED_HOSTS)"
        elif "ConnectError" in type(exc).__name__ or "connect" in msg.lower():
            hint = " (is the container up / port 8765 reachable?)"
        sys.exit(f"{type(exc).__name__}: {msg}{hint}")