"""Smoke test against a running server: python tests/smoke_client.py http://HOST:8765/mcp TOKEN [prompt]"""
import asyncio, base64, json, sys, time, urllib.request

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client


async def main(url: str, token: str, prompt: str) -> None:
    # Unauthenticated requests must be refused.
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST"))
        print("FAIL: no-token request accepted")
    except urllib.error.HTTPError as e:
        print("no token ->", e.code)

    client = create_mcp_http_client(headers={"Authorization": f"Bearer {token}"})
    async with streamable_http_client(url, http_client=client) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        print("tools:", [t.name for t in (await s.list_tools()).tools])
        models = json.loads((await s.call_tool("list_models", {"media": "image"})).content[0].text)
        print("device:", models["device"], models["memory_gb"], "GB")
        print("downloaded:", [m["id"] for m in models["models"] if m["downloaded"]])

        async def progress(p, total, msg):
            print(f"  progress {p}/{total} {msg}")

        t = time.time()
        r = await s.call_tool("generate_image", {"prompt": prompt, "width": 768, "height": 512, "seed": 7},
                              progress_callback=progress)
        print(f"generate_image ({time.time()-t:.0f}s) isError={r.is_error}")
        print(r.content[0].text)
        for c in r.content[1:]:
            print("inline", c.type, c.mime_type, len(base64.b64decode(c.data)), "bytes")
        if not r.is_error:
            info = json.loads(r.content[0].text)
            data = urllib.request.urlopen(info["url"]).read()
            print("fetched", info["url"], len(data), "bytes")


asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else
                 "a red fox reading a newspaper in a sunny cafe, photo"))
