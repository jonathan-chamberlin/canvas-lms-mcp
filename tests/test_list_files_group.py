"""Tests for list_files group_id support.

Unit tests use a stub client. Live tests (marked with LIVE env flag) hit
the real Canvas API using credentials from the environment.
"""
from __future__ import annotations

from typing import List

import httpx
import pytest

from canvas_lms_mcp import main as canvas_main
from canvas_lms_mcp.client import CanvasClient, _next_link


# ---------- unit: Link header parsing ----------

def test_next_link_parses_rel_next():
    header = (
        '<https://x/api?page=2>; rel="next", '
        '<https://x/api?page=1>; rel="current", '
        '<https://x/api?page=5>; rel="last"'
    )
    assert _next_link(header) == "https://x/api?page=2"


def test_next_link_none_when_missing():
    assert _next_link(None) is None
    assert _next_link('<https://x>; rel="last"') is None


# ---------- unit: list_files validation ----------

@pytest.mark.asyncio
async def test_list_files_rejects_multiple_scopes():
    with pytest.raises(ValueError, match="at most one"):
        await canvas_main.list_files(course_id=1, group_id=2)


@pytest.mark.asyncio
async def test_list_files_rejects_short_search_term(monkeypatch):
    monkeypatch.setattr(CanvasClient, "get_instance",
                        classmethod(lambda cls: _StubClient([])))
    with pytest.raises(ValueError, match="at least 2"):
        await canvas_main.list_files(group_id=1, search_term="a")


@pytest.mark.asyncio
async def test_list_files_rejects_bad_max_pages(monkeypatch):
    monkeypatch.setattr(CanvasClient, "get_instance",
                        classmethod(lambda cls: _StubClient([])))
    with pytest.raises(ValueError, match="max_pages"):
        await canvas_main.list_files(group_id=1, max_pages=0)


@pytest.mark.asyncio
async def test_list_files_search_term_passthrough(monkeypatch):
    stub = _StubClient([{"id": 1, "display_name": "x.pdf", "url": "https://x"}])
    monkeypatch.setattr(CanvasClient, "get_instance", classmethod(lambda cls: stub))
    await canvas_main.list_files(group_id=42, search_term="rough")
    _, params = stub.calls[0]
    assert params["search_term"] == "rough"


# ---------- unit: list_files routes to group endpoint ----------

class _StubClient:
    def __init__(self, payload: List[dict]):
        self._payload = payload
        self.calls: List[tuple] = []

    async def get_all(self, endpoint: str, params=None, max_pages: int = 50):
        self.calls.append((endpoint, dict(params or {})))
        return list(self._payload)


@pytest.mark.asyncio
async def test_list_files_group_id_hits_groups_endpoint(monkeypatch):
    stub = _StubClient([
        {"id": 1, "display_name": "a.pdf", "url": "https://x/a"},
        {"id": 2, "display_name": "b.pdf", "url": "https://x/b"},
    ])
    monkeypatch.setattr(CanvasClient, "get_instance", classmethod(lambda cls: stub))

    result = await canvas_main.list_files(group_id=325416, items_per_page=50)

    assert stub.calls[0][0] == "/api/v1/groups/325416/files"
    assert [f.display_name for f in result.items] == ["a.pdf", "b.pdf"]


# ---------- unit: 403/404 mapped to typed errors ----------

class _ErrorClient:
    def __init__(self, status: int):
        self._status = status

    async def get_all(self, endpoint: str, params=None, max_pages: int = 50):
        request = httpx.Request("GET", f"https://x{endpoint}")
        response = httpx.Response(self._status, text="denied", request=request)
        raise httpx.HTTPStatusError("err", request=request, response=response)


@pytest.mark.asyncio
async def test_list_files_maps_403_to_permission_error(monkeypatch):
    monkeypatch.setattr(CanvasClient, "get_instance",
                        classmethod(lambda cls: _ErrorClient(403)))
    with pytest.raises(PermissionError, match="Not authorized"):
        await canvas_main.list_files(group_id=999)


@pytest.mark.asyncio
async def test_list_files_maps_404_to_lookup_error(monkeypatch):
    monkeypatch.setattr(CanvasClient, "get_instance",
                        classmethod(lambda cls: _ErrorClient(404)))
    with pytest.raises(LookupError, match="not found"):
        await canvas_main.list_files(group_id=999)


# ---------- unit: get_all follows pagination ----------

class _PaginatedTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            return httpx.Response(
                200,
                json=[{"id": 1}, {"id": 2}],
                headers={"Link": '<https://c/next>; rel="next"'},
            )
        return httpx.Response(200, json=[{"id": 3}])


@pytest.mark.asyncio
async def test_get_all_follows_link_header():
    transport = _PaginatedTransport()
    client = httpx.AsyncClient(base_url="https://c", transport=transport)
    # Build a minimal CanvasClient-like object reusing get_all's logic.
    cc = CanvasClient.__new__(CanvasClient)
    cc.client = client
    try:
        results = await cc.get_all("/list", {"per_page": 2})
    finally:
        await client.aclose()
    assert [r["id"] for r in results] == [1, 2, 3]
    assert transport.calls == 2
