"""NASA Technical Reports Server (NTRS) OpenAPI client.

Documented limits (sti.nasa.gov, "OpenAPI Documentation", 2022-11-22):
  * search: up to 100 results per page, at most 10,000 records per query
  * 500 requests per 15 minutes; exceeding or circumventing limits can get a client blocked
  * /citations/redistributions must be checked at least weekly: records that became
    NONE / DO_NOT_DISTRIBUTE are removed, METADATA_ONLY ones lose their document copies
  * cite "NASA STI Compliance and Distribution Services" as the source, and link to
    https://ntrs.nasa.gov/citations/{id}

The transport is injectable (``Client(get=...)``) so tests run on recorded JSON.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

BASE = "https://ntrs.nasa.gov"
API = BASE + "/api"
USER_AGENT = "naca-archive/0.1 (+https://github.com/mparadkar225/naca-archive)"
PAGE_SIZE = 100
QUERY_CAP = 10_000


def citation_url(ntrs_id) -> str:
    return f"{BASE}/citations/{ntrs_id}"


class RateLimiter:
    """Sliding window: at most ``n`` calls per ``window`` seconds (default 450 / 15 min,
    a margin under the documented 500)."""

    def __init__(self, n: int = 450, window: float = 900.0, clock=time.monotonic, sleep=time.sleep):
        self.n, self.window, self.clock, self.sleep = n, window, clock, sleep
        self.calls: deque[float] = deque()

    def wait(self):
        now = self.clock()
        while self.calls and now - self.calls[0] >= self.window:
            self.calls.popleft()
        if len(self.calls) >= self.n:
            self.sleep(self.window - (now - self.calls[0]) + 0.5)
            return self.wait()
        self.calls.append(now)


def _urllib_get(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _flatten(params: dict, prefix: str = "") -> list[tuple[str, str]]:
    """{'page': {'size': 100}, 'published': {'lt': '1959'}} -> page.size=100&published.lt=1959"""
    out = []
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out += _flatten(v, key)
        elif isinstance(v, (list, tuple)):
            out += [(key, str(x)) for x in v]
        elif v is not None:
            out.append((key, str(v).lower() if isinstance(v, bool) else str(v)))
    return out


@dataclass
class Client:
    get: Callable[[str], bytes] = _urllib_get
    limiter: RateLimiter = field(default_factory=RateLimiter)
    retries: int = 4

    def _get(self, url: str) -> bytes:
        delay = 5.0
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                return self.get(url)
            except Exception as e:  # urllib.error.HTTPError, URLError, timeouts
                code = getattr(e, "code", None)
                if code is not None and code < 500 and code != 429 or attempt == self.retries:
                    raise
                time.sleep(delay)
                delay *= 3
        raise AssertionError("unreachable")

    def _json(self, path: str, params: dict | None = None) -> dict:
        url = API + path
        if params:
            url += "?" + urllib.parse.urlencode(_flatten(params))
        return json.loads(self._get(url))

    # ------------------------------------------------------------------ search
    def search_page(self, query: dict, start: int = 0, size: int = PAGE_SIZE) -> dict:
        q = dict(query)
        q["page"] = {"size": size, "from": start}
        return self._json("/citations/search", q)

    def search(self, query: dict, limit: int | None = None) -> Iterator[dict]:
        """All results of a query, paged (stops at the 10,000 cap, or ``limit``)."""
        start, total = 0, None
        cap = min(QUERY_CAP, limit or QUERY_CAP)
        while start < cap and (total is None or start < total):
            page = self.search_page(query, start, min(PAGE_SIZE, cap - start))
            total = page.get("stats", {}).get("total", 0)
            results = page.get("results", [])
            if not results:
                return
            yield from results
            start += len(results)

    def citation(self, ntrs_id) -> dict:
        return self._json(f"/citations/{ntrs_id}")

    def redistributions(self, since: str) -> Iterator[dict]:
        start = 0
        while True:
            page = self._json("/citations/redistributions",
                              {"redistributedDate": {"gt": since}, "page": {"size": PAGE_SIZE, "from": start}})
            res = page.get("results", [])
            yield from res
            if len(res) < PAGE_SIZE:
                return
            start += len(res)

    # --------------------------------------------------------------- downloads
    def download(self, record: dict, dest_dir: Path) -> Path | None:
        """The record's PDF into dest_dir/{id}.pdf (skipped if present). None if the
        record has no document available."""
        if record.get("disseminated") not in (None, "DOCUMENT_AND_METADATA"):
            return None
        pdf = next((d for d in record.get("downloads") or []
                    if d.get("mimetype") == "application/pdf" and not d.get("draft")), None)
        if pdf is None:
            return None
        link = pdf.get("links", {}).get("pdf") or pdf.get("links", {}).get("original")
        if not link:
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{record['id']}.pdf"
        if out.exists() and out.stat().st_size > 0:
            return out
        data = unwrap_pdf(self._get(BASE + link if link.startswith("/") else link), record["id"])
        tmp = out.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(out)
        return out


def unwrap_pdf(data: bytes, rid="") -> bytes:
    """The PDF in a download. Some NTRS files (about one in seven of the old scans) come
    wrapped: a header starting f5 52 80 df, a length and the file name, then the PDF."""
    if data.startswith(b"%PDF"):
        return data
    i = data.find(b"%PDF-", 0, 1 << 16)
    if i > 0:
        return data[i:]
    raise ValueError(f"{rid}: download is not a PDF (starts {data[:160].hex(' ')})")


def removal_action(item: dict) -> str | None:
    """What a redistribution notice requires of us: 'remove', 'strip_document' or None."""
    if item.get("disseminated") == "NONE" or item.get("distribution") == "DO_NOT_DISTRIBUTE":
        return "remove"
    if item.get("disseminated") == "METADATA_ONLY":
        return "strip_document"
    return None
