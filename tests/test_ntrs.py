import json

import pytest

from nacaarchive import ntrs
from nacaarchive.ntrs import Client, RateLimiter, _flatten, removal_action


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_flatten_nested_params():
    q = {"q": "propeller", "published": {"lt": "1959"}, "page": {"size": 100, "from": 0},
         "subjectCategory": ["Aerodynamics", "Aeronautics (General)"], "highlight": True}
    assert _flatten(q) == [("q", "propeller"), ("published.lt", "1959"), ("page.size", "100"), ("page.from", "0"),
                           ("subjectCategory", "Aerodynamics"), ("subjectCategory", "Aeronautics (General)"),
                           ("highlight", "true")]


def test_rate_limiter_waits_after_n_calls():
    c = FakeClock()
    lim = RateLimiter(n=3, window=10.0, clock=c.clock, sleep=c.sleep)
    for _ in range(3):
        lim.wait()
    assert c.slept == []
    lim.wait()
    assert len(c.slept) == 1 and c.slept[0] >= 10.0


def fake_search(total, pages_seen):
    def get(url):
        pages_seen.append(url)
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(url).query)
        start, size = int(q["page.from"][0]), int(q["page.size"][0])
        ids = list(range(start, min(start + size, total)))
        return json.dumps({"stats": {"total": total}, "results": [{"id": i} for i in ids]}).encode()
    return get


def test_search_pages_through_all_results():
    seen = []
    cl = Client(get=fake_search(250, seen), limiter=RateLimiter(10**6))
    got = [r["id"] for r in cl.search({"q": "x"})]
    assert got == list(range(250))
    assert len(seen) == 3 and "page.size=100" in seen[0] and "page.from=200" in seen[2]


def test_search_respects_limit_and_cap():
    seen = []
    cl = Client(get=fake_search(50_000, seen), limiter=RateLimiter(10**6))
    assert len(list(cl.search({"q": "x"}, limit=130))) == 130
    seen.clear()
    assert len(list(cl.search({"q": "x"}))) == ntrs.QUERY_CAP


def test_download(tmp_path):
    rec = {"id": 123, "disseminated": "DOCUMENT_AND_METADATA",
           "downloads": [{"mimetype": "application/pdf", "draft": False,
                          "links": {"pdf": "/api/citations/123/downloads/123.pdf"}}]}
    urls = []
    cl = Client(get=lambda u: urls.append(u) or b"%PDF-1.4 fake", limiter=RateLimiter(10**6))
    p = cl.download(rec, tmp_path)
    assert p.read_bytes().startswith(b"%PDF") and urls == ["https://ntrs.nasa.gov/api/citations/123/downloads/123.pdf"]
    cl.download(rec, tmp_path)          # cached: no second request
    assert len(urls) == 1
    assert cl.download(dict(rec, id=124, disseminated="METADATA_ONLY"), tmp_path) is None
    bad = Client(get=lambda u: b"<html>", limiter=RateLimiter(10**6))
    with pytest.raises(ValueError):
        bad.download(dict(rec, id=125), tmp_path)


def test_removal_action():
    assert removal_action({"disseminated": "NONE"}) == "remove"
    assert removal_action({"disseminated": "DOCUMENT_AND_METADATA", "distribution": "DO_NOT_DISTRIBUTE"}) == "remove"
    assert removal_action({"disseminated": "METADATA_ONLY"}) == "strip_document"
    assert removal_action({"disseminated": "DOCUMENT_AND_METADATA", "distribution": "PUBLIC"}) is None


def test_unwrap_pdf():
    from nacaarchive.ntrs import unwrap_pdf
    pdf = b"%PDF-1.4\n...%%EOF"
    head = bytes.fromhex("f55280df00000002011fcc99") + b"19930093302.pdf" + bytes(49)
    assert unwrap_pdf(pdf) == pdf and unwrap_pdf(head + pdf) == pdf
    with pytest.raises(ValueError, match="f5 52 80 df"):
        unwrap_pdf(head + b"compressed")
