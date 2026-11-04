import json

import pytest

from nacaarchive import harvest as H
from nacaarchive.ntrs import Client, RateLimiter


@pytest.mark.parametrize("rec,num", [
    ({"otherReportNumbers": ["NACA-TR-747"]}, "NACA-TR-747"),
    ({"otherReportNumbers": ["NACA-TN-1234"]}, "NACA-TN-1234"),
    ({"otherReportNumbers": ["NACA-RM-L50A01"]}, "NACA-RM-L50A01"),
    ({"otherReportNumbers": ["NACA-WR-L-123"]}, "NACA-WR-L-123"),
    ({"title": "NACA Report 640: The aerodynamic characteristics..."}, "NACA-TR-640"),
    ({"otherReportNumbers": ["NASA-TM-X-1234"]}, "NASA-TM-X-1234"),
    ({"title": "A study of propellers"}, None),
])
def test_report_number(rec, num):
    assert H.report_number(rec) == num


def test_year():
    assert H.year({"publications": [{"publicationDate": "1942-01-01T00:00:00Z"}]}) == 1942
    assert H.year({"distributionDate": "1950-03-01"}) == 1950
    assert H.year({}) is None


def test_on_topic(tmp_path):
    cfg = H.load_queries(H.Path(__file__).parents[1] / "queries" / "propellers.toml")
    assert H.on_topic({"title": "Tests of dual-rotating propellers"}, cfg)
    assert H.on_topic({"title": "Airscrew slipstream on a wing"}, cfg)
    assert not H.on_topic({"title": "Combustion of rocket propellants"}, cfg)
    assert not H.on_topic({"title": "Boundary layer on a flat plate"}, cfg)


def _client(results):
    def get(url):
        if "redistributions" in url:
            return json.dumps({"results": results.get("redis", [])}).encode()
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(url).query)["q"][0]
        start = int(parse_qs(urlparse(url).query)["page.from"][0])
        res = results.get(q, []) if start == 0 else []
        return json.dumps({"stats": {"total": len(results.get(q, []))}, "results": res}).encode()
    return Client(get=get, limiter=RateLimiter(10**6))


def test_harvest_and_redistributions(tmp_path):
    q = tmp_path / "q.toml"
    q.write_text('keep = "propell"\n[[query]]\nname = "a"\nq = "propeller"\n[[query]]\nname = "b"\nq = "slipstream"\n')
    r1 = {"id": 1, "title": "Propeller tests", "otherReportNumbers": ["NACA-TN-1"]}
    r2 = {"id": 2, "title": "Wing flutter"}
    r3 = {"id": 3, "title": "Propeller slipstream"}
    cl = _client({"propeller": [r1, r2, r3], "slipstream": [r3]})
    changed = H.harvest(q, tmp_path, client=cl, log=lambda *a: None)
    assert sorted(changed) == ["1", "3"]
    recs = H._read_jsonl(tmp_path / "metadata" / "records.jsonl")
    assert recs["1"]["_number"] == "NACA-TN-1" and recs["3"]["_queries"] == ["a", "b"]
    assert H.harvest(q, tmp_path, client=cl, log=lambda *a: None) == {}   # idempotent
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "1.pdf").write_bytes(b"%PDF")
    (tmp_path / "reports" / "1").mkdir(parents=True)
    cl2 = _client({"redis": [{"id": 1, "disseminated": "NONE"}, {"id": 3, "disseminated": "METADATA_ONLY"}]})
    assert H.check_redistributions(tmp_path, client=cl2, log=lambda *a: None) == ["1", "3"]
    recs = H._read_jsonl(tmp_path / "metadata" / "records.jsonl")
    assert "1" not in recs and recs["3"]["disseminated"] == "METADATA_ONLY"
    assert not (tmp_path / "pdfs" / "1.pdf").exists() and not (tmp_path / "reports" / "1").exists()
    # a removed record is never re-added
    assert "1" not in H.harvest(q, tmp_path, client=cl, log=lambda *a: None)
