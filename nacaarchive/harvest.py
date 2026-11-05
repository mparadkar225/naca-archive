"""Search NTRS with a query set, keep on-topic records, download their PDFs.

State lives in the data directory:
  metadata/records.jsonl   one NTRS record per line (latest version), sorted by id
  metadata/removed.json    ids removed after redistribution notices (never re-added)
  harvest_state.json       last redistribution check and per-query counts
"""
from __future__ import annotations

import json
import re
import tomllib
from datetime import date, timedelta
from pathlib import Path

from .ntrs import Client, removal_action

# NACA/NASA report numbers as they appear in titles and otherReportNumbers.
_SERIES = r"TR|TN|TM|RM|WR|ACR|ARR|CB|MR|AR|TP|CR|SP|TT|TM-X|TN-D|TR-R"
_NUM_RE = re.compile(
    rf"\b(NACA|NASA)[\s\-_]*(?:Report|Rept\.?)?[\s\-_]*({_SERIES})?[\s\-_]*(?:No\.?\s*)?"
    r"([A-Z]{0,2}-?\d[\w\-]*\d|[A-Z]{0,2}-?\d+)\b", re.I)


def report_number(record: dict) -> str | None:
    """Normalized NACA/NASA number ('NACA-TR-747', 'NACA-RM-L50A01'), from the report
    numbers first, then the title. 'NACA Report 747' is the TR series."""
    fields = list(record.get("otherReportNumbers") or []) + [record.get("title") or ""]
    for text in fields:
        m = _NUM_RE.search(text)
        if m:
            org, series, num = m.group(1).upper(), (m.group(2) or "").upper(), m.group(3).upper()
            if not series:
                if org == "NACA" and re.search(r"report|rept", m.group(0), re.I):
                    series = "TR"
                else:
                    continue
            return f"{org}-{series}-{num}"
    return None


def year(record: dict) -> int | None:
    for p in record.get("publications") or []:
        d = p.get("publicationDate") or ""
        if d[:4].isdigit():
            return int(d[:4])
    for k in ("distributionDate", "created"):
        d = record.get(k) or ""
        if d[:4].isdigit():
            return int(d[:4])
    return None


def load_queries(path: Path) -> dict:
    cfg = tomllib.loads(Path(path).read_text())
    cfg["_keep"] = re.compile(cfg.get("keep", "."), re.I)
    cfg["_drop"] = re.compile(cfg["drop"], re.I) if cfg.get("drop") else None
    return cfg


def on_topic(record: dict, cfg: dict) -> bool:
    text = f"{record.get('title') or ''} {record.get('abstract') or ''}"
    if not cfg["_keep"].search(text):
        return False
    return not (cfg["_drop"] and cfg["_drop"].search(text))


def _read_jsonl(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    return {str(r["id"]): r for r in map(json.loads, p.read_text().splitlines()) if r}


def _write_jsonl(p: Path, recs: dict[str, dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(recs[k], sort_keys=True) + "\n" for k in sorted(recs)))
    tmp.replace(p)


def harvest(queries: Path, data: Path, client: Client | None = None, limit: int | None = None,
            log=print) -> dict[str, dict]:
    """Run every query; merge kept records into metadata/records.jsonl. Returns the
    records added or changed."""
    client = client or Client()
    cfg = load_queries(queries)
    store = data / "metadata" / "records.jsonl"
    recs = _read_jsonl(store)
    removed = set(_load_json(data / "metadata" / "removed.json", []))
    state = _load_json(data / "harvest_state.json", {})
    changed = {}
    for q in cfg.get("query", []):
        params = {k: v for k, v in q.items() if k != "name"}
        seen = kept = 0
        for r in client.search(params, limit=limit):
            seen += 1
            rid = str(r["id"])
            if rid in removed or not on_topic(r, cfg):
                continue
            kept += 1
            r["_number"] = report_number(r)
            r["_year"] = year(r)
            r.setdefault("_queries", [])
            old = recs.get(rid)
            qs = sorted(set((old or {}).get("_queries", [])) | {q["name"]})
            r["_queries"] = qs
            if old != r:
                recs[rid] = r
                changed[rid] = r
        state.setdefault("queries", {})[q["name"]] = {"seen": seen, "kept": kept}
        log(f"{q['name']}: {seen} results, {kept} on topic")
    # reports asked for by id (`ids = [...]` in the query file): kept whatever their topic
    for rid in map(str, cfg.get("ids", [])):
        if rid in removed:
            continue
        try:
            r = client.citation(rid)
        except Exception as e:
            log(f"{rid}: citation failed: {e}")
            continue
        r["_number"] = report_number(r)
        r["_year"] = year(r)
        old = recs.get(rid)
        r["_queries"] = sorted(set((old or {}).get("_queries", [])) | {"ids"})
        if old != r:
            recs[rid] = r
            changed[rid] = r
        log(f"{rid}: {r['_number'] or ''} {r.get('title', '')[:80]}")
    _write_jsonl(store, recs)
    _dump_json(data / "harvest_state.json", state)
    log(f"{len(recs)} records, {len(changed)} new or changed")
    return changed


def check_redistributions(data: Path, client: Client | None = None, log=print) -> list[str]:
    """Apply NTRS redistribution notices since the last check (default: 30 days ago).
    Returns the ids acted on."""
    client = client or Client()
    state = _load_json(data / "harvest_state.json", {})
    since = state.get("redistributions_checked") or (date.today() - timedelta(days=30)).isoformat()
    recs = _read_jsonl(data / "metadata" / "records.jsonl")
    removed = set(_load_json(data / "metadata" / "removed.json", []))
    acted = []
    for item in client.redistributions(since):
        rid = str(item.get("id"))
        act = removal_action(item)
        if rid not in recs or act is None:
            continue
        acted.append(rid)
        if act == "remove":
            recs.pop(rid)
            removed.add(rid)
        else:
            recs[rid]["disseminated"] = "METADATA_ONLY"
            recs[rid]["downloads"] = []
        # Both actions drop our document copies: the PDF and everything derived from its
        # pages (text, figures, tables). Metadata stays only for METADATA_ONLY.
        for f in (data / "pdfs" / f"{rid}.pdf", data / "reports" / rid):
            _rm(f)
        log(f"redistribution: {rid} -> {act}")
    _write_jsonl(data / "metadata" / "records.jsonl", recs)
    _dump_json(data / "metadata" / "removed.json", sorted(removed))
    state["redistributions_checked"] = date.today().isoformat()
    _dump_json(data / "harvest_state.json", state)
    return acted


def select(recs: dict, ids=None, match: str | None = None) -> dict:
    """Records chosen by NTRS id and/or a regex over title and abstract (case-insensitive)."""
    out = dict(recs)
    if ids:
        out = {k: v for k, v in out.items() if str(k) in {str(i) for i in ids}}
    if match:
        rx = re.compile(match, re.I)
        out = {k: v for k, v in out.items() if rx.search(f"{v.get('title') or ''} {v.get('abstract') or ''}")}
    return out


def download_all(data: Path, pdf_dir: Path, client: Client | None = None, max_new: int | None = None,
                 log=print, ids=None, match: str | None = None) -> list[Path]:
    client = client or Client()
    recs = select(_read_jsonl(data / "metadata" / "records.jsonl"), ids, match)
    got, new = [], 0
    for rid, r in sorted(recs.items()):
        if max_new is not None and new >= max_new and not (pdf_dir / f"{rid}.pdf").exists():
            continue
        existed = (pdf_dir / f"{rid}.pdf").exists()
        try:
            p = client.download(r, pdf_dir)
        except Exception as e:
            log(f"{rid}: download failed: {e}")
            continue
        if p:
            got.append(p)
            new += not existed
    log(f"{len(got)} PDFs available, {new} downloaded now")
    return got


def _rm(p: Path):
    import shutil
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        p.unlink()


def _load_json(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def _dump_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
