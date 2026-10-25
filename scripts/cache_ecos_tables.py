from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ecos_tables.csv"
BASE_URL = "https://ecos.bok.or.kr/api"


def get_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit("Set ECOS_API_KEY or BOK_API_KEY.")
    return key


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts if part is not None)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def unwrap(payload: dict[str, Any], service: str) -> tuple[int, list[dict[str, Any]]]:
    if "RESULT" in payload:
        result = payload["RESULT"]
        raise RuntimeError(f"{result.get('CODE')}: {result.get('MESSAGE')}")
    body = payload.get(service) or {}
    rows = body.get("row") or []
    if isinstance(rows, dict):
        rows = [rows]
    return int(body.get("list_total_count", len(rows))), rows


def fetch_paged(service: str, key: str, *parts: str, page_size: int = 1000) -> list[dict[str, Any]]:
    start_no = 1
    total = None
    rows: list[dict[str, Any]] = []
    while total is None or start_no <= total:
        url = ecos_url(service, key, start_no, start_no + page_size - 1, *parts)
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        total, page = unwrap(payload, service)
        rows.extend(page)
        if not page:
            break
        start_no += page_size
    return rows


def main() -> int:
    rows = fetch_paged("StatisticTableList", get_key())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
