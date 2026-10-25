from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ecos_full_catalog.csv"
BASE_URL = "https://ecos.bok.or.kr/api"
CYCLES = ["M", "Q", "A", "D", "S", "Y"]


def get_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit("Set ECOS_API_KEY or BOK_API_KEY.")
    return key


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts if part is not None)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def request_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network: {exc.reason}") from exc


def unwrap(payload: dict[str, Any], service: str) -> tuple[int, list[dict[str, Any]]]:
    if "RESULT" in payload:
        result = payload["RESULT"]
        code = result.get("CODE")
        message = result.get("MESSAGE")
        if code == "INFO-200":
            return 0, []
        raise RuntimeError(f"{code}: {message}")
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
        payload = request_json(ecos_url(service, key, start_no, start_no + page_size - 1, *parts))
        total, page = unwrap(payload, service)
        rows.extend(page)
        if not page:
            break
        start_no += page_size
        time.sleep(0.08)
    return rows


def main() -> int:
    key = get_key()
    tables = fetch_paged("StatisticTableList", key)
    catalog: list[dict[str, Any]] = []
    searchable = [t for t in tables if t.get("SRCH_YN") == "Y"]
    for index, table in enumerate(searchable, start=1):
        stat_code = table.get("STAT_CODE")
        stat_name = table.get("STAT_NAME")
        print(f"[{index}/{len(searchable)}] {stat_code} {stat_name}")
        had_item = False
        for cycle in CYCLES:
            try:
                items = fetch_paged("StatisticItemList", key, stat_code, cycle)
            except RuntimeError as exc:
                if "INFO-200" in str(exc):
                    items = []
                else:
                    items = []
            for item in items:
                had_item = True
                row = dict(table)
                row.update(
                    {
                        "CYCLE": item.get("CYCLE") or cycle,
                        "ITEM_CODE": item.get("ITEM_CODE"),
                        "ITEM_NAME": item.get("ITEM_NAME"),
                        "P_ITEM_CODE": item.get("P_ITEM_CODE"),
                        "UNIT_NAME": item.get("UNIT_NAME"),
                        "START_TIME": item.get("START_TIME"),
                        "END_TIME": item.get("END_TIME"),
                    }
                )
                catalog.append(row)
        if not had_item:
            row = dict(table)
            row.update({"CYCLE": "", "ITEM_CODE": "", "ITEM_NAME": "", "P_ITEM_CODE": "", "UNIT_NAME": ""})
            catalog.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in catalog for key in row})
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(catalog)
    print(f"Saved {OUT} rows={len(catalog)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
