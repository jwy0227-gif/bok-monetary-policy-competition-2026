from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "data" / "raw" / "ecos_table_matches.csv"
TABLES = ROOT / "data" / "raw" / "ecos_tables.csv"
OUT = ROOT / "data" / "raw" / "ecos_matched_items.csv"
BASE_URL = "https://ecos.bok.or.kr/api"
CYCLES = ["M", "Q", "A", "D", "S", "Y"]
MANUAL_STAT_CODES = {
    "200Y001",  # 국민계정 주요지표
    "200Y002",
    "901Y009",  # 소비자물가
    "901Y033",  # 산업생산
    "901Y067",  # 소매판매
    "901Y066",  # 설비투자
    "901Y104",  # 건설기성
    "901Y027",  # 통관수출입
    "151Y005",  # 예금은행 대출/연체 후보
    "111Y001",  # 통화량 후보
    "101Y001",
}


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
        if result.get("CODE") == "INFO-200":
            return 0, []
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
        with urlopen(ecos_url(service, key, start_no, start_no + page_size - 1, *parts), timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        total, page = unwrap(payload, service)
        rows.extend(page)
        if not page:
            break
        start_no += page_size
        time.sleep(0.05)
    return rows


def main() -> int:
    stat_codes = set(MANUAL_STAT_CODES)
    with MATCHES.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["rank"]) <= 5 and int(row["score"]) >= 4:
                stat_codes.add(row["stat_code"])
    table_names = {}
    with TABLES.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            table_names[row["STAT_CODE"]] = row.get("STAT_NAME")

    key = get_key()
    out_rows: list[dict[str, Any]] = []
    for idx, stat_code in enumerate(sorted(stat_codes), start=1):
        print(f"[{idx}/{len(stat_codes)}] {stat_code} {table_names.get(stat_code, '')}")
        for cycle in CYCLES:
            try:
                items = fetch_paged("StatisticItemList", key, stat_code, cycle)
            except RuntimeError:
                items = []
            for item in items:
                row = {
                    "stat_code": stat_code,
                    "stat_name": table_names.get(stat_code, ""),
                    "cycle_requested": cycle,
                }
                row.update(item)
                out_rows.append(row)
    fields = sorted({key for row in out_rows for key in row})
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved {OUT} rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
