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


BASE_URL = "https://ecos.bok.or.kr/api"
START = "201001"
END = "202602"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


SERIES = {
    "dubai_oil_usd_bbl": {
        "stat_code": "902Y003",
        "cycle": "M",
        "item_code1": "010102",
        "label": "두바이유 월평균",
    },
    "exchange_rate_krw_usd": {
        "stat_code": "731Y004",
        "cycle": "M",
        "item_code1": "0000001",
        "label": "원/달러 환율 월평균",
    },
    "cpi": {
        "stat_code": "901Y009",
        "cycle": "M",
        "item_code1": "0",
        "label": "소비자물가지수 총지수",
    },
    "expected_inflation_1y": {
        "stat_code": "511Y003",
        "cycle": "M",
        "item_code1": "FMB",
        "label": "향후1년 기대인플레이션율",
    },
}

KEYWORDS = ("두바이", "국제유가", "기대인플레이션", "소비자동향")


def get_api_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit(
            "ECOS_API_KEY 환경변수가 없습니다. PowerShell에서 "
            '$env:ECOS_API_KEY="발급받은_인증키" 설정 후 다시 실행하세요.'
        )
    return key


def request_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts if part is not None)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def unwrap(payload: dict[str, Any], service: str) -> tuple[int, list[dict[str, Any]]]:
    if "RESULT" in payload:
        result = payload["RESULT"]
        raise RuntimeError(f"{result.get('CODE')}: {result.get('MESSAGE')}")
    body = payload.get(service)
    if not body:
        return 0, []
    rows = body.get("row") or []
    if isinstance(rows, dict):
        rows = [rows]
    return int(body.get("list_total_count", len(rows))), rows


def fetch_paged(service: str, key: str, *parts: str, page_size: int = 100) -> list[dict[str, Any]]:
    start_no = 1
    rows: list[dict[str, Any]] = []
    total = None
    while total is None or start_no <= total:
        end_no = start_no + page_size - 1
        payload = request_json(ecos_url(service, key, start_no, end_no, *parts))
        total, page_rows = unwrap(payload, service)
        rows.extend(page_rows)
        if not page_rows:
            break
        start_no += page_size
        time.sleep(0.15)
    return rows


def fetch_series(key: str, spec: dict[str, str]) -> list[dict[str, Any]]:
    return fetch_paged(
        "StatisticSearch",
        key,
        spec["stat_code"],
        spec["cycle"],
        START,
        END,
        spec["item_code1"],
    )


def save_metadata_candidates(key: str) -> None:
    tables = fetch_paged("StatisticTableList", key)
    matched_tables = [
        row
        for row in tables
        if any(keyword in str(row.get("STAT_NAME", "")) for keyword in KEYWORDS)
    ]

    candidates: list[dict[str, Any]] = []
    for table in matched_tables:
        stat_code = table["STAT_CODE"]
        try:
            items = fetch_paged("StatisticItemList", key, stat_code, "M")
        except RuntimeError:
            items = []
        if not items:
            candidates.append(table)
            continue
        for item in items:
            row = dict(table)
            row.update(
                {
                    "ITEM_CODE": item.get("ITEM_CODE"),
                    "ITEM_NAME": item.get("ITEM_NAME"),
                    "CYCLE": item.get("CYCLE") or table.get("CYCLE"),
                }
            )
            candidates.append(row)

    out = RAW_DIR / "ecos_metadata_candidates.csv"
    fieldnames = sorted({key for row in candidates for key in row})
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def save_monthly_data(key: str) -> None:
    by_time: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []

    for name, spec in SERIES.items():
        rows = fetch_series(key, spec)
        for row in rows:
            time_key = row["TIME"]
            by_time.setdefault(time_key, {"date": f"{time_key[:4]}-{time_key[4:6]}-01"})
            by_time[time_key][name] = float(row["DATA_VALUE"])
            source_rows.append(
                {
                    "variable": name,
                    "label": spec["label"],
                    "stat_code": row.get("STAT_CODE"),
                    "stat_name": row.get("STAT_NAME"),
                    "item_code1": row.get("ITEM_CODE1"),
                    "item_name1": row.get("ITEM_NAME1"),
                    "unit": row.get("UNIT_NAME"),
                }
            )

    out = RAW_DIR / "ecos_monthly_201001_202602.csv"
    fields = ["date", *SERIES.keys()]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for time_key in sorted(by_time):
            writer.writerow({field: by_time[time_key].get(field, "") for field in fields})

    source_out = RAW_DIR / "ecos_download_sources.csv"
    unique_sources = {tuple(sorted(row.items())): row for row in source_rows}.values()
    with source_out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variable", "label", "stat_code", "stat_name", "item_code1", "item_name1", "unit"],
        )
        writer.writeheader()
        writer.writerows(unique_sources)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    key = get_api_key()
    save_metadata_candidates(key)
    save_monthly_data(key)
    print(f"Saved: {RAW_DIR / 'ecos_monthly_201001_202602.csv'}")
    print(f"Saved: {RAW_DIR / 'ecos_metadata_candidates.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
