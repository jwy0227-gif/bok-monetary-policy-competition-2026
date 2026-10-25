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
OUT = ROOT / "data" / "raw" / "ecos_curated_items.csv"
BASE_URL = "https://ecos.bok.or.kr/api"
STAT_CYCLES = {
    "722Y001": ["D"],
    "721Y001": ["M"],
    "817Y002": ["D"],
    "102Y002": ["M"],
    "161Y002": ["M"],
    "161Y004": ["M"],
    "161Y006": ["M"],
    "101Y004": ["M"],
    "161Y008": ["M"],
    "171Y004": ["M"],
    "151Y001": ["Q"],
    "151Y004": ["Q"],
    "901Y009": ["M"],
    "901Y010": ["M"],
    "404Y014": ["M"],
    "404Y015": ["M"],
    "511Y002": ["M"],
    "511Y003": ["M"],
    "512Y007": ["M"],
    "901Y033": ["M"],
    "901Y034": ["M"],
    "901Y035": ["M"],
    "901Y025": ["M"],
    "901Y038": ["M"],
    "901Y066": ["M"],
    "901Y100": ["M"],
    "901Y104": ["M"],
    "802Y001": ["D"],
    "901Y014": ["M"],
    "901Y015": ["M"],
    "901Y055": ["M"],
    "901Y118": ["M"],
    "901Y119": ["M"],
    "901Y120": ["M"],
    "901Y121": ["M"],
    "901Y092": ["M"],
    "403Y001": ["M"],
    "403Y002": ["M"],
    "403Y003": ["M"],
    "403Y004": ["M"],
    "200Y102": ["Q"],
    "200Y104": ["Q"],
    "200Y108": ["Q"],
    "200Y110": ["Q"],
    "200Y107": ["Q"],
    "301Y013": ["M"],
    "731Y003": ["D"],
    "731Y006": ["M"],
    "732Y001": ["M"],
    "311Y004": ["Q"],
    "311Y006": ["Q"],
    "901Y062": ["M"],
    "901Y063": ["M"],
    "901Y113": ["M"],
    "901Y114": ["M"],
    "902Y003": ["M"],
    "902Y006": ["M"],
    "902Y008": ["M"],
    "902Y015": ["Q"],
}


def get_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit("Set ECOS_API_KEY or BOK_API_KEY.")
    return key


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts if part is not None)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def fetch(service: str, key: str, *parts: str) -> list[dict[str, Any]]:
    url = ecos_url(service, key, 1, 10000, *parts)
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "RESULT" in payload:
        result = payload["RESULT"]
        if result.get("CODE") == "INFO-200":
            return []
        raise RuntimeError(f"{result.get('CODE')}: {result.get('MESSAGE')}")
    body = payload.get(service) or {}
    rows = body.get("row") or []
    if isinstance(rows, dict):
        rows = [rows]
    return rows


def main() -> int:
    key = get_key()
    out_rows: list[dict[str, Any]] = []
    for stat_code, cycles in STAT_CYCLES.items():
        for cycle in cycles:
            print(f"{stat_code} {cycle}")
            try:
                rows = fetch("StatisticItemList", key, stat_code, cycle)
            except RuntimeError as exc:
                print(f"  skipped: {exc}")
                rows = []
            for row in rows:
                item = {"stat_code": stat_code, "cycle_requested": cycle}
                item.update(row)
                out_rows.append(item)
            time.sleep(0.08)
    fields = sorted({key for row in out_rows for key in row})
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved {OUT} rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
