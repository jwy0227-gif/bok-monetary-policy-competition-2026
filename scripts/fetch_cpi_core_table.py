from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_OUT = ROOT / "data" / "raw" / "cpi_core_202401_202605_ecos.csv"
REPORT_CSV = ROOT / "reports" / "cpi_core_table_202501_202605.csv"
REPORT_TSV = ROOT / "reports" / "cpi_core_table_202501_202605_for_docs.tsv"
REPORT_MD = ROOT / "reports" / "cpi_core_table_202501_202605.md"

BASE_URL = "https://ecos.bok.or.kr/api"
START = "202401"
END = "202605"

SERIES = {
    "cpi_index": {
        "stat_code": "901Y009",
        "cycle": "M",
        "item_code": "0",
        "label": "소비자물가지수",
    },
    "core_cpi_index": {
        "stat_code": "901Y010",
        "cycle": "M",
        "item_code": "DB",
        "label": "근원물가지수(식료품 및 에너지 제외)",
    },
}


def get_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit("ECOS_API_KEY 환경변수를 설정한 뒤 실행하세요.")
    return key


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def fetch_series(key: str, stat_code: str, cycle: str, item_code: str) -> list[dict]:
    url = ecos_url("StatisticSearch", key, 1, 1000, stat_code, cycle, START, END, item_code)
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "RESULT" in payload:
        result = payload["RESULT"]
        raise RuntimeError(f"{result.get('CODE')}: {result.get('MESSAGE')}")
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if isinstance(rows, dict):
        rows = [rows]
    return rows


def month_label(date: pd.Timestamp) -> str:
    return f"{date.year}.{date.month:02d}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "# 소비자물가 및 근원물가 추이",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    lines.extend(
        [
            "",
            "주: 근원물가는 식료품 및 에너지 제외 지수 기준. 전년동월대비 상승률은 지수로부터 계산.",
            "자료: 한국은행 ECOS, 저자 계산.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    key = get_key()
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    by_time: dict[str, dict[str, object]] = {}
    source_rows = []
    for name, spec in SERIES.items():
        rows = fetch_series(key, spec["stat_code"], spec["cycle"], spec["item_code"])
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
                    "item_code": row.get("ITEM_CODE1"),
                    "item_name": row.get("ITEM_NAME1"),
                    "unit": row.get("UNIT_NAME"),
                }
            )

    raw = pd.DataFrame([by_time[key] for key in sorted(by_time)])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date")
    raw["cpi_yoy_pct"] = (raw["cpi_index"] / raw["cpi_index"].shift(12) - 1) * 100
    raw["core_cpi_yoy_pct"] = (raw["core_cpi_index"] / raw["core_cpi_index"].shift(12) - 1) * 100
    raw.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")

    target_months = pd.date_range("2025-01-01", "2026-05-01", freq="MS")
    rows = []
    for date in target_months:
        sub = raw[raw["date"] == date]
        if sub.empty:
            rows.append(
                {
                    "월": month_label(date),
                    "소비자물가지수(2020=100)": "미공표",
                    "소비자물가 상승률(전년동월대비, %)": "미공표",
                    "근원물가지수(식료품 및 에너지 제외, 2020=100)": "미공표",
                    "근원물가 상승률(전년동월대비, %)": "미공표",
                }
            )
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "월": month_label(date),
                "소비자물가지수(2020=100)": f"{row['cpi_index']:.2f}",
                "소비자물가 상승률(전년동월대비, %)": f"{row['cpi_yoy_pct']:.2f}",
                "근원물가지수(식료품 및 에너지 제외, 2020=100)": f"{row['core_cpi_index']:.2f}",
                "근원물가 상승률(전년동월대비, %)": f"{row['core_cpi_yoy_pct']:.2f}",
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    table.to_csv(REPORT_TSV, index=False, sep="\t", encoding="utf-8-sig")
    REPORT_MD.write_text(dataframe_to_markdown(table), encoding="utf-8-sig")

    sources = ROOT / "reports" / "cpi_core_table_sources.csv"
    with sources.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variable", "label", "stat_code", "stat_name", "item_code", "item_name", "unit"],
        )
        writer.writeheader()
        writer.writerows({tuple(sorted(row.items())): row for row in source_rows}.values())

    print(RAW_OUT)
    print(REPORT_CSV)
    print(REPORT_TSV)
    print(REPORT_MD)
    print(sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
