from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_OUT = ROOT / "data" / "raw" / "long_term_inflation_expectations_202501_202605_ecos.csv"
REPORT_CSV = ROOT / "reports" / "long_term_inflation_expectations_202501_202605.csv"
REPORT_TSV = ROOT / "reports" / "long_term_inflation_expectations_202501_202605_for_docs.tsv"
REPORT_MD = ROOT / "reports" / "long_term_inflation_expectations_202501_202605.md"
SOURCE_CSV = ROOT / "reports" / "long_term_inflation_expectations_sources.csv"

BASE_URL = "https://ecos.bok.or.kr/api"
STAT_CODE = "511Y003"
CYCLE = "M"
START = "202501"
END = "202605"

SERIES = {
    "expected_inflation_3y": {"item_code": "FMC", "label": "3년후 기대인플레이션율"},
    "expected_inflation_5y": {"item_code": "FMD", "label": "5년후 기대인플레이션율"},
}


def get_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit("ECOS_API_KEY 환경변수를 설정한 뒤 실행하세요.")
    return key


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def fetch_series(key: str, item_code: str) -> list[dict]:
    url = ecos_url("StatisticSearch", key, 1, 1000, STAT_CODE, CYCLE, START, END, item_code)
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
        "# 장기 기대인플레이션율 추이",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    lines.extend(["", "주: 소비자동향조사 기준. 단위는 %.", "자료: 한국은행 ECOS."])
    return "\n".join(lines) + "\n"


def main() -> int:
    key = get_key()
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    by_time: dict[str, dict[str, object]] = {}
    source_rows = []
    for variable, spec in SERIES.items():
        rows = fetch_series(key, spec["item_code"])
        for row in rows:
            time_key = row["TIME"]
            by_time.setdefault(time_key, {"date": f"{time_key[:4]}-{time_key[4:6]}-01"})
            by_time[time_key][variable] = float(row["DATA_VALUE"])
            source_rows.append(
                {
                    "variable": variable,
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
    raw.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")

    target_months = pd.date_range("2025-01-01", "2026-05-01", freq="MS")
    rows = []
    for date in target_months:
        sub = raw[raw["date"] == date]
        if sub.empty:
            rows.append(
                {
                    "월": month_label(date),
                    "3년후 기대인플레이션율(%)": "미공표",
                    "5년후 기대인플레이션율(%)": "미공표",
                }
            )
        else:
            item = sub.iloc[0]
            rows.append(
                {
                    "월": month_label(date),
                    "3년후 기대인플레이션율(%)": f"{item['expected_inflation_3y']:.1f}",
                    "5년후 기대인플레이션율(%)": f"{item['expected_inflation_5y']:.1f}",
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    table.to_csv(REPORT_TSV, index=False, sep="\t", encoding="utf-8-sig")
    REPORT_MD.write_text(dataframe_to_markdown(table), encoding="utf-8-sig")
    pd.DataFrame(source_rows).drop_duplicates().to_csv(SOURCE_CSV, index=False, encoding="utf-8-sig")

    for path in [RAW_OUT, REPORT_CSV, REPORT_TSV, REPORT_MD, SOURCE_CSV]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
