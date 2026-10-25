from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://ecos.bok.or.kr/api"

CPI_CORE_RAW = ROOT / "data" / "raw" / "cpi_core_202401_202605_ecos.csv"
LONG_EXP_RAW = ROOT / "data" / "raw" / "long_term_inflation_expectations_202501_202605_ecos.csv"

ONE_YEAR_RAW = ROOT / "data" / "raw" / "one_year_inflation_expectations_202501_202605_ecos.csv"
OUT_CSV = ROOT / "reports" / "inflation_expectation_gap_table_202501_202605.csv"
OUT_TSV = ROOT / "reports" / "inflation_expectation_gap_table_202501_202605_for_docs.tsv"
OUT_MD = ROOT / "reports" / "inflation_expectation_gap_table_202501_202605.md"

START = "202501"
END = "202605"


def get_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit("ECOS_API_KEY 환경변수를 설정한 뒤 실행하세요.")
    return key


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def fetch_one_year_expectation(key: str) -> pd.DataFrame:
    url = ecos_url("StatisticSearch", key, 1, 1000, "511Y003", "M", START, END, "FMB")
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "RESULT" in payload:
        result = payload["RESULT"]
        raise RuntimeError(f"{result.get('CODE')}: {result.get('MESSAGE')}")
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if isinstance(rows, dict):
        rows = [rows]
    data = pd.DataFrame(
        {
            "date": pd.to_datetime([f"{row['TIME'][:4]}-{row['TIME'][4:6]}-01" for row in rows]),
            "expected_inflation_1y": [float(row["DATA_VALUE"]) for row in rows],
        }
    )
    data.to_csv(ONE_YEAR_RAW, index=False, encoding="utf-8-sig")
    return data


def fmt(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "미공표"
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "# 기대인플레이션과 물가상승률 격차",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    lines.extend(
        [
            "",
            "주: 격차는 앞의 변수에서 뒤의 변수를 차감한 값이며 단위는 %p. CPI 및 근원 CPI는 전년동월대비 상승률.",
            "자료: 한국은행 ECOS, 저자 계산.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    key = get_key()
    one = fetch_one_year_expectation(key)
    cpi = pd.read_csv(CPI_CORE_RAW, parse_dates=["date"])
    long = pd.read_csv(LONG_EXP_RAW, parse_dates=["date"])

    data = pd.DataFrame({"date": pd.date_range("2025-01-01", "2026-05-01", freq="MS")})
    data = data.merge(one, on="date", how="left")
    data = data.merge(long, on="date", how="left")
    data = data.merge(cpi[["date", "cpi_yoy_pct", "core_cpi_yoy_pct"]], on="date", how="left")

    data["gap_1y_minus_cpi"] = data["expected_inflation_1y"] - data["cpi_yoy_pct"]
    data["gap_1y_minus_core"] = data["expected_inflation_1y"] - data["core_cpi_yoy_pct"]
    data["gap_3y_minus_1y"] = data["expected_inflation_3y"] - data["expected_inflation_1y"]
    data["gap_5y_minus_1y"] = data["expected_inflation_5y"] - data["expected_inflation_1y"]
    data["gap_3y_minus_cpi"] = data["expected_inflation_3y"] - data["cpi_yoy_pct"]
    data["gap_5y_minus_cpi"] = data["expected_inflation_5y"] - data["cpi_yoy_pct"]
    data["gap_3y_minus_core"] = data["expected_inflation_3y"] - data["core_cpi_yoy_pct"]
    data["gap_5y_minus_core"] = data["expected_inflation_5y"] - data["core_cpi_yoy_pct"]

    table = pd.DataFrame(
        {
            "월": data["date"].dt.strftime("%Y.%m"),
            "1년 기대(%)": data["expected_inflation_1y"].map(lambda x: fmt(x, 1)),
            "3년 기대(%)": data["expected_inflation_3y"].map(lambda x: fmt(x, 1)),
            "5년 기대(%)": data["expected_inflation_5y"].map(lambda x: fmt(x, 1)),
            "CPI 상승률(%)": data["cpi_yoy_pct"].map(lambda x: fmt(x, 2)),
            "근원 CPI 상승률(%)": data["core_cpi_yoy_pct"].map(lambda x: fmt(x, 2)),
            "1년-CPI(%p)": data["gap_1y_minus_cpi"].map(lambda x: fmt(x, 2)),
            "1년-근원(%p)": data["gap_1y_minus_core"].map(lambda x: fmt(x, 2)),
            "3년-1년(%p)": data["gap_3y_minus_1y"].map(lambda x: fmt(x, 2)),
            "5년-1년(%p)": data["gap_5y_minus_1y"].map(lambda x: fmt(x, 2)),
            "3년-CPI(%p)": data["gap_3y_minus_cpi"].map(lambda x: fmt(x, 2)),
            "5년-CPI(%p)": data["gap_5y_minus_cpi"].map(lambda x: fmt(x, 2)),
            "3년-근원(%p)": data["gap_3y_minus_core"].map(lambda x: fmt(x, 2)),
            "5년-근원(%p)": data["gap_5y_minus_core"].map(lambda x: fmt(x, 2)),
        }
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    table.to_csv(OUT_TSV, index=False, sep="\t", encoding="utf-8-sig")
    OUT_MD.write_text(dataframe_to_markdown(table), encoding="utf-8-sig")

    for path in [ONE_YEAR_RAW, OUT_CSV, OUT_TSV, OUT_MD]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
