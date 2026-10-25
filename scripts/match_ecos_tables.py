from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "raw" / "data_workbook_features.json"
TABLES = ROOT / "data" / "raw" / "ecos_tables.csv"
OUT = ROOT / "data" / "raw" / "ecos_table_matches.csv"


ALIASES = {
    "기준금리": ["기준금리", "한국은행 기준금리"],
    "정책금리": ["정책금리", "국제금리", "Federal Funds", "미국"],
    "콜금리": ["콜금리"],
    "본원통화": ["본원통화"],
    "M1": ["M1", "통화"],
    "M2": ["M2", "통화"],
    "Lf": ["Lf", "유동성"],
    "소비자물가": ["소비자물가"],
    "생활물가": ["생활물가", "소비자물가"],
    "근원물가": ["근원", "소비자물가"],
    "생산자물가": ["생산자물가"],
    "기대인플레이션": ["기대인플레이션"],
    "BEI": ["BEI", "물가연동", "국고채"],
    "GDP": ["국내총생산", "GDP"],
    "GNI": ["국민총소득", "GNI"],
    "소비자심리지수": ["소비자동향", "소비자심리지수"],
    "CSI": ["소비자동향", "CSI"],
    "BSI": ["기업경기", "BSI"],
    "경상수지": ["국제수지", "경상수지"],
    "상품수지": ["국제수지", "상품수지"],
    "서비스수지": ["국제수지", "서비스수지"],
    "국고채": ["시장금리", "국고채"],
    "회사채": ["시장금리", "회사채"],
    "CD": ["시장금리", "CD"],
    "CP": ["시장금리", "CP"],
    "환율": ["환율"],
    "가계신용": ["가계신용"],
    "외환보유액": ["외환보유액"],
    "외채": ["외채"],
    "두바이유": ["국제상품가격", "Dubai"],
    "WTI": ["국제상품가격", "WTI"],
    "브렌트유": ["국제상품가격", "Brent"],
}


def tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^0-9A-Za-z가-힣]+", text) if tok]


def query_terms(name: str, source: str | None) -> list[str]:
    terms = tokens(name)
    for key, values in ALIASES.items():
        if key in name:
            terms.extend(values)
    if source and "한국은행" in source:
        terms.append("한국은행")
    return sorted(set(terms), key=len, reverse=True)


def score_table(terms: list[str], row: dict[str, str]) -> int:
    haystack = " ".join(
        str(row.get(key, "")) for key in ["STAT_NAME", "STAT_CODE", "ORG_NAME", "P_STAT_CODE"]
    )
    score = 0
    for term in terms:
        if term and term in haystack:
            score += max(1, min(5, len(term)))
    return score


def main() -> int:
    sheets = json.loads(FEATURES.read_text(encoding="utf-8"))
    with TABLES.open(encoding="utf-8-sig", newline="") as f:
        tables = list(csv.DictReader(f))
    rows: list[dict[str, object]] = []
    for sheet in sheets:
        for feature in sheet["features"]:
            name = feature["name"]
            source = feature["values"].get("출처")
            terms = query_terms(name, source)
            matches = sorted(
                ((score_table(terms, table), table) for table in tables),
                key=lambda item: item[0],
                reverse=True,
            )
            for rank, (score, table) in enumerate([m for m in matches if m[0] > 0][:8], start=1):
                rows.append(
                    {
                        "sheet": sheet["sheet"],
                        "feature": name,
                        "source": source,
                        "unit": feature["unit"],
                        "rank": rank,
                        "score": score,
                        "stat_code": table.get("STAT_CODE"),
                        "stat_name": table.get("STAT_NAME"),
                        "cycle": table.get("CYCLE"),
                        "org_name": table.get("ORG_NAME"),
                    }
                )
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["sheet", "feature", "source", "unit", "rank", "score", "stat_code", "stat_name", "cycle", "org_name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(OUT)
    for row in rows:
        if row["rank"] == 1:
            print(f"{row['feature']}\t{row['stat_code']}\t{row['stat_name']}\t(score={row['score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
