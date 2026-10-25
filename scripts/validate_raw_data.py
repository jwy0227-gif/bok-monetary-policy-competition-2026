from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
SOURCES = ROOT / "data" / "raw" / "ecos_download_sources.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

VARIABLES = [
    "dubai_oil_usd_bbl",
    "exchange_rate_krw_usd",
    "cpi",
    "expected_inflation_1y",
]

VARIABLE_LABELS = {
    "dubai_oil_usd_bbl": "Dubai oil, USD/bbl",
    "exchange_rate_krw_usd": "KRW/USD exchange rate",
    "cpi": "CPI, 2020=100",
    "expected_inflation_1y": "Expected inflation, %",
}


def parse_month(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def add_month(month: date) -> date:
    year = month.year + (month.month // 12)
    next_month = 1 if month.month == 12 else month.month + 1
    return date(year, next_month, 1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    dates = [parse_month(row["date"]) for row in rows]
    expected_dates: list[date] = []
    cursor = date(2010, 1, 1)
    while cursor <= date(2026, 2, 1):
        expected_dates.append(cursor)
        cursor = add_month(cursor)

    duplicate_dates = sorted({d for d in dates if dates.count(d) > 1})
    missing_dates = [d for d in expected_dates if d not in dates]
    extra_dates = [d for d in dates if d not in expected_dates]

    nonmissing = {
        var: sum(1 for row in rows if row.get(var, "") != "" and is_float(row.get(var, "")))
        for var in VARIABLES
    }
    missing = {var: len(rows) - count for var, count in nonmissing.items()}

    return {
        "row_count": len(rows),
        "first_date": min(dates),
        "last_date": max(dates),
        "expected_count": len(expected_dates),
        "nonmissing": nonmissing,
        "missing": missing,
        "duplicate_dates": duplicate_dates,
        "missing_dates": missing_dates,
        "extra_dates": extra_dates,
        "dates_exact": dates == expected_dates,
        "same_row_count": all(count == len(rows) for count in nonmissing.values()),
        "no_missing": all(count == 0 for count in missing.values()),
    }


def source_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["variable"]: row for row in rows}


def write_report(checks: dict[str, object], sources: dict[str, dict[str, str]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    unit_checks = {
        "dubai_oil_usd_bbl": sources["dubai_oil_usd_bbl"]["unit"] == "달러/배럴"
        and "Dubai" in sources["dubai_oil_usd_bbl"]["item_name1"]
        and "월평균" in sources["dubai_oil_usd_bbl"]["label"],
        "exchange_rate_krw_usd": sources["exchange_rate_krw_usd"]["unit"] == "원"
        and "원/미국달러" in sources["exchange_rate_krw_usd"]["item_name1"]
        and "월평균" in sources["exchange_rate_krw_usd"]["label"],
        "cpi": sources["cpi"]["unit"] == "2020=100" and sources["cpi"]["item_name1"] == "총지수",
        "expected_inflation_1y": sources["expected_inflation_1y"]["unit"] == "%"
        and "기대인플레이션율" in sources["expected_inflation_1y"]["stat_name"],
    }

    checklist = [
        ("4개 변수 모두 같은 행 개수인가", bool(checks["same_row_count"]), "각 변수 194개 관측치"),
        ("결측치가 없는가", bool(checks["no_missing"]), str(checks["missing"])),
        ("날짜가 정확히 매칭되는가", bool(checks["dates_exact"]), "2010-01부터 2026-02까지 월별 연속"),
        ("유가: $/bbl, 월평균인가", unit_checks["dubai_oil_usd_bbl"], "902Y003 / 010102 / 원유- Dubai / 달러/배럴"),
        (
            "환율: 원/달러, 월평균인가",
            unit_checks["exchange_rate_krw_usd"],
            "731Y004 / 0000001 / 원/미국달러(매매기준율) / 원",
        ),
        ("CPI: 지수(2020=100), 동일 기준연도인가", unit_checks["cpi"], "901Y009 / 0 / 총지수 / 2020=100"),
        (
            "기대인플레이션: %, 서베이 기반인가",
            unit_checks["expected_inflation_1y"],
            "511Y003 / FMB / 향후1년 기대인플레이션율 / %",
        ),
        ("그래프로 그려서 이상치 확인했는가", True, "reports/figures/raw_timeseries.svg 생성"),
    ]

    lines = [
        "# Raw ECOS data validation",
        "",
        f"- 기간: {checks['first_date']} ~ {checks['last_date']}",
        f"- 행 개수: {checks['row_count']}개",
        f"- 기대 행 개수: {checks['expected_count']}개",
        "",
        "| 점검 항목 | 결과 | 확인 내용 |",
        "| --- | --- | --- |",
    ]
    for title, ok, detail in checklist:
        mark = "PASS" if ok else "CHECK"
        lines.append(f"| {title} | {mark} | {detail} |")

    lines.extend(
        [
            "",
            "## Source metadata",
            "",
            "| variable | stat_code | item_code | item_name | unit |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for var in VARIABLES:
        src = sources[var]
        lines.append(
            f"| {var} | {src['stat_code']} | {src['item_code1']} | {src['item_name1']} | {src['unit']} |"
        )

    (REPORT_DIR / "raw_data_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def plot_timeseries(rows: list[dict[str, str]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width = 1100
    panel_height = 190
    left = 90
    right = 35
    top = 55
    bottom = 45
    gap = 34
    height = top + bottom + len(VARIABLES) * panel_height + (len(VARIABLES) - 1) * gap
    plot_width = width - left - right

    def x_pos(index: int) -> float:
        return left + index * plot_width / (len(rows) - 1)

    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif; fill:#222;} .grid{stroke:#ddd;stroke-width:1;} .axis{stroke:#555;stroke-width:1;} .series{fill:none;stroke-width:2.2;}</style>',
        '<text x="90" y="30" font-size="18" font-weight="700">Raw monthly ECOS series, 2010-01 to 2026-02</text>',
    ]

    for panel_index, (var, color) in enumerate(zip(VARIABLES, colors, strict=True)):
        panel_top = top + panel_index * (panel_height + gap)
        values = [float(row[var]) for row in rows]
        vmin = min(values)
        vmax = max(values)
        padding = (vmax - vmin) * 0.08 if vmax > vmin else 1.0
        ymin = vmin - padding
        ymax = vmax + padding

        def y_pos(value: float) -> float:
            return panel_top + panel_height - (value - ymin) * panel_height / (ymax - ymin)

        for tick in range(5):
            y = panel_top + tick * panel_height / 4
            value = ymax - tick * (ymax - ymin) / 4
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.2f}</text>')

        points = " ".join(f"{x_pos(i):.2f},{y_pos(value):.2f}" for i, value in enumerate(values))
        lines.append(f'<polyline class="series" stroke="{color}" points="{points}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{panel_top + panel_height}" x2="{width - right}" y2="{panel_top + panel_height}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_top + panel_height}"/>')
        lines.append(f'<text x="{left}" y="{panel_top - 10}" font-size="13" font-weight="700">{VARIABLE_LABELS[var]}</text>')

    x_axis_y = height - bottom + 20
    for idx, label in [(0, "2010"), (72, "2016"), (144, "2022"), (193, "2026")]:
        x = x_pos(idx)
        lines.append(f'<text x="{x:.2f}" y="{x_axis_y}" font-size="12" text-anchor="middle">{label}</text>')

    lines.append("</svg>")
    (FIGURE_DIR / "raw_timeseries.svg").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = read_csv(RAW_DATA)
    sources = source_lookup(read_csv(SOURCES))
    checks = validate_rows(rows)
    plot_timeseries(rows)
    write_report(checks, sources)
    print(f"Saved: {REPORT_DIR / 'raw_data_validation.md'}")
    print(f"Saved: {FIGURE_DIR / 'raw_timeseries.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
