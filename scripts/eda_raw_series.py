from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "eda_raw"
REPORT_PATH = REPORT_DIR / "raw_series_eda.md"

SERIES = {
    "dubai_oil_usd_bbl": {
        "title": "두바이유 월평균",
        "unit": "달러/배럴",
        "color": "#0072B2",
    },
    "exchange_rate_krw_usd": {
        "title": "원/달러 월평균 환율",
        "unit": "원",
        "color": "#009E73",
    },
    "cpi": {
        "title": "소비자물가지수",
        "unit": "2020=100",
        "color": "#D55E00",
    },
    "expected_inflation_1y": {
        "title": "향후 1년 기대인플레이션율",
        "unit": "%",
        "color": "#CC79A7",
    },
}


def read_rows() -> list[dict[str, str]]:
    with RAW_PATH.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def pct_change(prev: float, curr: float) -> float:
    return (curr / prev - 1.0) * 100.0


def fmt(value: float) -> str:
    return f"{value:,.2f}"


def svg_line_plot(rows: list[dict[str, str]], var: str, spec: dict[str, str]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 420
    left = 82
    right = 32
    top = 54
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row[var]) for row in rows]
    vmin = min(values)
    vmax = max(values)
    pad = (vmax - vmin) * 0.08 if vmax > vmin else 1.0
    ymin = vmin - pad
    ymax = vmax + pad

    def x_pos(index: int) -> float:
        return left + index * plot_width / (len(rows) - 1)

    def y_pos(value: float) -> float:
        return top + plot_height - (value - ymin) * plot_height / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}.series{fill:none;stroke-width:2.4}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">{spec["title"]}</text>',
        f'<text x="{left}" y="48" font-size="12" fill="#555">2010-01 ~ 2026-02, 단위: {spec["unit"]}</text>',
    ]

    for tick in range(6):
        y = top + tick * plot_height / 5
        value = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{fmt(value)}</text>')

    for idx, year in [(0, "2010"), (36, "2013"), (72, "2016"), (108, "2019"), (144, "2022"), (180, "2025")]:
        x = x_pos(idx)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 25}" font-size="12" text-anchor="middle">{year}</text>')

    points = " ".join(f"{x_pos(i):.2f},{y_pos(value):.2f}" for i, value in enumerate(values))
    lines.append(f'<polyline class="series" stroke="{spec["color"]}" points="{points}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    lines.append("</svg>")

    out = FIGURE_DIR / f"{var}.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def summarize_series(rows: list[dict[str, str]], var: str) -> dict[str, object]:
    values = [float(row[var]) for row in rows]
    dates = [row["date"] for row in rows]
    min_idx = min(range(len(values)), key=lambda i: values[i])
    max_idx = max(range(len(values)), key=lambda i: values[i])

    diffs = []
    for i in range(1, len(values)):
        diffs.append(
            {
                "date": dates[i],
                "prev_date": dates[i - 1],
                "diff": values[i] - values[i - 1],
                "abs_diff": abs(values[i] - values[i - 1]),
                "pct_change": pct_change(values[i - 1], values[i]) if values[i - 1] != 0 else 0.0,
            }
        )
    biggest_jumps = sorted(diffs, key=lambda x: float(x["abs_diff"]), reverse=True)[:5]
    return {
        "min_date": dates[min_idx],
        "min_value": values[min_idx],
        "max_date": dates[max_idx],
        "max_value": values[max_idx],
        "first_value": values[0],
        "last_value": values[-1],
        "biggest_jumps": biggest_jumps,
    }


def write_report(rows: list[dict[str, str]], figures: dict[str, Path], summaries: dict[str, dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 원자료 시계열 EDA",
        "",
        "기간: 2010-01 ~ 2026-02",
        "",
        "## 그래프",
        "",
    ]
    for var, spec in SERIES.items():
        rel = figures[var].relative_to(ROOT).as_posix()
        lines.append(f"- {spec['title']}: `{rel}`")

    lines.extend(
        [
            "",
            "## 요약 통계와 큰 점프",
            "",
            "| 변수 | 최솟값 | 최댓값 | 시작값 | 끝값 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for var, spec in SERIES.items():
        s = summaries[var]
        lines.append(
            f"| {spec['title']} | {fmt(float(s['min_value']))} ({s['min_date'][:7]}) | "
            f"{fmt(float(s['max_value']))} ({s['max_date'][:7]}) | "
            f"{fmt(float(s['first_value']))} | {fmt(float(s['last_value']))} |"
        )

    lines.extend(["", "## 육안 점검 메모", ""])
    notes = {
        "dubai_oil_usd_bbl": "2020년 4월 코로나19 충격 구간의 급락과 2021~2022년 회복 및 고유가 구간이 뚜렷합니다. 큰 점프가 있지만 국제유가의 실제 시장 변동과 부합합니다.",
        "exchange_rate_krw_usd": "2014년 저점 이후 완만한 상승 흐름이 있고, 2022년과 2024년 말 원화 약세 구간이 눈에 띕니다. 단절이나 비정상적 단일월 오류는 보이지 않습니다.",
        "cpi": "장기적으로 우상향하는 지수형 추세가 매우 뚜렷합니다. 수준값은 비정상성이 자연스럽고, 월간 로그차분으로 변환하는 것이 적절합니다.",
        "expected_inflation_1y": "2020년 저점 이후 2022년 물가 급등기에 크게 상승했다가 이후 하락합니다. 설문 기반 기대인플레이션답게 계단형 움직임이 일부 보이지만 자료 오류로 보이는 끊김은 없습니다.",
    }
    for var, spec in SERIES.items():
        lines.append(f"### {spec['title']}")
        lines.append("")
        lines.append(notes[var])
        lines.append("")
        lines.append("| 순위 | 구간 | 변화폭 | 변화율 |")
        lines.append("| ---: | --- | ---: | ---: |")
        for idx, jump in enumerate(summaries[var]["biggest_jumps"], start=1):
            lines.append(
                f"| {idx} | {jump['prev_date'][:7]} → {jump['date'][:7]} | "
                f"{float(jump['diff']):,.2f} | {float(jump['pct_change']):,.2f}% |"
            )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    rows = read_rows()
    figures: dict[str, Path] = {}
    summaries: dict[str, dict[str, object]] = {}
    for var, spec in SERIES.items():
        figures[var] = svg_line_plot(rows, var, spec)
        summaries[var] = summarize_series(rows, var)
    write_report(rows, figures, summaries)
    print(f"Saved: {REPORT_PATH}")
    for path in figures.values():
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
