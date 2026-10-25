from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "data" / "processed" / "svar_main_level_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "eda_final"
REPORT_PATH = REPORT_DIR / "final_series_stationarity_check.md"

SERIES = {
    "dln_oil_pct": {
        "title": "Δln(두바이유) × 100",
        "unit": "%",
        "color": "#0072B2",
        "stationarity_note": "평균 0 근처를 중심으로 큰 충격들이 일시적으로 나타나는 형태입니다. 뚜렷한 장기 추세는 보이지 않습니다.",
    },
    "dln_fx_pct": {
        "title": "Δln(원/달러 환율) × 100",
        "unit": "%",
        "color": "#009E73",
        "stationarity_note": "대부분 0 근처에서 등락하며 일부 환율 급변 구간만 튀는 형태입니다. 평균회귀적 패턴으로 보입니다.",
    },
    "dln_cpi_pct": {
        "title": "Δln(CPI) × 100",
        "unit": "%",
        "color": "#D55E00",
        "stationarity_note": "수준 CPI의 추세는 제거됐지만 2021~2023년 물가상승기에는 평균보다 높은 구간이 이어집니다. 완전히 깨끗한 정상성보다는 경계적 정상성으로 보는 것이 적절합니다.",
    },
    "expected_inflation": {
        "title": "기대인플레이션율 수준",
        "unit": "%",
        "color": "#CC79A7",
        "stationarity_note": "2020년 저점과 2022년 고점 이후 하락하는 저빈도 움직임이 남아 있습니다. 메인 분석에서는 해석력을 위해 수준을 쓰되, 차분 모형을 강건성 분석으로 확인해야 합니다.",
    },
}


def read_rows() -> list[dict[str, str]]:
    with MAIN_PATH.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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
    mean_value = statistics.fmean(values)
    vmin = min(min(values), mean_value)
    vmax = max(max(values), mean_value)
    pad = (vmax - vmin) * 0.08 if vmax > vmin else 1.0
    ymin = vmin - pad
    ymax = vmax + pad

    def x_pos(index: int) -> float:
        return left + index * plot_width / (len(rows) - 1)

    def y_pos(value: float) -> float:
        return top + plot_height - (value - ymin) * plot_height / (ymax - ymin)

    def fmt(value: float) -> str:
        return f"{value:,.2f}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}.series{fill:none;stroke-width:2.4}.mean{stroke:#222;stroke-width:1.5;stroke-dasharray:6 5}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">{spec["title"]}</text>',
        f'<text x="{left}" y="48" font-size="12" fill="#555">2010-02 ~ 2026-02, 평균선 포함, 단위: {spec["unit"]}</text>',
    ]

    for tick in range(6):
        y = top + tick * plot_height / 5
        value = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{fmt(value)}</text>')

    for idx, year in [(0, "2010"), (35, "2013"), (71, "2016"), (107, "2019"), (143, "2022"), (179, "2025")]:
        x = x_pos(idx)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 25}" font-size="12" text-anchor="middle">{year}</text>')

    mean_y = y_pos(mean_value)
    points = " ".join(f"{x_pos(i):.2f},{y_pos(value):.2f}" for i, value in enumerate(values))
    lines.append(f'<polyline class="series" stroke="{spec["color"]}" points="{points}"/>')
    lines.append(f'<line class="mean" x1="{left}" y1="{mean_y:.2f}" x2="{width - right}" y2="{mean_y:.2f}"/>')
    lines.append(f'<text x="{width - right - 5}" y="{mean_y - 6:.2f}" font-size="11" text-anchor="end">mean={fmt(mean_value)}</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    lines.append("</svg>")

    out = FIGURE_DIR / f"{var}.svg"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def summary(rows: list[dict[str, str]], var: str) -> dict[str, float]:
    values = [float(row[var]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def write_report(rows: list[dict[str, str]], figures: dict[str, Path], summaries: dict[str, dict[str, float]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 최종 VAR 투입 변수 시계열 점검",
        "",
        "자료: `data/processed/svar_main_level_201002_202602.csv`",
        "",
        "## 결론",
        "",
        "유가 변동률과 환율 변동률은 평균 주위를 맴도는 정상 시계열 패턴이 뚜렷합니다. CPI 월간 상승률은 수준 추세는 제거됐지만 2021~2023년 고물가 구간의 지속성이 남아 있어 ADF 결과처럼 경계적입니다. 기대인플레이션 수준은 해석상 메인 분석에 사용하되, 평균 주위를 완전히 안정적으로 맴돈다고 보기는 어려워 차분 모형을 강건성 분석으로 확인하는 것이 필요합니다.",
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
            "## 요약 통계",
            "",
            "| 변수 | 평균 | 표준편차 | 최솟값 | 최댓값 | 판단 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for var, spec in SERIES.items():
        s = summaries[var]
        lines.append(
            f"| {spec['title']} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {spec['stationarity_note']} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    rows = read_rows()
    figures: dict[str, Path] = {}
    summaries: dict[str, dict[str, float]] = {}
    for var, spec in SERIES.items():
        figures[var] = svg_line_plot(rows, var, spec)
        summaries[var] = summary(rows, var)
    write_report(rows, figures, summaries)
    print(f"Saved: {REPORT_PATH}")
    for path in figures.values():
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
