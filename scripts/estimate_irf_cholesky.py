from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "svar_main_level_stl_cpi_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "irf"

IRF_PATH = REPORT_DIR / "irf_oil_shock_cholesky_var3.csv"
SUMMARY_PATH = REPORT_DIR / "irf_oil_shock_summary_var3.csv"
CHOLESKY_PATH = REPORT_DIR / "cholesky_matrix_var3.csv"
REPORT_PATH = REPORT_DIR / "irf_oil_shock_cholesky_var3.md"

VAR_LAG = 3
HORIZON = 24

VAR_COLUMNS = ["dln_oil_pct", "dln_fx_pct", "dln_cpi_stl_pct", "expected_inflation"]
KOREAN_LABELS = {
    "dln_oil_pct": "국제유가",
    "dln_fx_pct": "환율",
    "dln_cpi_stl_pct": "CPI",
    "expected_inflation": "기대인플레이션",
}
UNIT_LABELS = {
    "dln_oil_pct": "%p",
    "dln_fx_pct": "%p",
    "dln_cpi_stl_pct": "%p",
    "expected_inflation": "%p",
}


def significance_periods(df: pd.DataFrame) -> str:
    sig = df[(df["lower_95"] > 0) | (df["upper_95"] < 0)]["horizon"].astype(int).tolist()
    if not sig:
        return "없음"

    ranges = []
    start = prev = sig[0]
    for value in sig[1:]:
        if value == prev + 1:
            prev = value
        else:
            ranges.append((start, prev))
            start = prev = value
    ranges.append((start, prev))

    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(f"T+{start}")
        else:
            parts.append(f"T+{start}~T+{end}")
    return ", ".join(parts)


def svg_irf_plot(irf_df: pd.DataFrame, out_path: Path) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    responses = ["dln_fx_pct", "dln_cpi_stl_pct", "expected_inflation"]
    colors = {
        "dln_fx_pct": "#009E73",
        "dln_cpi_stl_pct": "#D55E00",
        "expected_inflation": "#CC79A7",
    }
    width = 980
    panel_height = 230
    left = 80
    right = 30
    top = 82
    bottom = 44
    gap = 44
    height = top + bottom + len(responses) * panel_height + (len(responses) - 1) * gap
    plot_width = width - left - right

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}.zero{stroke:#333;stroke-width:1.2}.irf{fill:none;stroke-width:2.4}.band{fill-opacity:.18;stroke:none}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">Oil shock IRF, Cholesky VAR({VAR_LAG})</text>',
        f'<text x="{left}" y="48" font-size="12" fill="#555">Impulse: 1 s.d. positive Dubai oil shock, 95% asymptotic CI</text>',
    ]

    def x_pos(h: int) -> float:
        return left + h * plot_width / HORIZON

    for panel_idx, response in enumerate(responses):
        sub = irf_df[irf_df["response"] == response].copy()
        panel_top = top + panel_idx * (panel_height + gap)
        ymin = float(sub[["irf", "lower_95", "upper_95"]].min().min())
        ymax = float(sub[["irf", "lower_95", "upper_95"]].max().max())
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
        pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
        ymin -= pad
        ymax += pad

        def y_pos(value: float) -> float:
            return panel_top + panel_height - (value - ymin) * panel_height / (ymax - ymin)

        for tick in range(5):
            y = panel_top + tick * panel_height / 4
            value = ymax - tick * (ymax - ymin) / 4
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.3f}</text>')

        zero_y = y_pos(0.0)
        lines.append(f'<line class="zero" x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}"/>')
        lines.append(f'<text x="{left}" y="{panel_top - 12}" font-size="14" font-weight="700">{KOREAN_LABELS[response]}</text>')

        upper = [(x_pos(int(row.horizon)), y_pos(float(row.upper_95))) for row in sub.itertuples()]
        lower = [(x_pos(int(row.horizon)), y_pos(float(row.lower_95))) for row in sub.iloc[::-1].itertuples()]
        polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
        points = " ".join(f"{x_pos(int(row.horizon)):.2f},{y_pos(float(row.irf)):.2f}" for row in sub.itertuples())
        lines.append(f'<polygon class="band" fill="{colors[response]}" points="{polygon}"/>')
        lines.append(f'<polyline class="irf" stroke="{colors[response]}" points="{points}"/>')

        for h in [0, 6, 12, 18, 24]:
            x = x_pos(h)
            lines.append(f'<line class="grid" x1="{x:.2f}" y1="{panel_top}" x2="{x:.2f}" y2="{panel_top + panel_height}"/>')
            lines.append(f'<text x="{x:.2f}" y="{panel_top + panel_height + 20}" font-size="11" text-anchor="middle">{h}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    y = df.set_index("date")[VAR_COLUMNS]
    result = VAR(y).fit(VAR_LAG)
    irf = result.irf(HORIZON)
    stderr = irf.stderr(orth=True)
    lower = irf.orth_irfs - 1.96 * stderr
    upper = irf.orth_irfs + 1.96 * stderr

    shock_idx = VAR_COLUMNS.index("dln_oil_pct")
    rows = []
    for h in range(HORIZON + 1):
        for response_idx, response in enumerate(VAR_COLUMNS):
            rows.append(
                {
                    "horizon": h,
                    "impulse": "dln_oil_pct",
                    "response": response,
                    "response_label": KOREAN_LABELS[response],
                    "irf": float(irf.orth_irfs[h, response_idx, shock_idx]),
                    "lower_95": float(lower[h, response_idx, shock_idx]),
                    "upper_95": float(upper[h, response_idx, shock_idx]),
                    "unit": UNIT_LABELS[response],
                }
            )

    irf_df = pd.DataFrame(rows)
    irf_df.to_csv(IRF_PATH, index=False, encoding="utf-8-sig")

    summary_rows = []
    for response in ["expected_inflation", "dln_cpi_stl_pct", "dln_fx_pct"]:
        sub = irf_df[(irf_df["response"] == response) & (irf_df["horizon"] >= 0)].copy()
        max_idx = sub["irf"].idxmax()
        max_row = sub.loc[max_idx]
        summary_rows.append(
            {
                "shock_to_response": f"유가 → {KOREAN_LABELS[response]}",
                "max_response": float(max_row["irf"]),
                "max_response_horizon": int(max_row["horizon"]),
                "significant_periods_95": significance_periods(sub),
                "unit": UNIT_LABELS[response],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    cholesky = pd.DataFrame(
        np.linalg.cholesky(result.sigma_u),
        index=VAR_COLUMNS,
        columns=[f"shock_{col}" for col in VAR_COLUMNS],
    )
    oil_shock_size = float(cholesky.loc["dln_oil_pct", "shock_dln_oil_pct"])
    cholesky.to_csv(CHOLESKY_PATH, encoding="utf-8-sig")

    plot_path = FIGURE_DIR / "oil_shock_irf_var3_stl_cpi.svg"
    svg_irf_plot(irf_df, plot_path)

    lines = [
        "# 촐레스키 IRF: 유가 충격",
        "",
        f"- 사용 데이터: `{DATA_PATH.relative_to(ROOT).as_posix()}`",
        f"- 모형: VAR({VAR_LAG})",
        "- 변수 순서: 국제유가 → 환율 → CPI → 기대인플레이션",
        "- CPI: STL 계절조정 CPI의 월간 로그차분",
        f"- 유가 1표준편차 충격 크기: {oil_shock_size:.4f}%p, 약 {oil_shock_size:.2f}% 월간 상승",
        f"- IRF 기간: 0~{HORIZON}개월",
        "- 신뢰구간: asymptotic standard error 기반 95%",
        "",
        "## 핵심 결과 요약",
        "",
        "| 충격 → 반응 | 최대 반응치 | 최대 반응 시점 | 95% 유의 기간 |",
        "| --- | ---: | ---: | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['shock_to_response']} | {row['max_response']:.4f}{row['unit']} | "
            f"T+{int(row['max_response_horizon'])}개월 | {row['significant_periods_95']} |"
        )

    lines.extend(
        [
            "",
            "## 파일",
            "",
            f"- IRF 전체 표: `{IRF_PATH.relative_to(ROOT).as_posix()}`",
            f"- 요약표: `{SUMMARY_PATH.relative_to(ROOT).as_posix()}`",
            f"- 촐레스키 행렬: `{CHOLESKY_PATH.relative_to(ROOT).as_posix()}`",
            f"- IRF 그래프: `{plot_path.relative_to(ROOT).as_posix()}`",
            "",
            "## 해석 메모",
            "",
            "IRF는 유가 방정식 잔차의 1표준편차 구조충격을 촐레스키 분해로 식별한 결과입니다. 기대인플레이션 반응은 수준 변수이므로 단위는 %p로 해석합니다.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Saved: {IRF_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {CHOLESKY_PATH}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
