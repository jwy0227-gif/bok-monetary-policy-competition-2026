from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "svar_main_level_stl_cpi_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "robustness"

MAIN_IRF_PATH = REPORT_DIR / "irf_oil_shock_cholesky_var3.csv"
ROBUST_IRF_PATH = REPORT_DIR / "irf_oil_shock_cholesky_var3_swap_cpi_exp.csv"
ROBUST_SUMMARY_PATH = REPORT_DIR / "irf_oil_shock_summary_var3_swap_cpi_exp.csv"
COMPARISON_PATH = REPORT_DIR / "robust_cholesky_swap_cpi_exp_comparison.csv"
CHOLESKY_PATH = REPORT_DIR / "cholesky_matrix_var3_swap_cpi_exp.csv"
REPORT_PATH = REPORT_DIR / "robust_cholesky_swap_cpi_exp.md"
FIGURE_PATH = FIGURE_DIR / "oil_to_expected_irf_cholesky_order_comparison.svg"

VAR_LAG = 3
HORIZON = 24

MAIN_ORDER = ["dln_oil_pct", "dln_fx_pct", "dln_cpi_stl_pct", "expected_inflation"]
SWAP_ORDER = ["dln_oil_pct", "dln_fx_pct", "expected_inflation", "dln_cpi_stl_pct"]

LABELS = {
    "dln_oil_pct": "국제유가",
    "dln_fx_pct": "환율",
    "dln_cpi_stl_pct": "CPI",
    "expected_inflation": "기대인플레이션",
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


def estimate_irf(order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    y = df.set_index("date")[order]
    result = VAR(y).fit(VAR_LAG)
    irf = result.irf(HORIZON)
    stderr = irf.stderr(orth=True)
    lower = irf.orth_irfs - 1.96 * stderr
    upper = irf.orth_irfs + 1.96 * stderr

    shock_idx = order.index("dln_oil_pct")
    rows = []
    for h in range(HORIZON + 1):
        for response_idx, response in enumerate(order):
            rows.append(
                {
                    "horizon": h,
                    "impulse": "dln_oil_pct",
                    "response": response,
                    "response_label": LABELS[response],
                    "irf": float(irf.orth_irfs[h, response_idx, shock_idx]),
                    "lower_95": float(lower[h, response_idx, shock_idx]),
                    "upper_95": float(upper[h, response_idx, shock_idx]),
                    "unit": "%p",
                }
            )

    cholesky = pd.DataFrame(
        np.linalg.cholesky(result.sigma_u),
        index=order,
        columns=[f"shock_{col}" for col in order],
    )
    oil_shock_size = float(cholesky.loc["dln_oil_pct", "shock_dln_oil_pct"])
    return pd.DataFrame(rows), cholesky, oil_shock_size


def summarize(irf_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response in ["expected_inflation", "dln_cpi_stl_pct", "dln_fx_pct"]:
        sub = irf_df[(irf_df["response"] == response) & (irf_df["horizon"] >= 0)].copy()
        max_row = sub.loc[sub["irf"].idxmax()]
        rows.append(
            {
                "shock_to_response": f"유가 → {LABELS[response]}",
                "max_response": float(max_row["irf"]),
                "max_response_horizon": int(max_row["horizon"]),
                "significant_periods_95": significance_periods(sub),
                "unit": "%p",
            }
        )
    return pd.DataFrame(rows)


def build_comparison(main_irf: pd.DataFrame, robust_irf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response in ["expected_inflation", "dln_cpi_stl_pct", "dln_fx_pct"]:
        main = main_irf[main_irf["response"] == response].sort_values("horizon")
        robust = robust_irf[robust_irf["response"] == response].sort_values("horizon")
        diff = robust["irf"].to_numpy() - main["irf"].to_numpy()
        rows.append(
            {
                "response": LABELS[response],
                "max_abs_difference": float(np.max(np.abs(diff))),
                "mean_abs_difference": float(np.mean(np.abs(diff))),
                "main_max_response": float(main["irf"].max()),
                "robust_max_response": float(robust["irf"].max()),
                "main_max_horizon": int(main.loc[main["irf"].idxmax(), "horizon"]),
                "robust_max_horizon": int(robust.loc[robust["irf"].idxmax(), "horizon"]),
                "main_significant_periods_95": significance_periods(main),
                "robust_significant_periods_95": significance_periods(robust),
            }
        )
    return pd.DataFrame(rows)


def svg_comparison(main_irf: pd.DataFrame, robust_irf: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    main = main_irf[main_irf["response"] == "expected_inflation"].sort_values("horizon")
    robust = robust_irf[robust_irf["response"] == "expected_inflation"].sort_values("horizon")

    width = 980
    height = 430
    left = 78
    right = 34
    top = 76
    bottom = 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    ymin = min(float(main[["lower_95", "irf"]].min().min()), float(robust[["lower_95", "irf"]].min().min()), 0.0)
    ymax = max(float(main[["upper_95", "irf"]].max().max()), float(robust[["upper_95", "irf"]].max().max()), 0.0)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
    ymin -= pad
    ymax += pad

    def x_pos(h: int) -> float:
        return left + h * plot_width / HORIZON

    def y_pos(v: float) -> float:
        return top + plot_height - (v - ymin) * plot_height / (ymax - ymin)

    def points(df: pd.DataFrame, col: str) -> str:
        return " ".join(f"{x_pos(int(row.horizon)):.2f},{y_pos(float(getattr(row, col))):.2f}" for row in df.itertuples())

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.zero{stroke:#333;stroke-width:1.2}.main{fill:none;stroke:#CC79A7;stroke-width:2.8}.robust{fill:none;stroke:#0072B2;stroke-width:2.3;stroke-dasharray:7 5}.band{fill:#CC79A7;fill-opacity:.14;stroke:none}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">Oil Shock → Expected Inflation IRF</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#555">Main order: Oil → FX → CPI → Expected inflation / Robust order: Oil → FX → Expected inflation → CPI</text>',
    ]

    for tick in range(5):
        y = top + tick * plot_height / 4
        value = ymax - tick * (ymax - ymin) / 4
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.3f}</text>')

    zero_y = y_pos(0.0)
    lines.append(f'<line class="zero" x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}"/>')
    upper = [(x_pos(int(row.horizon)), y_pos(float(row.upper_95))) for row in main.itertuples()]
    lower = [(x_pos(int(row.horizon)), y_pos(float(row.lower_95))) for row in main.iloc[::-1].itertuples()]
    polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
    lines.append(f'<polygon class="band" points="{polygon}"/>')
    lines.append(f'<polyline class="main" points="{points(main, "irf")}"/>')
    lines.append(f'<polyline class="robust" points="{points(robust, "irf")}"/>')

    for h in [0, 6, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 24}" font-size="11" text-anchor="middle">T+{h}</text>')

    lines.extend(
        [
            f'<line class="main" x1="{left}" y1="{height - 24}" x2="{left + 34}" y2="{height - 24}"/>',
            f'<text x="{left + 44}" y="{height - 20}" font-size="12">Main order</text>',
            f'<line class="robust" x1="{left + 150}" y1="{height - 24}" x2="{left + 184}" y2="{height - 24}"/>',
            f'<text x="{left + 194}" y="{height - 20}" font-size="12">Swapped order</text>',
            "</svg>",
        ]
    )
    FIGURE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if MAIN_IRF_PATH.exists():
        main_irf = pd.read_csv(MAIN_IRF_PATH)
    else:
        main_irf, _, _ = estimate_irf(MAIN_ORDER)

    robust_irf, cholesky, oil_shock_size = estimate_irf(SWAP_ORDER)
    robust_summary = summarize(robust_irf)
    comparison = build_comparison(main_irf, robust_irf)

    robust_irf.to_csv(ROBUST_IRF_PATH, index=False, encoding="utf-8-sig")
    robust_summary.to_csv(ROBUST_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    cholesky.to_csv(CHOLESKY_PATH, encoding="utf-8-sig")
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    svg_comparison(main_irf, robust_irf)

    exp_comp = comparison[comparison["response"] == "기대인플레이션"].iloc[0]
    cpi_comp = comparison[comparison["response"] == "CPI"].iloc[0]

    lines = [
        "# 촐레스키 순서 강건성 검정: CPI와 기대인플레이션 순서 교체",
        "",
        "## 설정",
        "",
        f"- 사용 데이터: `{DATA_PATH.relative_to(ROOT).as_posix()}`",
        "- 추정 기간: 2010-02 ~ 2026-02",
        f"- 모형: VAR({VAR_LAG})",
        "- 최종 모형 순서: 국제유가 → 환율 → CPI → 기대인플레이션",
        "- 강건성 순서: 국제유가 → 환율 → 기대인플레이션 → CPI",
        f"- 유가 1표준편차 구조충격 크기: {oil_shock_size:.4f}%p",
        "",
        "## 강건성 IRF 요약",
        "",
        "| 충격 → 반응 | 최대 반응 | 최대 시점 | 95% 유의 기간 |",
        "| --- | ---: | ---: | --- |",
    ]
    for _, row in robust_summary.iterrows():
        lines.append(
            f"| {row['shock_to_response']} | {row['max_response']:.4f}{row['unit']} | "
            f"T+{int(row['max_response_horizon'])} | {row['significant_periods_95']} |"
        )

    lines.extend(
        [
            "",
            "## 최종 모형과 순서 교체 모형 비교",
            "",
            "| 반응 변수 | 최대 절대 차이 | 최종 모형 최대 반응 | 교체 모형 최대 반응 | 최종 모형 최대 시점 | 교체 모형 최대 시점 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['response']} | {row['max_abs_difference']:.10f} | {row['main_max_response']:.4f} | "
            f"{row['robust_max_response']:.4f} | T+{int(row['main_max_horizon'])} | "
            f"T+{int(row['robust_max_horizon'])} |"
        )

    lines.extend(
        [
            "",
            "## 판정",
            "",
            (
                "CPI와 기대인플레이션의 순서를 바꿔도 유가 충격에 대한 기대인플레이션 IRF는 "
                f"최대 절대 차이 {exp_comp['max_abs_difference']:.10f}%p로 사실상 동일합니다. "
                "CPI 반응도 동일하게 유지됩니다."
            ),
            "",
            (
                "이는 유가가 촐레스키 순서의 첫 번째 변수로 유지되어 유가 구조충격의 식별이 "
                "뒤쪽 변수들의 상대적 순서에 영향을 거의 받지 않기 때문입니다. 따라서 핵심 결과인 "
                "`유가 충격 → 기대인플레이션 상승`은 CPI와 기대인플레이션의 동시적 순서 가정에 대해 "
                "질적으로 강건하다고 해석할 수 있습니다."
            ),
            "",
            "## 파일",
            "",
            f"- 교체 순서 IRF: `{ROBUST_IRF_PATH.relative_to(ROOT).as_posix()}`",
            f"- 교체 순서 요약표: `{ROBUST_SUMMARY_PATH.relative_to(ROOT).as_posix()}`",
            f"- 최종 모형 비교표: `{COMPARISON_PATH.relative_to(ROOT).as_posix()}`",
            f"- 교체 순서 촐레스키 행렬: `{CHOLESKY_PATH.relative_to(ROOT).as_posix()}`",
            f"- 비교 그래프: `{FIGURE_PATH.relative_to(ROOT).as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Saved: {ROBUST_IRF_PATH}")
    print(f"Saved: {ROBUST_SUMMARY_PATH}")
    print(f"Saved: {COMPARISON_PATH}")
    print(f"Saved: {CHOLESKY_PATH}")
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
