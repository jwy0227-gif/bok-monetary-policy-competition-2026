from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_HIST_PATH = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
IRF_PATH = ROOT / "reports" / "irf_oil_shock_cholesky_var3.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "simulation"

OIL_PATH_OUT = REPORT_DIR / "scenario_quarterly_2026_oil_paths_main_model.csv"
SIM_OUT = REPORT_DIR / "scenario_quarterly_2026_expected_inflation_main_model.csv"
SUMMARY_OUT = REPORT_DIR / "scenario_quarterly_2026_expected_inflation_summary_main_model.csv"
REPORT_OUT = REPORT_DIR / "scenario_quarterly_2026_main_model.md"
EXP_FIGURE_OUT = FIGURE_DIR / "scenario_quarterly_2026_expected_inflation_main_model.svg"
OIL_FIGURE_OUT = FIGURE_DIR / "scenario_quarterly_2026_oil_path_main_model.svg"

OIL_SHOCK_SIGMA = 8.045945115677503
HORIZON = 24
START_DATE = pd.Timestamp("2026-03-01")
BASELINE_EXPECTED_INFLATION = 2.6
ANCHORING_THRESHOLD_LEVEL = 3.0

SCENARIOS = [
    {
        "scenario": "기준 시나리오",
        "oil_2026q2": 100.0,
        "oil_2026q3": 90.0,
        "oil_2026q4": 87.0,
        "oil_2027": 87.0,
        "color": "#0072B2",
    },
    {
        "scenario": "고유가 장기화 시나리오",
        "oil_2026q2": 105.0,
        "oil_2026q3": 105.0,
        "oil_2026q4": 105.0,
        "oil_2027": 105.0,
        "color": "#D55E00",
    },
    {
        "scenario": "유가 안정 시나리오",
        "oil_2026q2": 95.0,
        "oil_2026q3": 85.0,
        "oil_2026q4": 80.0,
        "oil_2027": 80.0,
        "color": "#009E73",
    },
]

ACTUAL_OIL = {
    pd.Timestamp("2026-03-01"): 126.71,
    pd.Timestamp("2026-04-01"): 105.30,
}


def scenario_price(spec: dict[str, float | str], month: pd.Timestamp, base_oil: float) -> float:
    if month in ACTUAL_OIL:
        return ACTUAL_OIL[month]
    if pd.Timestamp("2026-05-01") <= month <= pd.Timestamp("2026-06-01"):
        return float(spec["oil_2026q2"])
    if pd.Timestamp("2026-07-01") <= month <= pd.Timestamp("2026-09-01"):
        return float(spec["oil_2026q3"])
    if pd.Timestamp("2026-10-01") <= month <= pd.Timestamp("2026-12-01"):
        return float(spec["oil_2026q4"])
    return float(spec["oil_2027"])


def build_oil_paths() -> tuple[pd.DataFrame, float]:
    hist = pd.read_csv(RAW_HIST_PATH, parse_dates=["date"])
    base_oil = float(hist.loc[hist["date"] == pd.Timestamp("2026-02-01"), "dubai_oil_usd_bbl"].iloc[0])
    months = pd.date_range(START_DATE, periods=HORIZON + 1, freq="MS")

    rows = []
    for spec in SCENARIOS:
        prices = [scenario_price(spec, month, base_oil) for month in months]
        previous_prices = [base_oil] + prices[:-1]
        for horizon, (month, price, previous) in enumerate(zip(months, prices, previous_prices, strict=True)):
            dln_oil_pct = 100 * np.log(price / previous)
            rows.append(
                {
                    "scenario": spec["scenario"],
                    "horizon": horizon,
                    "date": month,
                    "oil_price_usd_bbl": price,
                    "previous_oil_price_usd_bbl": previous,
                    "dln_oil_pct": dln_oil_pct,
                    "shock_size_sigma": dln_oil_pct / OIL_SHOCK_SIGMA,
                }
            )
    return pd.DataFrame(rows), base_oil


def simulate(paths: pd.DataFrame) -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp = irf[irf["response"] == "expected_inflation"].sort_values("horizon").set_index("horizon")
    exp_irf = exp["irf"].reindex(range(HORIZON + 1)).to_numpy()
    exp_lower = exp["lower_95"].reindex(range(HORIZON + 1)).to_numpy()
    exp_upper = exp["upper_95"].reindex(range(HORIZON + 1)).to_numpy()

    rows = []
    for spec in SCENARIOS:
        scenario = str(spec["scenario"])
        shocks = paths[paths["scenario"] == scenario].sort_values("horizon")["shock_size_sigma"].to_numpy()
        for h in range(HORIZON + 1):
            response = 0.0
            lower = 0.0
            upper = 0.0
            for j, shock in enumerate(shocks):
                if h - j >= 0:
                    response += shock * exp_irf[h - j]
                    low_component = shock * exp_lower[h - j]
                    high_component = shock * exp_upper[h - j]
                    lower += min(low_component, high_component)
                    upper += max(low_component, high_component)
            rows.append(
                {
                    "scenario": scenario,
                    "horizon": h,
                    "date": START_DATE + pd.DateOffset(months=h),
                    "expected_inflation_response_pp": response,
                    "expected_inflation_response_lower_95_pp": lower,
                    "expected_inflation_response_upper_95_pp": upper,
                    "expected_inflation_level_pct": BASELINE_EXPECTED_INFLATION + response,
                    "expected_inflation_level_lower_95_pct": BASELINE_EXPECTED_INFLATION + lower,
                    "expected_inflation_level_upper_95_pct": BASELINE_EXPECTED_INFLATION + upper,
                    "above_anchoring_threshold": BASELINE_EXPECTED_INFLATION + response >= ANCHORING_THRESHOLD_LEVEL,
                }
            )
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        scenario = str(spec["scenario"])
        sub = sim[sim["scenario"] == scenario].copy()
        max_row = sub.loc[sub["expected_inflation_response_pp"].idxmax()]
        threshold_rows = sub[sub["above_anchoring_threshold"]]
        rows.append(
            {
                "scenario": scenario,
                "max_response_pp": float(max_row["expected_inflation_response_pp"]),
                "max_response_horizon": int(max_row["horizon"]),
                "max_response_date": max_row["date"],
                "max_level_pct": float(max_row["expected_inflation_level_pct"]),
                "max_level_lower_95_pct": float(max_row["expected_inflation_level_lower_95_pct"]),
                "max_level_upper_95_pct": float(max_row["expected_inflation_level_upper_95_pct"]),
                "first_threshold_horizon": int(threshold_rows["horizon"].iloc[0]) if not threshold_rows.empty else pd.NA,
                "first_threshold_date": threshold_rows["date"].iloc[0] if not threshold_rows.empty else pd.NaT,
                "level_t": float(sub.loc[sub["horizon"] == 0, "expected_inflation_level_pct"].iloc[0]),
                "level_t3": float(sub.loc[sub["horizon"] == 3, "expected_inflation_level_pct"].iloc[0]),
                "level_t6": float(sub.loc[sub["horizon"] == 6, "expected_inflation_level_pct"].iloc[0]),
                "level_t12": float(sub.loc[sub["horizon"] == 12, "expected_inflation_level_pct"].iloc[0]),
                "level_t24": float(sub.loc[sub["horizon"] == 24, "expected_inflation_level_pct"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def svg_line_plot(
    data: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    subtitle: str,
    y_label: str,
    threshold: float | None = None,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 520
    left = 82
    right = 34
    top = 70
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    ymin = float(data[value_col].min())
    ymax = float(data[value_col].max())
    if threshold is not None:
        ymin = min(ymin, threshold)
        ymax = max(ymax, threshold)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 1.0
    ymin -= pad
    ymax += pad

    def x_pos(h: int) -> float:
        return left + h * plot_width / HORIZON

    def y_pos(value: float) -> float:
        return top + plot_height - (value - ymin) * plot_height / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.threshold{stroke:#111;stroke-width:1.6;stroke-dasharray:7 5}.series{fill:none;stroke-width:2.8}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">{title}</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#555">{subtitle}</text>',
        f'<text x="20" y="{top + 12}" font-size="12" fill="#555">{y_label}</text>',
    ]

    for tick in range(6):
        y = top + tick * plot_height / 5
        value = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.2f}</text>')

    if threshold is not None:
        y = y_pos(threshold)
        lines.append(f'<line class="threshold" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{width - right - 4}" y="{y - 7:.2f}" font-size="12" text-anchor="end">3.0% 임계선</text>')

    for h in [0, 3, 6, 9, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 36}" font-size="11" text-anchor="middle">T+{h}</text>')

    legend_y = height - 18
    for idx, spec in enumerate(SCENARIOS):
        scenario = str(spec["scenario"])
        sub = data[data["scenario"] == scenario].sort_values("horizon")
        points = " ".join(
            f"{x_pos(int(row.horizon)):.2f},{y_pos(float(getattr(row, value_col))):.2f}"
            for row in sub.itertuples()
        )
        color = str(spec["color"])
        lines.append(f'<polyline class="series" stroke="{color}" points="{points}"/>')
        lx = left + idx * 270
        lines.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 36}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx + 44}" y="{legend_y + 4}" font-size="12">{scenario}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report(paths: pd.DataFrame, summary: pd.DataFrame, base_oil: float) -> None:
    shock_rows = paths[paths["dln_oil_pct"].abs() > 1e-12].copy()
    q2_shocks = shock_rows[shock_rows["date"] == pd.Timestamp("2026-04-01")]

    lines = [
        "# 2026년 분기 유가 시나리오 기반 기대인플레이션 시뮬레이션",
        "",
        "## 모형",
        "",
        "- 최종 메인 모형을 사용했습니다.",
        "- 표본: 2010-02 ~ 2026-02",
        "- 모형: VAR(3)",
        "- 변수: 유가 전월대비 로그차분, 환율 전월대비 로그차분, STL 계절조정 CPI 전월대비 로그차분, 기대인플레이션 수준",
        "- 촐레스키 순서: 국제유가 → 환율 → CPI → 기대인플레이션",
        f"- 유가 1표준편차 구조충격: {OIL_SHOCK_SIGMA:.4f}%p",
        "",
        "## 시나리오 변환 가정",
        "",
        f"- 기준 유가: 2026년 2월 두바이유 {base_oil:.2f}달러.",
        "- 2026년 3월은 126.71달러, 4월은 105.30달러를 실제값으로 사용했습니다.",
        "- 2026년 5~6월은 2/4분기 전제값, 7~9월은 3/4분기 전제값, 10월 이후와 2027년은 4/4분기 전제값이 유지된다고 두었습니다.",
        f"- 기준 기대인플레이션은 2026년 2월 실제값 {BASELINE_EXPECTED_INFLATION:.1f}%입니다.",
        f"- 기대인플레이션 앵커링 훼손 임계선은 {ANCHORING_THRESHOLD_LEVEL:.1f}%로 표시했습니다.",
        "",
        "## 2026년 3월 최초 실제 충격",
        "",
        "| 시나리오 | 2026년 3월 유가 | 전월 대비 로그변화율 | 충격 크기 |",
        "| --- | ---: | ---: | ---: |",
    ]
    first_shocks = paths[paths["date"] == pd.Timestamp("2026-03-01")]
    for _, row in first_shocks.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['oil_price_usd_bbl']:.2f}달러 | "
            f"{row['dln_oil_pct']:.4f}% | {row['shock_size_sigma']:.4f}σ |"
        )

    lines.extend(
        [
            "",
            "## 기대인플레이션 시뮬레이션 결과",
            "",
            "| 시나리오 | 최대 반응 | 최대 기대인플레이션 수준 | 최대 수준 95% CI | 최대 시점 | 3% 최초 도달 | T | T+3 | T+6 | T+12 | T+24 |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        threshold = "도달 안 함" if pd.isna(row["first_threshold_horizon"]) else f"T+{int(row['first_threshold_horizon'])}"
        lines.append(
            f"| {row['scenario']} | {row['max_response_pp']:.4f}%p | {row['max_level_pct']:.4f}% | "
            f"[{row['max_level_lower_95_pct']:.4f}%, {row['max_level_upper_95_pct']:.4f}%] | "
            f"T+{int(row['max_response_horizon'])} | {threshold} | "
            f"{row['level_t']:.4f}% | {row['level_t3']:.4f}% | {row['level_t6']:.4f}% | "
            f"{row['level_t12']:.4f}% | {row['level_t24']:.4f}% |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "세 시나리오 모두 2026년 2월 유가 68.51달러에서 2/4분기 전제 유가로 상승하는 구간에서 기대인플레이션이 상승합니다. 다만 3/4분기와 4/4분기에 유가가 내려가는 기준 시나리오와 유가 안정 시나리오는 이후 음의 유가 충격이 들어오면서 반응이 일부 상쇄됩니다.",
            "",
            "반대로 고유가 장기화 시나리오는 유가가 105달러에서 유지되므로 초기 상승 충격 이후 기대인플레이션 반응이 더 오래 유지됩니다.",
            "",
            "## 파일",
            "",
            f"- 유가 경로: `{OIL_PATH_OUT.relative_to(ROOT).as_posix()}`",
            f"- 기대인플레이션 시뮬레이션: `{SIM_OUT.relative_to(ROOT).as_posix()}`",
            f"- 요약표: `{SUMMARY_OUT.relative_to(ROOT).as_posix()}`",
            f"- 유가 시나리오 그래프: `{OIL_FIGURE_OUT.relative_to(ROOT).as_posix()}`",
            f"- 기대인플레이션 그래프: `{EXP_FIGURE_OUT.relative_to(ROOT).as_posix()}`",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    paths, base_oil = build_oil_paths()
    sim = simulate(paths)
    summary = summarize(sim)

    paths.to_csv(OIL_PATH_OUT, index=False, encoding="utf-8-sig")
    sim.to_csv(SIM_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    svg_line_plot(
        paths,
        "oil_price_usd_bbl",
        OIL_FIGURE_OUT,
        "2026년 분기별 유가 시나리오",
        "2026년 3~5월 실제·지정값 이후 분기별 전제를 월별 경로로 변환",
        "달러/배럴",
    )
    svg_line_plot(
        sim,
        "expected_inflation_level_pct",
        EXP_FIGURE_OUT,
        "유가 시나리오별 기대인플레이션 전망",
        "최종 VAR(3) 모형, 기준 기대인플레이션 2.6%",
        "%",
        threshold=ANCHORING_THRESHOLD_LEVEL,
    )
    write_report(paths, summary, base_oil)

    print(f"Saved: {OIL_PATH_OUT}")
    print(f"Saved: {SIM_OUT}")
    print(f"Saved: {SUMMARY_OUT}")
    print(f"Saved: {OIL_FIGURE_OUT}")
    print(f"Saved: {EXP_FIGURE_OUT}")
    print(f"Saved: {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
