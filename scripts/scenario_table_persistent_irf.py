from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_HIST_PATH = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
IRF_PATH = ROOT / "reports" / "irf_oil_shock_cholesky_var3.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "simulation"

OIL_PATH_OUT = REPORT_DIR / "scenario_table_oil_paths_actual_3m_5m.csv"
SIM_OUT = REPORT_DIR / "scenario_table_expected_inflation_irf_actual_3m_5m.csv"
SUMMARY_OUT = REPORT_DIR / "scenario_table_expected_inflation_summary_actual_3m_5m.csv"
REPORT_OUT = REPORT_DIR / "scenario_table_persistent_irf_actual_3m_5m.md"
FIGURE_OUT = FIGURE_DIR / "scenario_table_expected_inflation_irf_actual_3m_5m.svg"

OIL_SHOCK_SIGMA = 8.045945115677503
HORIZON = 24
EXPECTED_INFLATION_BASELINE = 2.6
ANCHORING_THRESHOLD_LEVEL = 3.0
ANCHORING_THRESHOLD_RESPONSE = ANCHORING_THRESHOLD_LEVEL - EXPECTED_INFLATION_BASELINE
ACTUAL_OIL = {
    pd.Timestamp("2026-03-01"): 126.71,
    pd.Timestamp("2026-04-01"): 105.30,
    pd.Timestamp("2026-05-01"): 105.00,
}

SCENARIOS = [
    {
        "scenario": "시나리오 1: 조기 종전",
        "production_loss": "-4% ~ -2% 잔존",
        "oil_2026q2": 85.0,
        "oil_2027q4": 90.0,
        "war_increase": "+43%",
        "color": "#0072B2",
    },
    {
        "scenario": "시나리오 2: 봉쇄 장기화",
        "production_loss": "-10% 지속",
        "oil_2026q2": 93.0,
        "oil_2027q4": 117.0,
        "war_increase": "+86%",
        "color": "#D55E00",
    },
    {
        "scenario": "시나리오 3: 에너지 시설 타격",
        "production_loss": "-20% 장기화",
        "oil_2026q2": 129.0,
        "oil_2027q4": 174.0,
        "war_increase": "+176%",
        "color": "#CC79A7",
    },
]


def build_oil_paths() -> pd.DataFrame:
    hist = pd.read_csv(RAW_HIST_PATH, parse_dates=["date"])
    base_oil = float(hist.loc[hist["date"] == pd.Timestamp("2026-02-01"), "dubai_oil_usd_bbl"].iloc[0])
    months = pd.date_range("2026-03-01", periods=HORIZON + 1, freq="MS")

    rows = []
    for spec in SCENARIOS:
        # Assumption: 2026-03~2026-05 use actual/user-specified values.
        # From 2026-06 onward, the path moves from the scenario's 2026Q2
        # price to the 2027Q4 price by 2027-12. It is held constant after that.
        end_index = list(months).index(pd.Timestamp("2027-12-01"))
        scenario_start_index = list(months).index(pd.Timestamp("2026-06-01"))
        price_path = []
        for idx, month in enumerate(months):
            if month in ACTUAL_OIL:
                price = ACTUAL_OIL[month]
            elif idx <= end_index:
                price = spec["oil_2026q2"] + (spec["oil_2027q4"] - spec["oil_2026q2"]) * (
                    idx - scenario_start_index
                ) / (end_index - scenario_start_index)
            else:
                price = spec["oil_2027q4"]
            price_path.append(price)

        prev_prices = [base_oil] + price_path[:-1]
        for idx, (month, price, prev) in enumerate(zip(months, price_path, prev_prices, strict=True)):
            dln_oil = 100 * np.log(price / prev)
            rows.append(
                {
                    "scenario": spec["scenario"],
                    "production_loss": spec["production_loss"],
                    "war_increase": spec["war_increase"],
                    "horizon": idx,
                    "date": month,
                    "oil_price_usd_bbl": price,
                    "previous_oil_price_usd_bbl": prev,
                    "dln_oil_pct": dln_oil,
                    "shock_size_sigma": dln_oil / OIL_SHOCK_SIGMA,
                }
            )
    return pd.DataFrame(rows)


def simulate(paths: pd.DataFrame) -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp = irf[irf["response"] == "expected_inflation"].sort_values("horizon").set_index("horizon")
    exp_irf = exp["irf"].reindex(range(HORIZON + 1)).to_numpy()
    exp_lower = exp["lower_95"].reindex(range(HORIZON + 1)).to_numpy()
    exp_upper = exp["upper_95"].reindex(range(HORIZON + 1)).to_numpy()

    rows = []
    for spec in SCENARIOS:
        scenario = spec["scenario"]
        shock = paths[paths["scenario"] == scenario].sort_values("horizon")["shock_size_sigma"].to_numpy()
        for h in range(HORIZON + 1):
            response = 0.0
            lower = 0.0
            upper = 0.0
            for j, multiplier in enumerate(shock):
                if h - j >= 0:
                    response += multiplier * exp_irf[h - j]
                    low_component = multiplier * exp_lower[h - j]
                    high_component = multiplier * exp_upper[h - j]
                    lower += min(low_component, high_component)
                    upper += max(low_component, high_component)
            rows.append(
                {
                    "scenario": scenario,
                    "horizon": h,
                    "date": pd.Timestamp("2026-03-01") + pd.DateOffset(months=h),
                    "expected_inflation_response_pp": response,
                    "expected_inflation_response_lower_95_pp": lower,
                    "expected_inflation_response_upper_95_pp": upper,
                    "expected_inflation_level_pct": EXPECTED_INFLATION_BASELINE + response,
                    "expected_inflation_level_lower_95_pct": EXPECTED_INFLATION_BASELINE + lower,
                    "expected_inflation_level_upper_95_pct": EXPECTED_INFLATION_BASELINE + upper,
                    "above_anchoring_threshold": EXPECTED_INFLATION_BASELINE + response >= ANCHORING_THRESHOLD_LEVEL,
                }
            )
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        sub = sim[sim["scenario"] == spec["scenario"]]
        max_row = sub.loc[sub["expected_inflation_response_pp"].idxmax()]
        threshold_rows = sub[sub["above_anchoring_threshold"]]
        rows.append(
            {
                "scenario": spec["scenario"],
                "max_response_pp": float(max_row["expected_inflation_response_pp"]),
                "max_response_horizon": int(max_row["horizon"]),
                "max_level_pct": EXPECTED_INFLATION_BASELINE + float(max_row["expected_inflation_response_pp"]),
                "max_level_lower_95_pct": float(max_row["expected_inflation_level_lower_95_pct"]),
                "max_level_upper_95_pct": float(max_row["expected_inflation_level_upper_95_pct"]),
                "first_threshold_horizon": int(threshold_rows["horizon"].iloc[0]) if not threshold_rows.empty else pd.NA,
                "response_t": float(sub.loc[sub["horizon"] == 0, "expected_inflation_response_pp"].iloc[0]),
                "response_t6": float(sub.loc[sub["horizon"] == 6, "expected_inflation_response_pp"].iloc[0]),
                "response_t12": float(sub.loc[sub["horizon"] == 12, "expected_inflation_response_pp"].iloc[0]),
                "response_t24": float(sub.loc[sub["horizon"] == 24, "expected_inflation_response_pp"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def svg_plot(sim: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 520
    left = 82
    right = 32
    top = 66
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    ymin = min(float(sim["expected_inflation_level_pct"].min()), ANCHORING_THRESHOLD_LEVEL, EXPECTED_INFLATION_BASELINE)
    ymax = max(float(sim["expected_inflation_level_pct"].max()), ANCHORING_THRESHOLD_LEVEL, EXPECTED_INFLATION_BASELINE)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
    ymin -= pad
    ymax += pad

    def x_pos(h: int) -> float:
        return left + h * plot_width / HORIZON

    def y_pos(value: float) -> float:
        return top + plot_height - (value - ymin) * plot_height / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.zero{stroke:#333;stroke-width:1.2}.threshold{stroke:#111;stroke-width:1.6;stroke-dasharray:7 5}.series{fill:none;stroke-width:2.8}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">Persistent oil-price scenarios: expected inflation level</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#555">Baseline 2.6% plus Cholesky VAR(3) IRF response</text>',
    ]

    for tick in range(6):
        y = top + tick * plot_height / 5
        value = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.2f}</text>')

    baseline_y = y_pos(EXPECTED_INFLATION_BASELINE)
    threshold_y = y_pos(ANCHORING_THRESHOLD_LEVEL)
    lines.append(f'<line class="zero" x1="{left}" y1="{baseline_y:.2f}" x2="{width - right}" y2="{baseline_y:.2f}"/>')
    lines.append(
        f'<text x="{width - right - 4}" y="{baseline_y + 15:.2f}" font-size="12" text-anchor="end">'
        f'baseline: {EXPECTED_INFLATION_BASELINE:.1f}%</text>'
    )
    lines.append(f'<line class="threshold" x1="{left}" y1="{threshold_y:.2f}" x2="{width - right}" y2="{threshold_y:.2f}"/>')
    lines.append(
        f'<text x="{width - right - 4}" y="{threshold_y - 7:.2f}" font-size="12" text-anchor="end">'
        f'anchoring threshold: {ANCHORING_THRESHOLD_LEVEL:.1f}%</text>'
    )

    for h in [0, 3, 6, 9, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 34}" font-size="11" text-anchor="middle">T+{h}</text>')

    legend_y = height - 16
    for idx, spec in enumerate(SCENARIOS):
        sub = sim[sim["scenario"] == spec["scenario"]].sort_values("horizon")
        points = " ".join(
            f"{x_pos(int(row.horizon)):.2f},{y_pos(float(row.expected_inflation_level_pct)):.2f}"
            for row in sub.itertuples()
        )
        lines.append(f'<polyline class="series" stroke="{spec["color"]}" points="{points}"/>')
        lx = left + idx * 260
        lines.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 36}" y2="{legend_y}" stroke="{spec["color"]}" stroke-width="3"/>')
        lines.append(f'<text x="{lx + 44}" y="{legend_y + 4}" font-size="12">{spec["scenario"].replace("시나리오 ", "S")}</text>')

    lines.append("</svg>")
    FIGURE_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_report(paths: pd.DataFrame, sim: pd.DataFrame, summary: pd.DataFrame) -> None:
    first_shocks = paths[paths["horizon"] == 0][["scenario", "oil_price_usd_bbl", "dln_oil_pct", "shock_size_sigma"]]
    lines = [
        "# 시나리오 표 기반 지속적 충격 IRF",
        "",
        "## 가정",
        "",
        "- 기준 유가: 2026년 2월 실제 두바이유 68.51달러.",
        "- 2026년 3월과 4월은 ECOS 실제값, 2026년 5월은 사용자가 지정한 105달러를 사용했습니다.",
        "- 2026년 6월부터 표의 시나리오별 2026년 2/4분기 유가 전망을 적용했습니다.",
        "- 2026년 6월부터 2027년 12월까지 표의 2027년 4/4분기 유가 수준에 도달하도록 월별 선형보간했습니다.",
        "- 2027년 12월 이후 24개월 horizon 안에서는 해당 가격이 유지된다고 가정했습니다.",
        "",
        "## 초기 충격 크기",
        "",
        f"- IRF 1σ 유가 충격: {OIL_SHOCK_SIGMA:.4f}%p",
        f"- 기준 기대인플레이션: {EXPECTED_INFLATION_BASELINE:.1f}%",
        f"- 앵커링 훼손 임계선: 기대인플레이션 {ANCHORING_THRESHOLD_LEVEL:.1f}%",
        f"- 그래프상 임계 반응폭: +{ANCHORING_THRESHOLD_RESPONSE:.1f}%p",
        "",
        "| 시나리오 | T 유가 | T 유가 변동률 | T 충격 크기 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in first_shocks.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['oil_price_usd_bbl']:.2f} | "
            f"{row['dln_oil_pct']:.4f}% | {row['shock_size_sigma']:.4f}σ |"
        )

    lines.extend(
        [
            "",
            "## 기대인플레이션 반응 요약",
            "",
            "| 시나리오 | 최대 반응 | 최대 수준 | 최대 수준 95% CI | 최대 시점 | 임계선 최초 도달 | T | T+6 | T+12 | T+24 |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        threshold_text = "도달 안 함" if pd.isna(row["first_threshold_horizon"]) else f"T+{int(row['first_threshold_horizon'])}"
        lines.append(
            f"| {row['scenario']} | {row['max_response_pp']:.4f}%p | {row['max_level_pct']:.4f}% | "
            f"[{row['max_level_lower_95_pct']:.4f}%, {row['max_level_upper_95_pct']:.4f}%] | "
            f"T+{int(row['max_response_horizon'])} | {threshold_text} | "
            f"{row['response_t']:.4f} | {row['response_t6']:.4f} | {row['response_t12']:.4f} | {row['response_t24']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 파일",
            "",
            f"- 유가 경로: `{OIL_PATH_OUT.relative_to(ROOT).as_posix()}`",
            f"- 기대인플레이션 IRF 시뮬레이션: `{SIM_OUT.relative_to(ROOT).as_posix()}`",
            f"- 요약표: `{SUMMARY_OUT.relative_to(ROOT).as_posix()}`",
            f"- 그래프: `{FIGURE_OUT.relative_to(ROOT).as_posix()}`",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = build_oil_paths()
    sim = simulate(paths)
    summary = summarize(sim)
    paths.to_csv(OIL_PATH_OUT, index=False, encoding="utf-8-sig")
    sim.to_csv(SIM_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    svg_plot(sim)
    write_report(paths, sim, summary)
    print(f"Saved: {OIL_PATH_OUT}")
    print(f"Saved: {SIM_OUT}")
    print(f"Saved: {SUMMARY_OUT}")
    print(f"Saved: {FIGURE_OUT}")
    print(f"Saved: {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
