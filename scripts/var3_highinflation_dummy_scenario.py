from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
DATA_PATH = ROOT / "data" / "processed" / "svar_main_level_stl_cpi_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "simulation"

COEF_OUT = REPORT_DIR / "var3_highinflation_dummy_coefficients.csv"
SCENARIO_OIL_OUT = REPORT_DIR / "scenario_highinflation_dummy_oil_paths.csv"
SCENARIO_SIM_OUT = REPORT_DIR / "scenario_highinflation_dummy_expected_inflation.csv"
SCENARIO_SUMMARY_OUT = REPORT_DIR / "scenario_highinflation_dummy_summary.csv"
REPORT_OUT = REPORT_DIR / "var3_highinflation_dummy_scenario.md"
FIGURE_OUT = FIGURE_DIR / "scenario_highinflation_dummy_expected_inflation.svg"

VAR_LAG = 3
HORIZON = 24
START_DATE = pd.Timestamp("2026-03-01")
BASELINE_EXPECTED_INFLATION = 2.6
ANCHORING_THRESHOLD_LEVEL = 3.0
OIL_SHOCK_SIGMA = 8.045945115677503

VAR_COLUMNS = ["dln_oil_pct", "dln_fx_pct", "dln_cpi_stl_pct", "expected_inflation"]

ACTUAL_OIL = {
    pd.Timestamp("2026-03-01"): 126.71,
    pd.Timestamp("2026-04-01"): 105.30,
    pd.Timestamp("2026-05-01"): 105.00,
}

SCENARIOS = [
    {"scenario": "기준 시나리오", "oil_2026q2": 100.0, "oil_2026q3": 90.0, "oil_2026q4": 87.0, "oil_2027": 87.0, "color": "#0072B2"},
    {"scenario": "고유가 장기화 시나리오", "oil_2026q2": 105.0, "oil_2026q3": 105.0, "oil_2026q4": 105.0, "oil_2027": 105.0, "color": "#D55E00"},
    {"scenario": "유가 안정 시나리오", "oil_2026q2": 95.0, "oil_2026q3": 85.0, "oil_2026q4": 80.0, "oil_2027": 80.0, "color": "#009E73"},
]


def add_high_inflation_dummy(var_data: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"]).sort_values("date")
    raw["cpi_yoy_pct"] = 100 * (np.log(raw["cpi"]) - np.log(raw["cpi"].shift(12)))
    raw["high_inflation_dummy"] = (raw["cpi_yoy_pct"] >= 3.0).astype(int)
    merged = var_data.merge(raw[["date", "cpi_yoy_pct", "high_inflation_dummy"]], on="date", how="left")
    merged["high_inflation_dummy"] = merged["high_inflation_dummy"].fillna(0).astype(int)
    merged["cpi_yoy_pct"] = merged["cpi_yoy_pct"].fillna(0.0)
    return merged


def build_design(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = data.copy().sort_values("date").reset_index(drop=True)
    df["const"] = 1.0
    for lag in range(1, VAR_LAG + 1):
        for col in VAR_COLUMNS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    x_cols = ["const", "high_inflation_dummy"]
    for lag in range(1, VAR_LAG + 1):
        x_cols.extend([f"{col}_lag{lag}" for col in VAR_COLUMNS])
    y = df[VAR_COLUMNS].iloc[VAR_LAG:].copy()
    x = df[x_cols].iloc[VAR_LAG:].copy()
    return y, x


def estimate_varx(data: pd.DataFrame) -> pd.DataFrame:
    y, x = build_design(data)
    x_mat = x.to_numpy(dtype=float)
    coef = {}
    for col in VAR_COLUMNS:
        coef[col] = np.linalg.lstsq(x_mat, y[col].to_numpy(dtype=float), rcond=None)[0]
    return pd.DataFrame(coef, index=x.columns)


def scenario_price(spec: dict[str, float | str], month: pd.Timestamp) -> float:
    if month in ACTUAL_OIL:
        return ACTUAL_OIL[month]
    if month == pd.Timestamp("2026-06-01"):
        return float(spec["oil_2026q2"])
    if pd.Timestamp("2026-07-01") <= month <= pd.Timestamp("2026-09-01"):
        return float(spec["oil_2026q3"])
    if pd.Timestamp("2026-10-01") <= month <= pd.Timestamp("2026-12-01"):
        return float(spec["oil_2026q4"])
    return float(spec["oil_2027"])


def build_oil_paths() -> tuple[pd.DataFrame, float]:
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
    base_oil = float(raw.loc[raw["date"] == pd.Timestamp("2026-02-01"), "dubai_oil_usd_bbl"].iloc[0])
    months = pd.date_range(START_DATE, periods=HORIZON + 1, freq="MS")
    rows = []
    for spec in SCENARIOS:
        prices = [scenario_price(spec, month) for month in months]
        previous = [base_oil] + prices[:-1]
        for h, (month, price, prev) in enumerate(zip(months, prices, previous, strict=True)):
            dln = 100 * np.log(price / prev)
            rows.append({
                "scenario": spec["scenario"],
                "horizon": h,
                "date": month,
                "oil_price_usd_bbl": price,
                "previous_oil_price_usd_bbl": prev,
                "dln_oil_pct": dln,
                "shock_size_sigma": dln / OIL_SHOCK_SIGMA,
            })
    return pd.DataFrame(rows), base_oil


def design_row(history: list[dict[str, float]], high_dummy: int) -> pd.Series:
    row: dict[str, float] = {"const": 1.0, "high_inflation_dummy": float(high_dummy)}
    for lag in range(1, VAR_LAG + 1):
        lag_values = history[-lag]
        for col in VAR_COLUMNS:
            row[f"{col}_lag{lag}"] = float(lag_values[col])
    return pd.Series(row)


def conditional_forecast(coef: pd.DataFrame, initial_history: list[dict[str, float]], oil_dln_path: np.ndarray, high_dummy_path: np.ndarray) -> pd.DataFrame:
    history = [dict(row) for row in initial_history]
    rows = []
    for h in range(HORIZON + 1):
        x = design_row(history, int(high_dummy_path[h]))
        yhat = {}
        for col in VAR_COLUMNS:
            yhat[col] = float(x.reindex(coef.index).to_numpy(dtype=float) @ coef[col].to_numpy(dtype=float))
        yhat["dln_oil_pct"] = float(oil_dln_path[h])
        history.append(yhat)
        rows.append({"horizon": h, "date": START_DATE + pd.DateOffset(months=h), **yhat})
    return pd.DataFrame(rows)


def simulate(coef: pd.DataFrame, data: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    initial = data.sort_values("date").tail(VAR_LAG)[VAR_COLUMNS].to_dict("records")
    high_dummy_path = np.ones(HORIZON + 1, dtype=int)
    zero_oil = np.zeros(HORIZON + 1)
    baseline_forecast = conditional_forecast(coef, initial, zero_oil, high_dummy_path)
    rows = []
    for spec in SCENARIOS:
        scenario = str(spec["scenario"])
        oil_path = paths[paths["scenario"] == scenario].sort_values("horizon")["dln_oil_pct"].to_numpy(dtype=float)
        scenario_forecast = conditional_forecast(coef, initial, oil_path, high_dummy_path)
        response = scenario_forecast["expected_inflation"] - baseline_forecast["expected_inflation"]
        for h in range(HORIZON + 1):
            level = BASELINE_EXPECTED_INFLATION + float(response.iloc[h])
            rows.append({
                "scenario": scenario,
                "horizon": h,
                "date": START_DATE + pd.DateOffset(months=h),
                "expected_inflation_response_pp": float(response.iloc[h]),
                "expected_inflation_level_pct": level,
                "above_anchoring_threshold": level >= ANCHORING_THRESHOLD_LEVEL,
            })
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        scenario = str(spec["scenario"])
        sub = sim[sim["scenario"] == scenario]
        max_row = sub.loc[sub["expected_inflation_response_pp"].idxmax()]
        threshold_rows = sub[sub["above_anchoring_threshold"]]
        rows.append({
            "scenario": scenario,
            "max_response_pp": float(max_row["expected_inflation_response_pp"]),
            "max_response_horizon": int(max_row["horizon"]),
            "max_level_pct": float(max_row["expected_inflation_level_pct"]),
            "first_threshold_horizon": int(threshold_rows["horizon"].iloc[0]) if not threshold_rows.empty else pd.NA,
            "level_t": float(sub.loc[sub["horizon"] == 0, "expected_inflation_level_pct"].iloc[0]),
            "level_t3": float(sub.loc[sub["horizon"] == 3, "expected_inflation_level_pct"].iloc[0]),
            "level_t6": float(sub.loc[sub["horizon"] == 6, "expected_inflation_level_pct"].iloc[0]),
            "level_t12": float(sub.loc[sub["horizon"] == 12, "expected_inflation_level_pct"].iloc[0]),
            "level_t24": float(sub.loc[sub["horizon"] == 24, "expected_inflation_level_pct"].iloc[0]),
        })
    return pd.DataFrame(rows)


def svg_plot(sim: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 980, 520
    left, right, top, bottom = 82, 34, 70, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    ymin = min(float(sim["expected_inflation_level_pct"].min()), ANCHORING_THRESHOLD_LEVEL)
    ymax = max(float(sim["expected_inflation_level_pct"].max()), ANCHORING_THRESHOLD_LEVEL)
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
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.threshold{stroke:#111;stroke-width:1.6;stroke-dasharray:7 5}.series{fill:none;stroke-width:2.8}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">고인플레 더미 모형: 기대인플레이션 전망</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#555">VAR(3)+고인플레 국면 더미, 기준 기대인플레이션 2.6%</text>',
    ]
    for tick in range(6):
        y = top + tick * plot_height / 5
        value = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.2f}</text>')
    y = y_pos(ANCHORING_THRESHOLD_LEVEL)
    lines.append(f'<line class="threshold" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
    lines.append(f'<text x="{width - right - 4}" y="{y - 7:.2f}" font-size="12" text-anchor="end">3.0% 임계선</text>')
    for h in [0, 3, 6, 9, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 36}" font-size="11" text-anchor="middle">T+{h}</text>')
    legend_y = height - 18
    for idx, spec in enumerate(SCENARIOS):
        scenario = str(spec["scenario"])
        sub = sim[sim["scenario"] == scenario].sort_values("horizon")
        points = " ".join(f"{x_pos(int(row.horizon)):.2f},{y_pos(float(row.expected_inflation_level_pct)):.2f}" for row in sub.itertuples())
        color = str(spec["color"])
        lines.append(f'<polyline class="series" stroke="{color}" points="{points}"/>')
        lx = left + idx * 270
        lines.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 36}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx + 44}" y="{legend_y + 4}" font-size="12">{scenario}</text>')
    lines.append("</svg>")
    FIGURE_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_report(data: pd.DataFrame, coef: pd.DataFrame, paths: pd.DataFrame, summary: pd.DataFrame, base_oil: float) -> None:
    high_count = int(data["high_inflation_dummy"].sum())
    total_count = len(data)
    dummy_coef = float(coef.loc["high_inflation_dummy", "expected_inflation"])
    first_shocks = paths[paths["date"] == START_DATE]
    lines = [
        "# 고인플레 국면 더미 VAR(3) 시나리오",
        "",
        "## 모형 설정",
        "",
        "- 기본 VAR(3)은 유지했습니다.",
        "- 추가 외생항은 고인플레 국면 더미 하나만 사용했습니다.",
        "- 유가 변화율 lag × 고인플레 더미 상호작용항은 제외했습니다.",
        "- 고인플레 국면은 전년동월대비 CPI 상승률이 3% 이상인 달로 정의했습니다.",
        f"- 표본 내 고인플레 더미=1 관측치: {high_count}/{total_count}개.",
        f"- 기대인플레이션 방정식의 고인플레 더미 계수: {dummy_coef:.6f}.",
        "- 시나리오 구간은 고인플레 국면이 지속되는 조건으로 계산했습니다.",
        "- 유가 효과만 분리하기 위해 동일한 고인플레 조건의 무충격 경로를 기준선으로 두고 차이를 계산했습니다.",
        "",
        "## 시나리오 유가 경로",
        "",
        f"- 기준 유가: 2026년 2월 두바이유 {base_oil:.2f}달러.",
        "- 2026년 3월 126.71달러, 4월 105.30달러, 5월 105.00달러를 사용했습니다.",
        "- 2026년 6월부터 사용자가 제시한 분기별 시나리오를 적용했습니다.",
        "",
        "## 2026년 3월 최초 충격",
        "",
        "| 시나리오 | 유가 | 전월 대비 로그변화율 | 충격 크기 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in first_shocks.iterrows():
        lines.append(f"| {row['scenario']} | {row['oil_price_usd_bbl']:.2f}달러 | {row['dln_oil_pct']:.4f}% | {row['shock_size_sigma']:.4f}σ |")
    lines.extend([
        "",
        "## 기대인플레이션 결과",
        "",
        "| 시나리오 | 최대 반응 | 최대 수준 | 최대 시점 | 3% 최초 도달 | T | T+3 | T+6 | T+12 | T+24 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in summary.iterrows():
        threshold = "도달 안 함" if pd.isna(row["first_threshold_horizon"]) else f"T+{int(row['first_threshold_horizon'])}"
        lines.append(
            f"| {row['scenario']} | {row['max_response_pp']:.4f}%p | {row['max_level_pct']:.4f}% | "
            f"T+{int(row['max_response_horizon'])} | {threshold} | {row['level_t']:.4f}% | "
            f"{row['level_t3']:.4f}% | {row['level_t6']:.4f}% | {row['level_t12']:.4f}% | {row['level_t24']:.4f}% |"
        )
    lines.extend([
        "",
        "## 해석 메모",
        "",
        "이 모형은 고인플레 국면의 평균적인 수준 이동만 통제하고, 유가 충격의 전달계수 자체는 메인 VAR과 동일한 방식으로 추정합니다. 따라서 상호작용 모형보다 덜 민감한 확장 모형입니다.",
        "",
        "## 파일",
        "",
        f"- 계수표: `{COEF_OUT.relative_to(ROOT).as_posix()}`",
        f"- 유가 경로: `{SCENARIO_OIL_OUT.relative_to(ROOT).as_posix()}`",
        f"- 시뮬레이션: `{SCENARIO_SIM_OUT.relative_to(ROOT).as_posix()}`",
        f"- 요약표: `{SCENARIO_SUMMARY_OUT.relative_to(ROOT).as_posix()}`",
        f"- 그래프: `{FIGURE_OUT.relative_to(ROOT).as_posix()}`",
    ])
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    data = add_high_inflation_dummy(data)
    coef = estimate_varx(data)
    paths, base_oil = build_oil_paths()
    sim = simulate(coef, data, paths)
    summary = summarize(sim)
    coef.to_csv(COEF_OUT, encoding="utf-8-sig")
    paths.to_csv(SCENARIO_OIL_OUT, index=False, encoding="utf-8-sig")
    sim.to_csv(SCENARIO_SIM_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SCENARIO_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    svg_plot(sim)
    write_report(data, coef, paths, summary, base_oil)
    print(f"Saved: {COEF_OUT}")
    print(f"Saved: {SCENARIO_OIL_OUT}")
    print(f"Saved: {SCENARIO_SIM_OUT}")
    print(f"Saved: {SCENARIO_SUMMARY_OUT}")
    print(f"Saved: {FIGURE_OUT}")
    print(f"Saved: {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
