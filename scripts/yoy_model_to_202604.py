from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
RAW_OLD = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
RAW_NEW = ROOT / "data" / "raw" / "ecos_monthly_202603_202604.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "yoy_model_to_202604"

RAW_COMBINED = ROOT / "data" / "raw" / "ecos_monthly_201001_202604.csv"
VAR_DATA = PROCESSED_DIR / "svar_yoy_level_exp_201101_202604.csv"
ADF_OUT = REPORT_DIR / "adf_yoy_model_to_202604.csv"
LAG_OUT = REPORT_DIR / "var_lag_selection_yoy_model_to_202604.csv"
RESID_OUT = REPORT_DIR / "var_residual_diagnostics_yoy_model_to_202604.csv"
LM_OUT = REPORT_DIR / "var_individual_lm_yoy_model_to_202604.csv"
IRF_OUT = REPORT_DIR / "irf_oil_shock_yoy_model_to_202604.csv"
IRF_SUMMARY_OUT = REPORT_DIR / "irf_oil_shock_summary_yoy_model_to_202604.csv"
CHOLESKY_OUT = REPORT_DIR / "cholesky_matrix_yoy_model_to_202604.csv"
SCENARIO_OIL_OUT = REPORT_DIR / "scenario_oil_paths_yoy_model_to_202604.csv"
SCENARIO_SIM_OUT = REPORT_DIR / "scenario_expected_inflation_yoy_model_to_202604.csv"
SCENARIO_SUMMARY_OUT = REPORT_DIR / "scenario_expected_inflation_summary_yoy_model_to_202604.csv"
REPORT_OUT = REPORT_DIR / "yoy_model_to_202604_full_analysis.md"
IRF_FIGURE = FIGURE_DIR / "oil_shock_irf_yoy_model_to_202604.svg"
SCENARIO_FIGURE = FIGURE_DIR / "scenario_expected_inflation_yoy_model_to_202604.svg"

VAR_COLUMNS = ["yoy_oil_pct", "yoy_fx_pct", "yoy_cpi_pct", "expected_inflation"]
LABELS = {
    "yoy_oil_pct": "국제유가 전년동월대비 로그변화율",
    "yoy_fx_pct": "환율 전년동월대비 로그변화율",
    "yoy_cpi_pct": "CPI 전년동월대비 로그변화율",
    "expected_inflation": "기대인플레이션 수준",
}
HORIZON = 24
EXPECTED_INFLATION_BASELINE = 2.6
ANCHORING_THRESHOLD = 3.0
SCENARIOS = [
    {"scenario": "시나리오 1: 조기 종전", "oil_2026q2": 85.0, "oil_2027q4": 90.0, "color": "#0072B2"},
    {"scenario": "시나리오 2: 봉쇄 장기화", "oil_2026q2": 93.0, "oil_2027q4": 117.0, "color": "#D55E00"},
    {"scenario": "시나리오 3: 에너지 시설 타격", "oil_2026q2": 129.0, "oil_2027q4": 174.0, "color": "#CC79A7"},
]
ACTUAL_FORECAST_OIL = {
    pd.Timestamp("2026-03-01"): 126.71,
    pd.Timestamp("2026-04-01"): 105.30,
    pd.Timestamp("2026-05-01"): 105.00,
}


def combine_raw() -> pd.DataFrame:
    old = pd.read_csv(RAW_OLD, parse_dates=["date"])
    new = pd.read_csv(RAW_NEW, parse_dates=["date"])
    raw = pd.concat([old, new], ignore_index=True)
    raw = raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    raw.to_csv(RAW_COMBINED, index=False, encoding="utf-8-sig")
    return raw


def build_var_data(raw: pd.DataFrame) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = raw.copy()
    data["yoy_oil_pct"] = 100 * (np.log(data["dubai_oil_usd_bbl"]) - np.log(data["dubai_oil_usd_bbl"].shift(12)))
    data["yoy_fx_pct"] = 100 * (
        np.log(data["exchange_rate_krw_usd"]) - np.log(data["exchange_rate_krw_usd"].shift(12))
    )
    data["yoy_cpi_pct"] = 100 * (np.log(data["cpi"]) - np.log(data["cpi"].shift(12)))
    data["expected_inflation"] = data["expected_inflation_1y"]
    var_data = data[["date", *VAR_COLUMNS]].dropna().copy()
    var_data.to_csv(VAR_DATA, index=False, encoding="utf-8-sig")
    return var_data


def run_adf(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in VAR_COLUMNS:
        stat, pvalue, usedlag, nobs, crit, icbest = adfuller(data[col].dropna(), regression="c", autolag="AIC")
        rows.append(
            {
                "variable": col,
                "label": LABELS[col],
                "adf_stat": stat,
                "p_value": pvalue,
                "used_lag": usedlag,
                "nobs": nobs,
                "decision_5pct": "정상성 있음" if pvalue < 0.05 else "단위근 가능",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ADF_OUT, index=False, encoding="utf-8-sig")
    return out


def lag_selection(data: pd.DataFrame) -> pd.DataFrame:
    y = data.set_index("date")[VAR_COLUMNS]
    model = VAR(y)
    rows = []
    for lag in range(1, 7):
        res = model.fit(lag)
        rows.append(
            {
                "lag": lag,
                "nobs": int(res.nobs),
                "aic": float(res.aic),
                "bic": float(res.bic),
                "hqic": float(res.hqic),
                "fpe": float(res.fpe),
            }
        )
    out = pd.DataFrame(rows)
    out["aic_best"] = out["lag"].eq(int(out.loc[out["aic"].idxmin(), "lag"]))
    out["bic_best"] = out["lag"].eq(int(out.loc[out["bic"].idxmin(), "lag"]))
    out["hqic_best"] = out["lag"].eq(int(out.loc[out["hqic"].idxmin(), "lag"]))
    out.to_csv(LAG_OUT, index=False, encoding="utf-8-sig")
    return out


def system_lm_for_lag(endog: pd.DataFrame, resid: pd.DataFrame, var_lag: int, test_lag: int) -> dict[str, float | int]:
    rows_y, rows_x_base, rows_x_test = [], [], []
    for t in range(var_lag + test_lag, len(endog)):
        date_t = endog.index[t]
        date_test = endog.index[t - test_lag]
        if date_t not in resid.index or date_test not in resid.index:
            continue
        rows_y.append(resid.loc[date_t, VAR_COLUMNS].to_numpy(dtype=float))
        base = [1.0]
        for p in range(1, var_lag + 1):
            base.extend(endog.iloc[t - p].to_numpy(dtype=float))
        rows_x_base.append(base)
        rows_x_test.append(resid.loc[date_test, VAR_COLUMNS].to_numpy(dtype=float))

    y = np.asarray(rows_y, dtype=float)
    x_r = np.asarray(rows_x_base, dtype=float)
    x_u = np.column_stack([x_r, np.asarray(rows_x_test, dtype=float)])
    er, eu = [], []
    for eq in range(y.shape[1]):
        beta_r = np.linalg.lstsq(x_r, y[:, eq], rcond=None)[0]
        beta_u = np.linalg.lstsq(x_u, y[:, eq], rcond=None)[0]
        er.append(y[:, eq] - x_r @ beta_r)
        eu.append(y[:, eq] - x_u @ beta_u)
    er = np.column_stack(er)
    eu = np.column_stack(eu)
    nobs = len(y)
    df = len(VAR_COLUMNS) ** 2
    det_r = max(float(np.linalg.det((er.T @ er) / nobs)), 1e-300)
    det_u = max(float(np.linalg.det((eu.T @ eu) / nobs)), 1e-300)
    lm_stat = max(0.0, nobs * (math.log(det_r) - math.log(det_u)))
    p_value = float(chi2.sf(lm_stat, df))
    return {"lag": test_lag, "nobs": nobs, "lm_stat": lm_stat, "df": df, "p_value": p_value}


def diagnostics(data: pd.DataFrame, var_lag: int) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    y = data.set_index("date")[VAR_COLUMNS]
    res = VAR(y).fit(var_lag)
    port_rows = []
    for test_lag in [6, 12, 18, 24]:
        if test_lag <= var_lag:
            continue
        test = res.test_whiteness(nlags=test_lag, adjusted=True)
        port_rows.append(
            {
                "var_lag": var_lag,
                "test_lags": test_lag,
                "test_statistic": float(test.test_statistic),
                "critical_value_5pct": float(test.crit_value),
                "p_value": float(test.pvalue),
                "df": int(test.df),
            }
        )
    port = pd.DataFrame(port_rows)
    port.to_csv(RESID_OUT, index=False, encoding="utf-8-sig")
    lm = pd.DataFrame([system_lm_for_lag(y, res.resid, var_lag, lag) for lag in range(1, 7)])
    lm.to_csv(LM_OUT, index=False, encoding="utf-8-sig")
    return port, lm, res


def irf_outputs(res) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    irf = res.irf(HORIZON)
    stderr = irf.stderr(orth=True)
    lower = irf.orth_irfs - 1.96 * stderr
    upper = irf.orth_irfs + 1.96 * stderr
    shock_idx = VAR_COLUMNS.index("yoy_oil_pct")
    rows = []
    for h in range(HORIZON + 1):
        for ridx, response in enumerate(VAR_COLUMNS):
            rows.append(
                {
                    "horizon": h,
                    "impulse": "yoy_oil_pct",
                    "response": response,
                    "response_label": LABELS[response],
                    "irf": float(irf.orth_irfs[h, ridx, shock_idx]),
                    "lower_95": float(lower[h, ridx, shock_idx]),
                    "upper_95": float(upper[h, ridx, shock_idx]),
                }
            )
    irf_df = pd.DataFrame(rows)
    irf_df.to_csv(IRF_OUT, index=False, encoding="utf-8-sig")

    summary_rows = []
    for response in ["expected_inflation", "yoy_cpi_pct", "yoy_fx_pct"]:
        sub = irf_df[irf_df["response"] == response]
        max_row = sub.loc[sub["irf"].idxmax()]
        sig = sub[(sub["lower_95"] > 0) | (sub["upper_95"] < 0)]["horizon"].tolist()
        summary_rows.append(
            {
                "shock_to_response": f"유가 → {LABELS[response]}",
                "max_response": float(max_row["irf"]),
                "max_response_horizon": int(max_row["horizon"]),
                "significant_horizons_95": ", ".join(f"T+{int(x)}" for x in sig) if sig else "없음",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(IRF_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    chol = pd.DataFrame(np.linalg.cholesky(res.sigma_u), index=VAR_COLUMNS, columns=[f"shock_{c}" for c in VAR_COLUMNS])
    chol.to_csv(CHOLESKY_OUT, encoding="utf-8-sig")
    oil_sigma = float(chol.loc["yoy_oil_pct", "shock_yoy_oil_pct"])
    return irf_df, summary, oil_sigma


def svg_irf(irf_df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    responses = ["yoy_fx_pct", "yoy_cpi_pct", "expected_inflation"]
    colors = {"yoy_fx_pct": "#009E73", "yoy_cpi_pct": "#D55E00", "expected_inflation": "#CC79A7"}
    width, panel_h, left, right, top, bottom, gap = 980, 220, 82, 32, 78, 44, 42
    height = top + bottom + len(responses) * panel_h + (len(responses) - 1) * gap
    plot_w = width - left - right
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd}.zero{stroke:#333}.band{fill-opacity:.16}.line{fill:none;stroke-width:2.6}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">YoY model oil shock IRF</text>',
        f'<text x="{left}" y="50" font-size="12">Impulse: 1 s.d. oil YoY log-change shock, Cholesky VAR</text>',
    ]

    def x_pos(h: int) -> float:
        return left + h * plot_w / HORIZON

    for idx, resp in enumerate(responses):
        sub = irf_df[irf_df["response"] == resp].sort_values("horizon")
        y_min = min(0.0, float(sub[["irf", "lower_95", "upper_95"]].min().min()))
        y_max = max(0.0, float(sub[["irf", "lower_95", "upper_95"]].max().max()))
        pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.1
        y_min -= pad
        y_max += pad
        ptop = top + idx * (panel_h + gap)

        def y_pos(v: float) -> float:
            return ptop + panel_h - (v - y_min) * panel_h / (y_max - y_min)

        for tick in range(5):
            y = ptop + tick * panel_h / 4
            val = y_max - tick * (y_max - y_min) / 4
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left-10}" y="{y+4:.2f}" font-size="11" text-anchor="end">{val:.3f}</text>')
        zero_y = y_pos(0.0)
        lines.append(f'<line class="zero" x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}"/>')
        lines.append(f'<text x="{left}" y="{ptop-10}" font-size="14" font-weight="700">{LABELS[resp]}</text>')
        upper_points = [(x_pos(int(r.horizon)), y_pos(float(r.upper_95))) for r in sub.itertuples()]
        lower_points = [(x_pos(int(r.horizon)), y_pos(float(r.lower_95))) for r in sub.iloc[::-1].itertuples()]
        polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper_points + lower_points)
        points = " ".join(f"{x_pos(int(r.horizon)):.2f},{y_pos(float(r.irf)):.2f}" for r in sub.itertuples())
        lines.append(f'<polygon class="band" fill="{colors[resp]}" points="{polygon}"/>')
        lines.append(f'<polyline class="line" stroke="{colors[resp]}" points="{points}"/>')
        for h in [0, 6, 12, 18, 24]:
            x = x_pos(h)
            lines.append(f'<text x="{x:.2f}" y="{ptop+panel_h+18}" font-size="11" text-anchor="middle">{h}</text>')
    lines.append("</svg>")
    IRF_FIGURE.write_text("\n".join(lines), encoding="utf-8")


def scenario_simulation(raw: pd.DataFrame, irf_df: pd.DataFrame, oil_sigma: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.date_range("2026-03-01", periods=HORIZON + 1, freq="MS")
    hist_oil = raw.set_index("date")["dubai_oil_usd_bbl"].to_dict()
    exp = irf_df[irf_df["response"] == "expected_inflation"].sort_values("horizon").set_index("horizon")
    response_irf = exp["irf"].reindex(range(HORIZON + 1)).to_numpy()
    lower_irf = exp["lower_95"].reindex(range(HORIZON + 1)).to_numpy()
    upper_irf = exp["upper_95"].reindex(range(HORIZON + 1)).to_numpy()

    oil_rows = []
    sim_rows = []
    for spec in SCENARIOS:
        end_idx = list(months).index(pd.Timestamp("2027-12-01"))
        start_idx = list(months).index(pd.Timestamp("2026-06-01"))
        prices = []
        for idx, month in enumerate(months):
            if month in ACTUAL_FORECAST_OIL:
                price = ACTUAL_FORECAST_OIL[month]
            elif idx <= end_idx:
                price = spec["oil_2026q2"] + (spec["oil_2027q4"] - spec["oil_2026q2"]) * (
                    idx - start_idx
                ) / (end_idx - start_idx)
            else:
                price = spec["oil_2027q4"]
            prices.append(price)

        scenario_oil = {month: price for month, price in zip(months, prices, strict=True)}
        shocks = []
        for h, month in enumerate(months):
            last_year_month = month - pd.DateOffset(months=12)
            last_year_month = pd.Timestamp(last_year_month)
            if last_year_month in hist_oil:
                previous_year_oil = float(hist_oil[last_year_month])
            else:
                previous_year_oil = float(scenario_oil[last_year_month])
            yoy_oil = 100 * np.log(scenario_oil[month] / previous_year_oil)
            shock = yoy_oil / oil_sigma
            shocks.append(shock)
            oil_rows.append(
                {
                    "scenario": spec["scenario"],
                    "horizon": h,
                    "date": month,
                    "oil_price_usd_bbl": scenario_oil[month],
                    "previous_year_oil_price_usd_bbl": previous_year_oil,
                    "yoy_oil_pct": yoy_oil,
                    "shock_size_sigma": shock,
                }
            )

        for h, month in enumerate(months):
            response = lower = upper = 0.0
            for j, shock in enumerate(shocks):
                if h - j >= 0:
                    response += shock * response_irf[h - j]
                    lo = shock * lower_irf[h - j]
                    hi = shock * upper_irf[h - j]
                    lower += min(lo, hi)
                    upper += max(lo, hi)
            sim_rows.append(
                {
                    "scenario": spec["scenario"],
                    "horizon": h,
                    "date": month,
                    "response_pp": response,
                    "lower_95_pp": lower,
                    "upper_95_pp": upper,
                    "level_pct": EXPECTED_INFLATION_BASELINE + response,
                    "lower_95_level_pct": EXPECTED_INFLATION_BASELINE + lower,
                    "upper_95_level_pct": EXPECTED_INFLATION_BASELINE + upper,
                    "above_threshold": EXPECTED_INFLATION_BASELINE + response >= ANCHORING_THRESHOLD,
                }
            )

    oil = pd.DataFrame(oil_rows)
    sim = pd.DataFrame(sim_rows)
    summary_rows = []
    for spec in SCENARIOS:
        sub = sim[sim["scenario"] == spec["scenario"]]
        max_row = sub.loc[sub["level_pct"].idxmax()]
        threshold_rows = sub[sub["above_threshold"]]
        summary_rows.append(
            {
                "scenario": spec["scenario"],
                "max_response_pp": float(max_row["response_pp"]),
                "max_level_pct": float(max_row["level_pct"]),
                "max_level_lower_95_pct": float(max_row["lower_95_level_pct"]),
                "max_level_upper_95_pct": float(max_row["upper_95_level_pct"]),
                "max_response_horizon": int(max_row["horizon"]),
                "first_threshold_horizon": int(threshold_rows["horizon"].iloc[0]) if not threshold_rows.empty else pd.NA,
            }
        )
    oil.to_csv(SCENARIO_OIL_OUT, index=False, encoding="utf-8-sig")
    sim.to_csv(SCENARIO_SIM_OUT, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SCENARIO_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    return oil, sim, summary


def svg_scenario(sim: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 980, 520
    left, right, top, bottom = 82, 32, 66, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin = min(float(sim["level_pct"].min()), EXPECTED_INFLATION_BASELINE, ANCHORING_THRESHOLD)
    ymax = max(float(sim["level_pct"].max()), EXPECTED_INFLATION_BASELINE, ANCHORING_THRESHOLD)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
    ymin -= pad
    ymax += pad

    def x_pos(h: int) -> float:
        return left + h * plot_w / HORIZON

    def y_pos(v: float) -> float:
        return top + plot_h - (v - ymin) * plot_h / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd}.base{stroke:#333;stroke-width:1.2}.thr{stroke:#111;stroke-width:1.6;stroke-dasharray:7 5}.series{fill:none;stroke-width:2.8}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">YoY model scenario: expected inflation level</text>',
        f'<text x="{left}" y="50" font-size="12">Baseline 2.6%, threshold 3.0%, oil scenarios from table</text>',
    ]
    for tick in range(6):
        y = top + tick * plot_h / 5
        val = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" font-size="11" text-anchor="end">{val:.2f}</text>')
    base_y, thr_y = y_pos(EXPECTED_INFLATION_BASELINE), y_pos(ANCHORING_THRESHOLD)
    lines.append(f'<line class="base" x1="{left}" y1="{base_y:.2f}" x2="{width-right}" y2="{base_y:.2f}"/>')
    lines.append(f'<line class="thr" x1="{left}" y1="{thr_y:.2f}" x2="{width-right}" y2="{thr_y:.2f}"/>')
    lines.append(f'<text x="{width-right-4}" y="{thr_y-7:.2f}" font-size="12" text-anchor="end">anchoring threshold: 3.0%</text>')
    lines.append(f'<text x="{width-right-4}" y="{base_y+15:.2f}" font-size="12" text-anchor="end">baseline: 2.6%</text>')
    for h in [0, 3, 6, 9, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height-34}" font-size="11" text-anchor="middle">T+{h}</text>')
    for idx, spec in enumerate(SCENARIOS):
        sub = sim[sim["scenario"] == spec["scenario"]].sort_values("horizon")
        points = " ".join(f"{x_pos(int(r.horizon)):.2f},{y_pos(float(r.level_pct)):.2f}" for r in sub.itertuples())
        lines.append(f'<polyline class="series" stroke="{spec["color"]}" points="{points}"/>')
        lx = left + idx * 260
        lines.append(f'<line x1="{lx}" y1="{height-16}" x2="{lx+36}" y2="{height-16}" stroke="{spec["color"]}" stroke-width="3"/>')
        lines.append(f'<text x="{lx+44}" y="{height-12}" font-size="12">{spec["scenario"].replace("시나리오 ", "S")}</text>')
    lines.append("</svg>")
    SCENARIO_FIGURE.write_text("\n".join(lines), encoding="utf-8")


def write_report(raw, data, adf, lag_table, port, lm, irf_summary, oil_sigma, selected_lag, scenario_summary):
    lines = [
        "# 전년동월대비 로그차분 기반 업데이트 모형",
        "",
        "## 데이터와 변수 정의",
        "",
        f"- 원자료 기간: {raw['date'].min().date()} ~ {raw['date'].max().date()}",
        f"- VAR 표본: {data['date'].min().date()} ~ {data['date'].max().date()}, {len(data)}개",
        "- 유가, 환율, CPI는 `100 * (ln(x_t) - ln(x_{t-12}))`로 정의했습니다.",
        "- 기대인플레이션은 전년동월대비 변환하지 않고 수준값을 사용했습니다.",
        "",
        "## ADF 검정",
        "",
        "| 변수 | ADF | p-value | 판정 |",
        "| --- | ---: | ---: | --- |",
    ]
    for _, r in adf.iterrows():
        lines.append(f"| {r['label']} | {r['adf_stat']:.4f} | {r['p_value']:.4f} | {r['decision_5pct']} |")

    lines.extend(["", "## 시차 선택", "", "| lag | AIC | BIC | HQIC | 선택 |", "| ---: | ---: | ---: | ---: | --- |"])
    for _, r in lag_table.iterrows():
        marks = []
        if r["aic_best"]:
            marks.append("AIC")
        if r["bic_best"]:
            marks.append("BIC")
        if r["hqic_best"]:
            marks.append("HQIC")
        lines.append(f"| {int(r['lag'])} | {r['aic']:.4f} | {r['bic']:.4f} | {r['hqic']:.4f} | {', '.join(marks)} |")
    lines.append("")
    lines.append(f"- 진단용 추정 시차: VAR({selected_lag})")

    lines.extend(["", "## 잔차 진단", "", "| 검정 시차 | p-value | 판정 |", "| ---: | ---: | --- |"])
    for _, r in port.iterrows():
        verdict = "자기상관 있음" if r["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(f"| {int(r['test_lags'])} | {r['p_value']:.4f} | {verdict} |")
    lines.extend(["", "개별 LM lag 1~6:", "", "| residual lag | p-value | 판정 |", "| ---: | ---: | --- |"])
    for _, r in lm.iterrows():
        verdict = "자기상관 있음" if r["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(f"| {int(r['lag'])} | {r['p_value']:.4f} | {verdict} |")

    lines.extend(
        [
            "",
            "## 촐레스키 IRF",
            "",
            "- 변수 순서: 국제유가 → 환율 → CPI → 기대인플레이션",
            f"- 유가 1표준편차 충격: {oil_sigma:.4f}%p, 전년동월대비 유가 로그변화율 단위",
            "",
            "| 충격 → 반응 | 최대 반응 | 최대 시점 | 95% 유의 horizon |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for _, r in irf_summary.iterrows():
        lines.append(
            f"| {r['shock_to_response']} | {r['max_response']:.4f} | T+{int(r['max_response_horizon'])} | {r['significant_horizons_95']} |"
        )

    lines.extend(
        [
            "",
            "## 시나리오 분석",
            "",
            "- 사진 표의 3개 유가 전망을 사용했습니다.",
            "- 2026년 3월과 4월은 실제값, 2026년 5월은 105달러, 2026년 6월부터 시나리오별 2026년 2/4분기 유가를 적용했습니다.",
            "- 전년동월대비 모형이므로 시나리오 유가도 매월 전년동월대비 로그변화율로 환산해 IRF에 투입했습니다.",
            "- 기준 기대인플레이션은 2.6%, 앵커링 임계선은 3.0%입니다.",
            "",
            "| 시나리오 | 최대 수준 | 95% CI | 최대 시점 | 임계선 최초 도달 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, r in scenario_summary.iterrows():
        threshold = "도달 안 함" if pd.isna(r["first_threshold_horizon"]) else f"T+{int(r['first_threshold_horizon'])}"
        lines.append(
            f"| {r['scenario']} | {r['max_level_pct']:.4f}% | [{r['max_level_lower_95_pct']:.4f}%, {r['max_level_upper_95_pct']:.4f}%] | "
            f"T+{int(r['max_response_horizon'])} | {threshold} |"
        )

    lines.extend(
        [
            "",
            "## 파일",
            "",
            f"- VAR 데이터: `{VAR_DATA.relative_to(ROOT).as_posix()}`",
            f"- ADF: `{ADF_OUT.relative_to(ROOT).as_posix()}`",
            f"- 시차 선택: `{LAG_OUT.relative_to(ROOT).as_posix()}`",
            f"- 잔차 진단: `{RESID_OUT.relative_to(ROOT).as_posix()}`",
            f"- IRF: `{IRF_OUT.relative_to(ROOT).as_posix()}`",
            f"- IRF 요약: `{IRF_SUMMARY_OUT.relative_to(ROOT).as_posix()}`",
            f"- IRF 그래프: `{IRF_FIGURE.relative_to(ROOT).as_posix()}`",
            f"- 시나리오 유가 경로: `{SCENARIO_OIL_OUT.relative_to(ROOT).as_posix()}`",
            f"- 시나리오 결과: `{SCENARIO_SUMMARY_OUT.relative_to(ROOT).as_posix()}`",
            f"- 시나리오 그래프: `{SCENARIO_FIGURE.relative_to(ROOT).as_posix()}`",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw = combine_raw()
    data = build_var_data(raw)
    adf = run_adf(data)
    lag_table = lag_selection(data)
    selected_lag = int(lag_table.loc[lag_table["aic"].idxmin(), "lag"])
    port, lm, res = diagnostics(data, selected_lag)
    irf_df, irf_summary, oil_sigma = irf_outputs(res)
    svg_irf(irf_df)
    _, scenario_sim_df, scenario_summary = scenario_simulation(raw, irf_df, oil_sigma)
    svg_scenario(scenario_sim_df)
    write_report(raw, data, adf, lag_table, port, lm, irf_summary, oil_sigma, selected_lag, scenario_summary)
    print(f"Saved: {REPORT_OUT}")
    print(f"Saved: {VAR_DATA}")
    print(f"Saved: {IRF_SUMMARY_OUT}")
    print(f"Saved: {IRF_FIGURE}")
    print(f"Saved: {SCENARIO_SUMMARY_OUT}")
    print(f"Saved: {SCENARIO_FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
