from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
RAW_OLD = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
RAW_NEW = ROOT / "data" / "raw" / "ecos_monthly_202603_202604.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "updated_to_202604"

RAW_COMBINED = ROOT / "data" / "raw" / "ecos_monthly_201001_202604.csv"
STL_COMPONENTS = PROCESSED_DIR / "cpi_stl_components_201001_202604.csv"
VAR_DATA = PROCESSED_DIR / "svar_main_level_stl_cpi_201002_202604.csv"
ADF_OUT = REPORT_DIR / "adf_updated_to_202604.csv"
LAG_OUT = REPORT_DIR / "var_lag_selection_updated_to_202604.csv"
RESID_OUT = REPORT_DIR / "var_residual_diagnostics_updated_to_202604.csv"
LM_OUT = REPORT_DIR / "var_individual_lm_updated_to_202604.csv"
IRF_OUT = REPORT_DIR / "irf_oil_shock_cholesky_updated_to_202604.csv"
IRF_SUMMARY_OUT = REPORT_DIR / "irf_oil_shock_summary_updated_to_202604.csv"
CHOLESKY_OUT = REPORT_DIR / "cholesky_matrix_updated_to_202604.csv"
SCENARIO_OIL_OUT = REPORT_DIR / "scenario_oil_paths_updated_to_202604.csv"
SCENARIO_SIM_OUT = REPORT_DIR / "scenario_expected_inflation_updated_to_202604.csv"
SCENARIO_SUMMARY_OUT = REPORT_DIR / "scenario_expected_inflation_summary_updated_to_202604.csv"
REPORT_OUT = REPORT_DIR / "updated_model_to_202604_full_analysis.md"
IRF_FIGURE = FIGURE_DIR / "oil_shock_irf_updated_to_202604.svg"
SCENARIO_FIGURE = FIGURE_DIR / "scenario_expected_inflation_updated_to_202604.svg"

VAR_COLUMNS = ["dln_oil_pct", "dln_fx_pct", "dln_cpi_stl_pct", "expected_inflation"]
LABELS = {
    "dln_oil_pct": "국제유가",
    "dln_fx_pct": "환율",
    "dln_cpi_stl_pct": "CPI",
    "expected_inflation": "기대인플레이션",
}
HORIZON = 24
EXPECTED_INFLATION_BASELINE = 2.6
ANCHORING_THRESHOLD_LEVEL = 3.0

SCENARIOS = [
    {"scenario": "시나리오 1: 조기 종전", "oil_2026q2": 85.0, "oil_2027q4": 90.0, "color": "#0072B2"},
    {"scenario": "시나리오 2: 봉쇄 장기화", "oil_2026q2": 93.0, "oil_2027q4": 117.0, "color": "#D55E00"},
    {"scenario": "시나리오 3: 에너지 시설 타격", "oil_2026q2": 129.0, "oil_2027q4": 174.0, "color": "#CC79A7"},
]


def combine_raw() -> pd.DataFrame:
    old = pd.read_csv(RAW_OLD, parse_dates=["date"])
    new = pd.read_csv(RAW_NEW, parse_dates=["date"])
    raw = pd.concat([old, new], ignore_index=True)
    raw = raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    raw.to_csv(RAW_COMBINED, index=False, encoding="utf-8-sig")
    return raw


def build_var_data(raw: pd.DataFrame) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log_cpi = np.log(raw["cpi"])
    stl = STL(log_cpi, period=12, robust=True).fit()
    raw = raw.copy()
    raw["log_cpi"] = log_cpi
    raw["log_cpi_stl_seasonal"] = stl.seasonal
    raw["log_cpi_stl_sa"] = raw["log_cpi"] - raw["log_cpi_stl_seasonal"]
    raw["cpi_stl_sa"] = np.exp(raw["log_cpi_stl_sa"])
    raw[["date", "cpi", "cpi_stl_sa", "log_cpi", "log_cpi_stl_sa", "log_cpi_stl_seasonal"]].to_csv(
        STL_COMPONENTS, index=False, encoding="utf-8-sig"
    )

    raw["dln_oil_pct"] = 100 * np.log(raw["dubai_oil_usd_bbl"]).diff()
    raw["dln_fx_pct"] = 100 * np.log(raw["exchange_rate_krw_usd"]).diff()
    raw["dln_cpi_stl_pct"] = 100 * raw["log_cpi_stl_sa"].diff()
    raw["expected_inflation"] = raw["expected_inflation_1y"]
    data = raw[["date", *VAR_COLUMNS]].dropna().copy()
    data.to_csv(VAR_DATA, index=False, encoding="utf-8-sig")
    return data


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


def system_lm_for_lag(endog: pd.DataFrame, resid: pd.DataFrame, var_lag: int, test_lag: int) -> dict[str, float | int | str]:
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


def irf_outputs(res, oil_sigma: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    irf = res.irf(HORIZON)
    stderr = irf.stderr(orth=True)
    lower = irf.orth_irfs - 1.96 * stderr
    upper = irf.orth_irfs + 1.96 * stderr
    shock_idx = VAR_COLUMNS.index("dln_oil_pct")
    rows = []
    for h in range(HORIZON + 1):
        for ridx, response in enumerate(VAR_COLUMNS):
            rows.append(
                {
                    "horizon": h,
                    "impulse": "dln_oil_pct",
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
    for response in ["expected_inflation", "dln_cpi_stl_pct", "dln_fx_pct"]:
        sub = irf_df[irf_df["response"] == response]
        max_row = sub.loc[sub["irf"].idxmax()]
        sig = sub[(sub["lower_95"] > 0) | (sub["upper_95"] < 0)]["horizon"].tolist()
        summary_rows.append(
            {
                "shock_to_response": f"유가 → {LABELS[response]}",
                "max_response": float(max_row["irf"]),
                "max_response_horizon": int(max_row["horizon"]),
                "significant_horizons_95": ", ".join(f"T+{int(x)}" for x in sig) if sig else "없음",
                "oil_shock_sigma": oil_sigma,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(IRF_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    chol = pd.DataFrame(np.linalg.cholesky(res.sigma_u), index=VAR_COLUMNS, columns=[f"shock_{c}" for c in VAR_COLUMNS])
    chol.to_csv(CHOLESKY_OUT, encoding="utf-8-sig")
    return irf_df, summary


def scenario_sim(irf_df: pd.DataFrame, oil_sigma: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.date_range("2026-05-01", periods=HORIZON + 1, freq="MS")
    base_oil = 105.30
    actual_oil = {pd.Timestamp("2026-05-01"): 105.0}
    exp_irf = irf_df[irf_df["response"] == "expected_inflation"].sort_values("horizon").set_index("horizon")
    shock_response = exp_irf["irf"].reindex(range(HORIZON + 1)).to_numpy()
    shock_lower = exp_irf["lower_95"].reindex(range(HORIZON + 1)).to_numpy()
    shock_upper = exp_irf["upper_95"].reindex(range(HORIZON + 1)).to_numpy()

    oil_rows, sim_rows = [], []
    for spec in SCENARIOS:
        end_idx = list(months).index(pd.Timestamp("2027-12-01"))
        start_idx = list(months).index(pd.Timestamp("2026-06-01"))
        prices = []
        for idx, month in enumerate(months):
            if month in actual_oil:
                price = actual_oil[month]
            elif idx <= end_idx:
                price = spec["oil_2026q2"] + (spec["oil_2027q4"] - spec["oil_2026q2"]) * (idx - start_idx) / (
                    end_idx - start_idx
                )
            else:
                price = spec["oil_2027q4"]
            prices.append(price)
        prevs = [base_oil] + prices[:-1]
        shocks = []
        for h, (month, price, prev) in enumerate(zip(months, prices, prevs, strict=True)):
            dln = 100 * np.log(price / prev)
            shock = dln / oil_sigma
            shocks.append(shock)
            oil_rows.append(
                {
                    "scenario": spec["scenario"],
                    "horizon": h,
                    "date": month,
                    "oil_price_usd_bbl": price,
                    "dln_oil_pct": dln,
                    "shock_size_sigma": shock,
                }
            )
        for h, month in enumerate(months):
            response = lower = upper = 0.0
            for j, shock in enumerate(shocks):
                if h - j >= 0:
                    response += shock * shock_response[h - j]
                    lo = shock * shock_lower[h - j]
                    hi = shock * shock_upper[h - j]
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
                    "above_threshold": EXPECTED_INFLATION_BASELINE + response >= ANCHORING_THRESHOLD_LEVEL,
                }
            )
    oil = pd.DataFrame(oil_rows)
    sim = pd.DataFrame(sim_rows)
    summary_rows = []
    for spec in SCENARIOS:
        sub = sim[sim["scenario"] == spec["scenario"]]
        max_row = sub.loc[sub["response_pp"].idxmax()]
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
    scen_summary = pd.DataFrame(summary_rows)
    scen_summary.to_csv(SCENARIO_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    return oil, sim, scen_summary


def svg_line_plot(sim: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width, height, left, right, top, bottom = 980, 520, 82, 32, 66, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    ymin = min(float(sim["level_pct"].min()), EXPECTED_INFLATION_BASELINE, ANCHORING_THRESHOLD_LEVEL)
    ymax = max(float(sim["level_pct"].max()), EXPECTED_INFLATION_BASELINE, ANCHORING_THRESHOLD_LEVEL)
    pad = (ymax - ymin) * 0.12
    ymin -= pad
    ymax += pad

    def x_pos(h: int) -> float:
        return left + h * plot_width / HORIZON

    def y_pos(v: float) -> float:
        return top + plot_height - (v - ymin) * plot_height / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd}.base{stroke:#333;stroke-width:1.2}.thr{stroke:#111;stroke-width:1.6;stroke-dasharray:7 5}.series{fill:none;stroke-width:2.8}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">Updated model to 2026-04: expected inflation level</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#555">Baseline 2026-02 expected inflation 2.6% plus IRF response</text>',
    ]
    for tick in range(6):
        y = top + tick * plot_height / 5
        value = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" font-size="11" text-anchor="end">{value:.2f}</text>')
    base_y = y_pos(EXPECTED_INFLATION_BASELINE)
    thr_y = y_pos(ANCHORING_THRESHOLD_LEVEL)
    lines.append(f'<line class="base" x1="{left}" y1="{base_y:.2f}" x2="{width-right}" y2="{base_y:.2f}"/>')
    lines.append(f'<line class="thr" x1="{left}" y1="{thr_y:.2f}" x2="{width-right}" y2="{thr_y:.2f}"/>')
    lines.append(f'<text x="{width-right-4}" y="{thr_y-7:.2f}" font-size="12" text-anchor="end">anchoring threshold: 3.0%</text>')
    lines.append(
        f'<text x="{width-right-4}" y="{base_y+15:.2f}" font-size="12" text-anchor="end">'
        f'baseline: {EXPECTED_INFLATION_BASELINE:.1f}%</text>'
    )
    for h in [0, 3, 6, 9, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}"/>')
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


def write_report(raw, data, adf, lag_table, port, lm, irf_summary, scen_summary, oil_sigma, selected_lag):
    lines = [
        "# 2026년 4월까지 포함한 업데이트 모형 전체 분석",
        "",
        "## 데이터와 전처리",
        "",
        f"- 원자료 파일: `{RAW_COMBINED.relative_to(ROOT).as_posix()}`",
        f"- VAR 데이터 파일: `{VAR_DATA.relative_to(ROOT).as_posix()}`",
        f"- 원자료 기간: {raw['date'].min().date()} ~ {raw['date'].max().date()}",
        f"- VAR 표본: {data['date'].min().date()} ~ {data['date'].max().date()}, {len(data)}개",
        "- CPI는 로그 STL 계절조정 후 월간 로그차분을 사용했습니다.",
        "- 기대인플레이션은 수준값을 사용했습니다.",
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
    lines.append(f"- 잔차 진단과 기존 설계를 고려한 업데이트 모형 시차: VAR({selected_lag})")

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
            f"- 유가 1표준편차 충격: {oil_sigma:.4f}%p",
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
            "## 시나리오 시뮬레이션",
            "",
            "- 기준 기대인플레이션은 2026년 2월 실제값 2.6%로 설정했습니다.",
            "- 2026년 5월 유가는 사용자가 지정한 105달러를 사용하고, 2026년 6월부터 시나리오별 유가 경로를 적용했습니다.",
            "- 앵커링 훼손 임계선은 기대인플레이션 3.0%입니다.",
            "",
            "| 시나리오 | 최대 수준 | 95% CI | 최대 시점 | 임계선 최초 도달 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, r in scen_summary.iterrows():
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
            f"- ADF: `{ADF_OUT.relative_to(ROOT).as_posix()}`",
            f"- 시차 선택: `{LAG_OUT.relative_to(ROOT).as_posix()}`",
            f"- 잔차 진단: `{RESID_OUT.relative_to(ROOT).as_posix()}`",
            f"- IRF: `{IRF_OUT.relative_to(ROOT).as_posix()}`",
            f"- IRF 요약: `{IRF_SUMMARY_OUT.relative_to(ROOT).as_posix()}`",
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
    selected_lag = 3
    port, lm, res = diagnostics(data, selected_lag)
    oil_sigma = float(np.linalg.cholesky(res.sigma_u)[0, 0])
    irf_df, irf_summary = irf_outputs(res, oil_sigma)
    _, scenario_sim_df, scenario_summary = scenario_sim(irf_df, oil_sigma)
    svg_line_plot(scenario_sim_df)
    write_report(raw, data, adf, lag_table, port, lm, irf_summary, scenario_summary, oil_sigma, selected_lag)
    print(f"Saved: {REPORT_OUT}")
    print(f"Saved: {VAR_DATA}")
    print(f"Saved: {IRF_SUMMARY_OUT}")
    print(f"Saved: {SCENARIO_SUMMARY_OUT}")
    print(f"Saved: {SCENARIO_FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
