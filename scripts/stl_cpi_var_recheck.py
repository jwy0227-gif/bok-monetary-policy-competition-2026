from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import STL


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

STL_COMPONENTS_PATH = PROCESSED_DIR / "cpi_stl_components_201001_202602.csv"
STL_VAR_DATA_PATH = PROCESSED_DIR / "svar_main_level_stl_cpi_201002_202602.csv"
LAG_SELECTION_PATH = REPORT_DIR / "var_lag_selection_stl_cpi.csv"
DIAGNOSTICS_PATH = REPORT_DIR / "var_stl_cpi_residual_diagnostics.csv"
LM_PATH = REPORT_DIR / "var_stl_cpi_individual_lm_lag1_6.csv"
REPORT_PATH = REPORT_DIR / "stl_cpi_var_recheck.md"

VAR_COLUMNS = ["dln_oil_pct", "dln_fx_pct", "dln_cpi_stl_pct", "expected_inflation"]
LABELS = {
    "dln_oil_pct": "Δln(두바이유) × 100",
    "dln_fx_pct": "Δln(원/달러 환율) × 100",
    "dln_cpi_stl_pct": "Δln(STL 계절조정 CPI) × 100",
    "expected_inflation": "기대인플레이션 수준",
}


def build_stl_data() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    log_cpi = np.log(raw["cpi"])
    stl = STL(log_cpi, period=12, robust=True).fit()
    raw["log_cpi"] = log_cpi
    raw["log_cpi_stl_trend"] = stl.trend
    raw["log_cpi_stl_seasonal"] = stl.seasonal
    raw["log_cpi_stl_resid"] = stl.resid
    raw["log_cpi_stl_sa"] = raw["log_cpi"] - raw["log_cpi_stl_seasonal"]
    raw["cpi_stl_sa"] = np.exp(raw["log_cpi_stl_sa"])

    components = raw[
        [
            "date",
            "cpi",
            "cpi_stl_sa",
            "log_cpi",
            "log_cpi_stl_sa",
            "log_cpi_stl_trend",
            "log_cpi_stl_seasonal",
            "log_cpi_stl_resid",
        ]
    ].copy()
    components.to_csv(STL_COMPONENTS_PATH, index=False, encoding="utf-8-sig")

    raw["dln_oil_pct"] = 100 * np.log(raw["dubai_oil_usd_bbl"]).diff()
    raw["dln_fx_pct"] = 100 * np.log(raw["exchange_rate_krw_usd"]).diff()
    raw["dln_cpi_stl_pct"] = 100 * raw["log_cpi_stl_sa"].diff()
    raw["expected_inflation"] = raw["expected_inflation_1y"]

    var_data = raw[["date", *VAR_COLUMNS]].dropna().copy()
    var_data.to_csv(STL_VAR_DATA_PATH, index=False, encoding="utf-8-sig")
    return var_data


def lag_selection(var_data: pd.DataFrame) -> pd.DataFrame:
    y = var_data.set_index("date")[VAR_COLUMNS]
    model = VAR(y)
    rows = []
    for lag in range(1, 7):
        result = model.fit(lag)
        rows.append(
            {
                "lag": lag,
                "nobs": int(result.nobs),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "hqic": float(result.hqic),
                "fpe": float(result.fpe),
            }
        )
    out = pd.DataFrame(rows)
    out["aic_best"] = out["lag"].eq(int(out.loc[out["aic"].idxmin(), "lag"]))
    out["bic_best"] = out["lag"].eq(int(out.loc[out["bic"].idxmin(), "lag"]))
    out["hqic_best"] = out["lag"].eq(int(out.loc[out["hqic"].idxmin(), "lag"]))
    out.to_csv(LAG_SELECTION_PATH, index=False, encoding="utf-8-sig")
    return out


def residual_diagnostics(var_data: pd.DataFrame, selected_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = var_data.set_index("date")[VAR_COLUMNS]
    result = VAR(y).fit(selected_lag)

    diag_rows = []
    for nlags in [6, 12, 18, 24]:
        test = result.test_whiteness(nlags=nlags, adjusted=True)
        diag_rows.append(
            {
                "var_lag": selected_lag,
                "test_lags": nlags,
                "test_statistic": float(test.test_statistic),
                "critical_value_5pct": float(test.crit_value),
                "p_value": float(test.pvalue),
                "df": int(test.df),
                "decision_5pct": "autocorrelation" if test.pvalue < 0.05 else "no clear autocorrelation",
            }
        )
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig")

    lm = individual_lm(y, result.resid, selected_lag)
    lm.to_csv(LM_PATH, index=False, encoding="utf-8-sig")
    return diagnostics, lm


def system_lm_for_lag(endog: pd.DataFrame, resid: pd.DataFrame, var_lag: int, test_lag: int) -> dict[str, float | int | str]:
    rows_y = []
    rows_x_base = []
    rows_x_test = []

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
    x_restricted = np.asarray(rows_x_base, dtype=float)
    tested = np.asarray(rows_x_test, dtype=float)
    x_unrestricted = np.column_stack([x_restricted, tested])

    e_restricted = []
    e_unrestricted = []
    for eq in range(y.shape[1]):
        yr = y[:, eq]
        beta_r = np.linalg.lstsq(x_restricted, yr, rcond=None)[0]
        beta_u = np.linalg.lstsq(x_unrestricted, yr, rcond=None)[0]
        e_restricted.append(yr - x_restricted @ beta_r)
        e_unrestricted.append(yr - x_unrestricted @ beta_u)

    nobs = len(y)
    n_equations = y.shape[1]
    df = n_equations * n_equations
    err_r = np.column_stack(e_restricted)
    err_u = np.column_stack(e_unrestricted)
    sigma_r = (err_r.T @ err_r) / nobs
    sigma_u = (err_u.T @ err_u) / nobs
    det_r = max(float(np.linalg.det(sigma_r)), 1e-300)
    det_u = max(float(np.linalg.det(sigma_u)), 1e-300)
    lm_stat = max(0.0, nobs * (math.log(det_r) - math.log(det_u)))
    p_value = float(chi2.sf(lm_stat, df))
    return {
        "lag": test_lag,
        "nobs": nobs,
        "lm_stat": lm_stat,
        "df": df,
        "p_value": p_value,
        "decision_5pct": "autocorrelation" if p_value < 0.05 else "no clear autocorrelation",
    }


def individual_lm(endog: pd.DataFrame, resid: pd.DataFrame, var_lag: int) -> pd.DataFrame:
    return pd.DataFrame([system_lm_for_lag(endog, resid, var_lag, lag) for lag in range(1, 7)])


def write_report(var_data: pd.DataFrame, lag_table: pd.DataFrame, diagnostics: pd.DataFrame, lm: pd.DataFrame) -> None:
    best_aic = int(lag_table.loc[lag_table["aic"].idxmin(), "lag"])
    best_bic = int(lag_table.loc[lag_table["bic"].idxmin(), "lag"])
    best_hqic = int(lag_table.loc[lag_table["hqic"].idxmin(), "lag"])
    selected_lag = best_aic

    lines = [
        "# STL 계절조정 CPI 기반 VAR 재점검",
        "",
        "## STL 처리",
        "",
        "CPI 수준값의 로그에 STL을 적용해 계절요인을 제거한 뒤 `100 * Δln(CPI_sa)`를 만들었습니다.",
        "",
        f"- STL 구성요소 파일: `{STL_COMPONENTS_PATH.relative_to(ROOT).as_posix()}`",
        f"- VAR 데이터 파일: `{STL_VAR_DATA_PATH.relative_to(ROOT).as_posix()}`",
        f"- 표본: {var_data['date'].min().date()} ~ {var_data['date'].max().date()}",
        f"- 관측치: {len(var_data)}개",
        "",
        "## VAR 시차 선택",
        "",
        "| lag | nobs | AIC | BIC | HQIC | FPE | 선택 표시 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in lag_table.iterrows():
        marks = []
        if row["aic_best"]:
            marks.append("AIC")
        if row["bic_best"]:
            marks.append("BIC")
        if row["hqic_best"]:
            marks.append("HQIC")
        lines.append(
            f"| {int(row['lag'])} | {int(row['nobs'])} | {row['aic']:.4f} | {row['bic']:.4f} | "
            f"{row['hqic']:.4f} | {row['fpe']:.6g} | {', '.join(marks)} |"
        )

    lines.extend(
        [
            "",
            f"- AIC 선택 시차: {best_aic}",
            f"- BIC 선택 시차: {best_bic}",
            f"- HQIC 선택 시차: {best_hqic}",
            f"- 잔차 진단 기준 시차: {selected_lag}",
            "",
            "기존 분석에서 AIC 선택 시차를 중시했으므로, 잔차 진단은 AIC가 선택한 시차로 수행했습니다.",
            "",
            "## 잔차 자기상관 진단",
            "",
            "| 검정 시차 | 통계량 | 5% 임계값 | p-value | df | 판정 |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in diagnostics.iterrows():
        verdict = "자기상관 있음" if row["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(
            f"| {int(row['test_lags'])} | {row['test_statistic']:.3f} | "
            f"{row['critical_value_5pct']:.3f} | {row['p_value']:.4f} | {int(row['df'])} | {verdict} |"
        )

    lines.extend(
        [
            "",
            "## 개별 lag 1~6 LM 검정",
            "",
            "| lag | LM 통계량 | df | p-value | 판정 |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in lm.iterrows():
        verdict = "자기상관 있음" if row["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(f"| {int(row['lag'])} | {row['lm_stat']:.3f} | {int(row['df'])} | {row['p_value']:.4f} | {verdict} |")

    white_12 = float(diagnostics.loc[diagnostics["test_lags"] == 12, "p_value"].iloc[0])
    lm_sig = lm[lm["p_value"] < 0.05]
    if white_12 >= 0.05 and lm_sig.empty:
        conclusion = "STL 계절조정 CPI를 사용하면 12개월 Portmanteau와 개별 lag 1~6 LM 기준에서 잔차 자기상관 문제가 뚜렷하지 않습니다."
    elif white_12 >= 0.05:
        sig_lags = ", ".join(str(int(x)) for x in lm_sig["lag"])
        conclusion = f"12개월 Portmanteau 기준은 통과하지만, 개별 LM에서는 lag {sig_lags}에서 자기상관이 남습니다."
    else:
        conclusion = "STL 계절조정 CPI를 사용해도 12개월 Portmanteau 기준에서 잔차 자기상관이 남습니다."

    lines.extend(["", "## 결론", "", conclusion])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    var_data = build_stl_data()
    lag_table = lag_selection(var_data)
    selected_lag = int(lag_table.loc[lag_table["aic"].idxmin(), "lag"])
    diagnostics, lm = residual_diagnostics(var_data, selected_lag)
    write_report(var_data, lag_table, diagnostics, lm)
    print(f"Saved: {STL_COMPONENTS_PATH}")
    print(f"Saved: {STL_VAR_DATA_PATH}")
    print(f"Saved: {LAG_SELECTION_PATH}")
    print(f"Saved: {DIAGNOSTICS_PATH}")
    print(f"Saved: {LM_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
