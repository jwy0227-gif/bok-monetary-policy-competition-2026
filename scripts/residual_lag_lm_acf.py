from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, pacf


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "svar_main_level_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "residual_diagnostics"
LM_PATH = REPORT_DIR / "var_lag3_individual_lm_lag1_6.csv"
ACF_PATH = REPORT_DIR / "var_lag3_residual_acf_pacf.csv"
REPORT_PATH = REPORT_DIR / "var_lag3_individual_lm_acf.md"

VAR_LAG = 3
MAX_TEST_LAG = 6
ACF_MAX_LAG = 24
VAR_COLUMNS = ["dln_oil_pct", "dln_fx_pct", "dln_cpi_pct", "expected_inflation"]
LABELS = {
    "dln_oil_pct": "유가",
    "dln_fx_pct": "환율",
    "dln_cpi_pct": "CPI",
    "expected_inflation": "기대인플레이션",
}


def system_lm_for_lag(endog: pd.DataFrame, resid: pd.DataFrame, lag: int) -> dict[str, float | int | str]:
    resid_by_date = resid.copy()
    rows_y = []
    rows_x_base = []
    rows_x_test = []

    # Test whether residuals at t-lag explain residuals at t after controlling
    # for the original VAR(p) regressors, y_{t-1}, ..., y_{t-p}.
    for t in range(VAR_LAG + lag, len(endog)):
        date_t = endog.index[t]
        date_test = endog.index[t - lag]
        if date_t not in resid_by_date.index or date_test not in resid_by_date.index:
            continue

        rows_y.append(resid_by_date.loc[date_t, VAR_COLUMNS].to_numpy(dtype=float))
        base = [1.0]
        for p in range(1, VAR_LAG + 1):
            base.extend(endog.iloc[t - p].to_numpy(dtype=float))
        rows_x_base.append(base)
        rows_x_test.append(resid_by_date.loc[date_test, VAR_COLUMNS].to_numpy(dtype=float))

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
        "lag": lag,
        "nobs": nobs,
        "lm_stat": lm_stat,
        "df": df,
        "p_value": p_value,
        "decision_5pct": "autocorrelation" if p_value < 0.05 else "no clear autocorrelation",
    }


def acf_pacf_rows(resid: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows = []
    conf = 1.96 / math.sqrt(len(resid))
    for column in VAR_COLUMNS:
        values = resid[column].to_numpy()
        acf_values = acf(values, nlags=ACF_MAX_LAG, fft=False)
        pacf_values = pacf(values, nlags=ACF_MAX_LAG, method="ywm")
        for lag in range(1, ACF_MAX_LAG + 1):
            rows.append(
                {
                    "variable": column,
                    "label": LABELS[column],
                    "lag": lag,
                    "acf": float(acf_values[lag]),
                    "pacf": float(pacf_values[lag]),
                    "approx_95_bound": conf,
                    "acf_outside_95": abs(float(acf_values[lag])) > conf,
                    "pacf_outside_95": abs(float(pacf_values[lag])) > conf,
                }
            )
    return rows


def svg_stem_plot(rows: pd.DataFrame, value_col: str, title_suffix: str, out_path: Path) -> None:
    width = 1050
    panel_height = 170
    left = 72
    right = 30
    top = 48
    bottom = 36
    gap = 30
    height = top + bottom + len(VAR_COLUMNS) * panel_height + (len(VAR_COLUMNS) - 1) * gap
    plot_width = width - left - right

    all_values = rows[value_col].to_numpy()
    max_abs = max(0.25, float(np.max(np.abs(all_values))) * 1.15)

    def x_pos(lag: int) -> float:
        return left + (lag - 1) * plot_width / (ACF_MAX_LAG - 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}.stem{stroke:#0072B2;stroke-width:2}.bound{stroke:#999;stroke-width:1;stroke-dasharray:5 4}</style>',
        f'<text x="{left}" y="28" font-size="18" font-weight="700">VAR(3) residual {title_suffix}</text>',
    ]

    for panel_idx, col in enumerate(VAR_COLUMNS):
        sub = rows[rows["variable"] == col]
        panel_top = top + panel_idx * (panel_height + gap)

        def y_pos(value: float) -> float:
            return panel_top + panel_height / 2 - value * (panel_height / 2) / max_abs

        zero_y = y_pos(0.0)
        upper_y = y_pos(float(sub["approx_95_bound"].iloc[0]))
        lower_y = y_pos(-float(sub["approx_95_bound"].iloc[0]))
        lines.append(f'<line class="grid" x1="{left}" y1="{panel_top}" x2="{width - right}" y2="{panel_top}"/>')
        lines.append(f'<line class="grid" x1="{left}" y1="{panel_top + panel_height}" x2="{width - right}" y2="{panel_top + panel_height}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}"/>')
        lines.append(f'<line class="bound" x1="{left}" y1="{upper_y:.2f}" x2="{width - right}" y2="{upper_y:.2f}"/>')
        lines.append(f'<line class="bound" x1="{left}" y1="{lower_y:.2f}" x2="{width - right}" y2="{lower_y:.2f}"/>')
        lines.append(f'<text x="{left}" y="{panel_top - 8}" font-size="13" font-weight="700">{LABELS[col]}</text>')
        lines.append(f'<text x="{left - 10}" y="{zero_y + 4:.2f}" font-size="11" text-anchor="end">0</text>')

        for _, row in sub.iterrows():
            x = x_pos(int(row["lag"]))
            y = y_pos(float(row[value_col]))
            color = "#D55E00" if bool(row[f"{value_col}_outside_95"]) else "#0072B2"
            lines.append(f'<line x1="{x:.2f}" y1="{zero_y:.2f}" x2="{x:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{color}"/>')
        for lag in [1, 6, 12, 18, 24]:
            x = x_pos(lag)
            lines.append(f'<text x="{x:.2f}" y="{panel_top + panel_height + 18}" font-size="10" text-anchor="middle">{lag}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df.set_index("date")[VAR_COLUMNS]
    result = VAR(y).fit(VAR_LAG)
    resid = result.resid

    lm = pd.DataFrame([system_lm_for_lag(y, resid, lag) for lag in range(1, MAX_TEST_LAG + 1)])
    lm.to_csv(LM_PATH, index=False, encoding="utf-8-sig")

    acf_pacf = pd.DataFrame(acf_pacf_rows(resid))
    acf_pacf.to_csv(ACF_PATH, index=False, encoding="utf-8-sig")
    svg_stem_plot(acf_pacf, "acf", "ACF", FIGURE_DIR / "var_lag3_residual_acf.svg")
    svg_stem_plot(acf_pacf, "pacf", "PACF", FIGURE_DIR / "var_lag3_residual_pacf.svg")

    sig_lm = lm[lm["p_value"] < 0.05]
    if sig_lm.empty:
        lm_conclusion = "개별 lag 1~6 LM 검정에서는 5% 기준으로 유의한 잔차 자기상관이 확인되지 않았습니다."
    else:
        lags = ", ".join(str(int(x)) for x in sig_lm["lag"])
        lm_conclusion = f"개별 lag 1~6 LM 검정에서는 lag {lags}에서 5% 기준 잔차 자기상관이 확인됩니다."

    focus = acf_pacf[(acf_pacf["lag"] <= 6) & (acf_pacf["acf_outside_95"] | acf_pacf["pacf_outside_95"])]
    lines = [
        "# VAR(3) 잔차 개별 lag LM 및 ACF/PACF",
        "",
        f"- 사용 데이터: `{DATA_PATH.relative_to(ROOT).as_posix()}`",
        f"- VAR 시차: {VAR_LAG}",
        f"- 잔차 관측치: {len(resid)}개",
        "",
        "## 개별 lag별 LM 검정",
        "",
        "각 검정은 원래 VAR(3)의 설명변수들을 통제한 뒤, 특정 lag의 잔차항이 현재 잔차를 추가로 설명하는지 보는 시스템 LM 검정입니다. 귀무가설은 '해당 lag의 잔차 자기상관이 없다'입니다.",
        "",
        "| lag | LM 통계량 | df | p-value | 판정 |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in lm.iterrows():
        verdict = "자기상관 있음" if row["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(f"| {int(row['lag'])} | {row['lm_stat']:.3f} | {int(row['df'])} | {row['p_value']:.4f} | {verdict} |")

    lines.extend(
        [
            "",
            "## ACF/PACF 그래프",
            "",
            f"- ACF: `{(FIGURE_DIR / 'var_lag3_residual_acf.svg').relative_to(ROOT).as_posix()}`",
            f"- PACF: `{(FIGURE_DIR / 'var_lag3_residual_pacf.svg').relative_to(ROOT).as_posix()}`",
            "",
            "주황색 막대는 대략적 95% 경계 밖에 있는 lag입니다.",
            "",
            "## lag 1~6에서 경계 밖 ACF/PACF",
            "",
        ]
    )
    if focus.empty:
        lines.append("lag 1~6 구간에서 개별 변수 잔차 ACF/PACF가 대략적 95% 경계를 벗어나는 지점은 없습니다.")
    else:
        lines.extend(["| 변수 | lag | ACF | PACF | 경계 밖 |", "| --- | ---: | ---: | ---: | --- |"])
        for _, row in focus.iterrows():
            flags = []
            if row["acf_outside_95"]:
                flags.append("ACF")
            if row["pacf_outside_95"]:
                flags.append("PACF")
            lines.append(
                f"| {row['label']} | {int(row['lag'])} | {row['acf']:.3f} | {row['pacf']:.3f} | {', '.join(flags)} |"
            )

    lines.extend(["", "## 결론", "", lm_conclusion])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Saved: {LM_PATH}")
    print(f"Saved: {ACF_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {FIGURE_DIR / 'var_lag3_residual_acf.svg'}")
    print(f"Saved: {FIGURE_DIR / 'var_lag3_residual_pacf.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
