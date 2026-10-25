from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.api import VAR


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "svar_main_level_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
RESULT_PATH = REPORT_DIR / "var_lag3_residual_diagnostics.csv"
REPORT_PATH = REPORT_DIR / "var_lag3_residual_diagnostics.md"

LAG = 3
VAR_COLUMNS = ["dln_oil_pct", "dln_fx_pct", "dln_cpi_pct", "expected_inflation"]
LABELS = {
    "dln_oil_pct": "Δln(두바이유) × 100",
    "dln_fx_pct": "Δln(원/달러 환율) × 100",
    "dln_cpi_pct": "Δln(CPI) × 100",
    "expected_inflation": "기대인플레이션 수준",
}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df.set_index("date")[VAR_COLUMNS]
    model = VAR(y)
    result = model.fit(LAG)

    rows = []
    for nlags in [6, 12, 18, 24]:
        test = result.test_whiteness(nlags=nlags, adjusted=True)
        rows.append(
            {
                "var_lag": LAG,
                "test_lags": nlags,
                "test": "Adjusted Portmanteau residual autocorrelation test",
                "test_statistic": float(test.test_statistic),
                "critical_value_5pct": float(test.crit_value),
                "p_value": float(test.pvalue),
                "df": int(test.df),
                "decision_5pct": "reject autocorrelation-free residuals"
                if test.pvalue < 0.05
                else "do not reject autocorrelation-free residuals",
            }
        )

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    primary = diagnostics.loc[diagnostics["test_lags"] == 12].iloc[0]
    if primary["p_value"] < 0.05:
        conclusion = (
            "12개월까지의 잔차 자기상관이 없다는 귀무가설을 5% 기준에서 기각합니다. "
            "즉 VAR(3) 잔차에 자기상관이 남아 있을 가능성이 있습니다."
        )
    else:
        conclusion = (
            "12개월까지의 잔차 자기상관이 없다는 귀무가설을 5% 기준에서 기각하지 못합니다. "
            "즉 VAR(3)의 잔차 자기상관 문제는 뚜렷하지 않습니다."
        )

    lines = [
        "# VAR(3) 추정 및 잔차 자기상관 진단",
        "",
        f"- 사용 데이터: `{DATA_PATH.relative_to(ROOT).as_posix()}`",
        f"- 표본: {df['date'].min().date()} ~ {df['date'].max().date()}",
        f"- 원 관측치: {len(df)}개",
        f"- VAR 시차: {LAG}",
        f"- 추정에 사용된 관측치: {result.nobs}개",
        "- 변수 순서: 국제유가 → 환율 → CPI → 기대인플레이션",
        "",
        "## 추정 변수",
        "",
    ]
    for col in VAR_COLUMNS:
        lines.append(f"- `{col}`: {LABELS[col]}")

    lines.extend(
        [
            "",
            "## 모형 정보기준",
            "",
            f"- AIC: {result.aic:.4f}",
            f"- BIC: {result.bic:.4f}",
            f"- HQIC: {result.hqic:.4f}",
            f"- FPE: {result.fpe:.6g}",
            "",
            "## 잔차 자기상관 검정",
            "",
            "검정은 VAR 시스템 전체 잔차에 대한 adjusted Portmanteau test입니다. 귀무가설은 '해당 시차까지 잔차 자기상관이 없다'입니다.",
            "",
            "| 검정 시차 | 통계량 | 5% 임계값 | p-value | df | 판정 |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in diagnostics.iterrows():
        verdict = "자기상관 있음" if row["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(
            f"| {int(row['test_lags'])} | {row['test_statistic']:.3f} | "
            f"{row['critical_value_5pct']:.3f} | {row['p_value']:.4f} | "
            f"{int(row['df'])} | {verdict} |"
        )

    lines.extend(
        [
            "",
            "## 결론",
            "",
            conclusion,
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Saved: {RESULT_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
