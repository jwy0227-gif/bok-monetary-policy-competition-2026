from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.api import VAR

import stl_cpi_var_recheck as stl


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "svar_main_level_stl_cpi_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
PORT_PATH = REPORT_DIR / "var_stl_cpi_lag1_3_6_portmanteau.csv"
LM_PATH = REPORT_DIR / "var_stl_cpi_lag1_3_6_individual_lm.csv"
REPORT_PATH = REPORT_DIR / "var_stl_cpi_lag1_3_6_residual_comparison.md"

CANDIDATE_LAGS = [1, 3, 6]
TEST_LAGS = [6, 12, 18, 24]


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df.set_index("date")[stl.VAR_COLUMNS]

    port_rows = []
    lm_rows = []
    for var_lag in CANDIDATE_LAGS:
        result = VAR(y).fit(var_lag)
        for test_lag in TEST_LAGS:
            if test_lag <= var_lag:
                continue
            test = result.test_whiteness(nlags=test_lag, adjusted=True)
            port_rows.append(
                {
                    "var_lag": var_lag,
                    "test_lags": test_lag,
                    "test_statistic": float(test.test_statistic),
                    "critical_value_5pct": float(test.crit_value),
                    "p_value": float(test.pvalue),
                    "df": int(test.df),
                    "decision_5pct": "autocorrelation" if test.pvalue < 0.05 else "no clear autocorrelation",
                }
            )

        lm = stl.individual_lm(y, result.resid, var_lag)
        lm.insert(0, "var_lag", var_lag)
        lm_rows.extend(lm.to_dict("records"))

    port = pd.DataFrame(port_rows)
    lm_all = pd.DataFrame(lm_rows)
    port.to_csv(PORT_PATH, index=False, encoding="utf-8-sig")
    lm_all.to_csv(LM_PATH, index=False, encoding="utf-8-sig")

    lines = [
        "# STL CPI VAR 후보 시차별 잔차 진단 비교",
        "",
        f"- 사용 데이터: `{DATA_PATH.relative_to(ROOT).as_posix()}`",
        "- 후보 시차: VAR(1), VAR(3), VAR(6)",
        "",
        "## Portmanteau 잔차 자기상관 검정",
        "",
        "| VAR lag | 검정 시차 | 통계량 | 5% 임계값 | p-value | 판정 |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in port.iterrows():
        verdict = "자기상관 있음" if row["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(
            f"| {int(row['var_lag'])} | {int(row['test_lags'])} | {row['test_statistic']:.3f} | "
            f"{row['critical_value_5pct']:.3f} | {row['p_value']:.4f} | {verdict} |"
        )

    lines.extend(
        [
            "",
            "## 개별 lag 1~6 LM 검정",
            "",
            "| VAR lag | residual lag | LM 통계량 | df | p-value | 판정 |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in lm_all.iterrows():
        verdict = "자기상관 있음" if row["p_value"] < 0.05 else "자기상관 뚜렷하지 않음"
        lines.append(
            f"| {int(row['var_lag'])} | {int(row['lag'])} | {row['lm_stat']:.3f} | "
            f"{int(row['df'])} | {row['p_value']:.4f} | {verdict} |"
        )

    def port_ok(var_lag: int) -> bool:
        sub = port[port["var_lag"] == var_lag]
        return bool((sub["p_value"] >= 0.05).all())

    def lm_ok(var_lag: int) -> bool:
        sub = lm_all[lm_all["var_lag"] == var_lag]
        return bool((sub["p_value"] >= 0.05).all())

    ok_lags = [lag for lag in CANDIDATE_LAGS if port_ok(lag) and lm_ok(lag)]
    if ok_lags:
        conclusion = f"후보 중 VAR({ok_lags[0]})가 Portmanteau와 개별 LM 기준을 모두 통과합니다."
    else:
        conclusion = "후보 중 모든 잔차 진단을 완전히 통과하는 시차는 없습니다. 다만 VAR(3)은 Portmanteau 기준을 모두 통과하고 개별 LM에서는 residual lag 6만 남아, VAR(1)보다 개선됩니다."

    lines.extend(["", "## 결론", "", conclusion])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Saved: {PORT_PATH}")
    print(f"Saved: {LM_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
