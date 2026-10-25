from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = ROOT / "data" / "processed" / "svar_main_level_201002_202602.csv"
STL_PATH = ROOT / "data" / "processed" / "svar_main_level_stl_cpi_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
CSV_PATH = REPORT_DIR / "adf_stl_cpi.csv"
REPORT_PATH = REPORT_DIR / "adf_stl_cpi.md"


def run_adf(series: pd.Series) -> dict[str, float | int]:
    stat, pvalue, usedlag, nobs, crit, icbest = adfuller(series.dropna(), regression="c", autolag="AIC")
    return {
        "adf_stat": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(usedlag),
        "nobs": int(nobs),
        "critical_1pct": float(crit["1%"]),
        "critical_5pct": float(crit["5%"]),
        "critical_10pct": float(crit["10%"]),
        "icbest": float(icbest),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(ORIGINAL_PATH, parse_dates=["date"])
    stl = pd.read_csv(STL_PATH, parse_dates=["date"])

    rows = []
    specs = [
        ("dln_cpi_pct", "원계열 CPI 월간 로그차분", original["dln_cpi_pct"]),
        ("dln_cpi_stl_pct", "STL 계절조정 CPI 월간 로그차분", stl["dln_cpi_stl_pct"]),
    ]
    for variable, label, series in specs:
        result = run_adf(series)
        rows.append(
            {
                "variable": variable,
                "label": label,
                **result,
                "decision_5pct": "정상성 있음" if result["p_value"] < 0.05 else "단위근 가능",
                "decision_10pct": "정상성 있음" if result["p_value"] < 0.10 else "단위근 가능",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    lines = [
        "# STL 계절조정 CPI ADF 검정",
        "",
        "| 변수 | ADF 통계량 | p-value | 사용 lag | 5% 판정 | 10% 판정 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['label']} | {row['adf_stat']:.4f} | {row['p_value']:.4f} | "
            f"{int(row['used_lag'])} | {row['decision_5pct']} | {row['decision_10pct']} |"
        )

    stl_p = float(out.loc[out["variable"] == "dln_cpi_stl_pct", "p_value"].iloc[0])
    if stl_p < 0.05:
        conclusion = "STL 계절조정 CPI 월간 로그차분은 5% 기준에서 정상성을 보입니다."
    else:
        conclusion = "STL 계절조정 CPI 월간 로그차분은 5% 기준에서 단위근 가능성이 남습니다."
    lines.extend(["", "## 결론", "", conclusion])

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
