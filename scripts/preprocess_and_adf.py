from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

PROCESSED_PATH = PROCESSED_DIR / "svar_preprocessed_201002_202602.csv"
MAIN_PATH = PROCESSED_DIR / "svar_main_level_201002_202602.csv"
ROBUST_PATH = PROCESSED_DIR / "svar_robust_diffexp_201002_202602.csv"
ADF_PATH = REPORT_DIR / "adf_results.csv"
REPORT_PATH = REPORT_DIR / "preprocessing_adf.md"


ADF_SERIES = {
    "dln_oil_pct": "Δln(두바이유) × 100",
    "dln_fx_pct": "Δln(원/달러 환율) × 100",
    "dln_cpi_pct": "Δln(CPI) × 100",
    "expected_inflation_level": "기대인플레이션 수준",
    "d_expected_inflation": "Δ기대인플레이션",
}


def run_adf(series: pd.Series) -> dict[str, float | int]:
    clean = series.dropna()
    stat, pvalue, usedlag, nobs, crit, icbest = adfuller(clean, regression="c", autolag="AIC")
    return {
        "adf_stat": stat,
        "p_value": pvalue,
        "used_lag": usedlag,
        "nobs": nobs,
        "critical_1pct": crit["1%"],
        "critical_5pct": crit["5%"],
        "critical_10pct": crit["10%"],
        "icbest": icbest,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    data = raw.copy()
    data["dln_oil_pct"] = 100 * np.log(data["dubai_oil_usd_bbl"]).diff()
    data["dln_fx_pct"] = 100 * np.log(data["exchange_rate_krw_usd"]).diff()
    data["dln_cpi_pct"] = 100 * np.log(data["cpi"]).diff()
    data["expected_inflation_level"] = data["expected_inflation_1y"]
    data["d_expected_inflation"] = data["expected_inflation_1y"].diff()
    data["ru_war_dummy"] = (
        (data["date"] >= pd.Timestamp("2022-02-01"))
        & (data["date"] <= pd.Timestamp("2023-06-01"))
    ).astype(int)

    adf_rows = []
    for column, label in ADF_SERIES.items():
        result = run_adf(data[column])
        판정 = "정상성 있음" if result["p_value"] < 0.05 else "단위근 가능"
        adf_rows.append({"variable": column, "label": label, **result, "decision": 판정})

    adf = pd.DataFrame(adf_rows)
    processed = data[
        [
            "date",
            "dln_oil_pct",
            "dln_fx_pct",
            "dln_cpi_pct",
            "expected_inflation_level",
            "d_expected_inflation",
            "ru_war_dummy",
        ]
    ].dropna().copy()
    processed["expected_inflation_for_var"] = processed["expected_inflation_level"]
    processed["expected_inflation_for_var_robust"] = processed["d_expected_inflation"]

    main = processed[
        ["date", "dln_oil_pct", "dln_fx_pct", "dln_cpi_pct", "expected_inflation_level"]
    ].copy()
    main = main.rename(columns={"expected_inflation_level": "expected_inflation"})

    robust = processed[
        ["date", "dln_oil_pct", "dln_fx_pct", "dln_cpi_pct", "d_expected_inflation"]
    ].copy()
    robust = robust.rename(columns={"d_expected_inflation": "expected_inflation"})
    robust["ru_war_dummy"] = processed["ru_war_dummy"].values

    processed.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    main.to_csv(MAIN_PATH, index=False, encoding="utf-8-sig")
    robust.to_csv(ROBUST_PATH, index=False, encoding="utf-8-sig")
    adf.to_csv(ADF_PATH, index=False, encoding="utf-8-sig")

    report_lines = [
        "# 데이터 전처리 및 ADF 검정",
        "",
        "## 변환",
        "",
        "- 유가: `100 * (ln(oil_t) - ln(oil_{t-1}))`",
        "- 환율: `100 * (ln(fx_t) - ln(fx_{t-1}))`",
        "- CPI: `100 * (ln(cpi_t) - ln(cpi_{t-1}))`",
        "- 기대인플레이션: 수준과 전월 대비 차분을 모두 검정",
        "",
        f"- 전처리 후 표본: {processed['date'].min().date()} ~ {processed['date'].max().date()}",
        f"- 전처리 후 행 개수: {len(processed)}개",
        "",
        "## ADF 검정 결과",
        "",
        "| 변수 | ADF 통계량 | p-value | 사용 lag | 판정 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in adf.iterrows():
        report_lines.append(
            f"| {row['label']} | {format_float(row['adf_stat'])} | {format_float(row['p_value'])} | "
            f"{int(row['used_lag'])} | {row['decision']} |"
        )

    report_lines.extend(
        [
            "",
            "## 기대인플레이션 처리 결정",
            "",
            "메인 분석에서는 기대인플레이션율 수준(level, %)을 사용합니다.",
            "",
            "이유: 연구 질문이 '유가 충격이 기대인플레이션율을 몇 %p 움직이는가'이므로 수준 변수의 해석이 가장 직접적입니다. "
            "ADF 검정에서는 기대인플레이션 수준의 p-value가 0.106으로 5% 기준 단위근 가능성이 있으므로, 마지막 강건성 분석에서 "
            "전월 대비 차분(Δ기대인플레이션)을 사용한 대체 모형을 함께 보고합니다.",
            "",
            "- 메인 VAR 투입용 기대인플레이션 열: `expected_inflation_level`",
            "- 강건성 VAR 투입용 기대인플레이션 열: `d_expected_inflation`",
            "- 러-우 전쟁 더미: `ru_war_dummy`, 2022-02부터 2023-06까지 1, 그 외 0",
            "- 전체 전처리 파일의 `expected_inflation_for_var`는 메인 분석 기준에 맞춰 수준 변수로 반영했습니다.",
            "",
            "## 저장 파일",
            "",
            f"- 전체 전처리 파일: `{PROCESSED_PATH.relative_to(ROOT).as_posix()}`",
            f"- 메인 분석용 파일: `{MAIN_PATH.relative_to(ROOT).as_posix()}`",
            f"- 강건성 분석용 파일: `{ROBUST_PATH.relative_to(ROOT).as_posix()}`",
            "",
            "강건성 분석에서는 같은 종속변수 구성에서 `ru_war_dummy`를 외생변수로 포함한 VAR과 포함하지 않은 VAR을 비교합니다.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8-sig")

    print(f"Saved: {PROCESSED_PATH}")
    print(f"Saved: {MAIN_PATH}")
    print(f"Saved: {ROBUST_PATH}")
    print(f"Saved: {ADF_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
