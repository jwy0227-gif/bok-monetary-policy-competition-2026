from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.api import VAR


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "svar_main_level_201002_202602.csv"
REPORT_DIR = ROOT / "reports"
RESULT_PATH = REPORT_DIR / "var_lag_selection.csv"
REPORT_PATH = REPORT_DIR / "var_lag_selection.md"

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
    best_aic = int(out.loc[out["aic"].idxmin(), "lag"])
    best_bic = int(out.loc[out["bic"].idxmin(), "lag"])
    best_hqic = int(out.loc[out["hqic"].idxmin(), "lag"])

    out["aic_best"] = out["lag"].eq(best_aic)
    out["bic_best"] = out["lag"].eq(best_bic)
    out["hqic_best"] = out["lag"].eq(best_hqic)
    out.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    if best_bic == best_hqic:
        selected = best_bic
        reason = "BIC와 HQIC가 같은 시차를 선택하므로 더 보수적인 기준에 따라 해당 시차를 우선 선택합니다."
    elif best_aic == best_hqic:
        selected = best_hqic
        reason = "AIC와 HQIC가 같은 시차를 선택하므로 해당 시차를 우선 선택합니다."
    else:
        selected = best_bic
        reason = "기준들이 갈릴 경우 표본 수와 파라미터 수를 고려해 가장 보수적인 BIC 기준 시차를 우선 선택합니다."

    lines = [
        "# VAR 시차 선택",
        "",
        f"- 사용 데이터: `{DATA_PATH.relative_to(ROOT).as_posix()}`",
        f"- 표본: {df['date'].min().date()} ~ {df['date'].max().date()}",
        f"- 관측치: {len(df)}개",
        "- 변수 순서: 국제유가 → 환율 → CPI → 기대인플레이션",
        "",
        "## 변수",
        "",
    ]
    for col in VAR_COLUMNS:
        lines.append(f"- `{col}`: {LABELS[col]}")

    lines.extend(
        [
            "",
            "## 정보기준 비교",
            "",
            "| 시차 | AIC | BIC | HQIC | FPE | 선택 표시 |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in out.iterrows():
        marks = []
        if row["aic_best"]:
            marks.append("AIC")
        if row["bic_best"]:
            marks.append("BIC")
        if row["hqic_best"]:
            marks.append("HQIC")
        mark_text = ", ".join(marks) if marks else ""
        lines.append(
            f"| {int(row['lag'])} | {row['aic']:.4f} | {row['bic']:.4f} | "
            f"{row['hqic']:.4f} | {row['fpe']:.6g} | {mark_text} |"
        )

    lines.extend(
        [
            "",
            "## 선택",
            "",
            f"- AIC 선택 시차: {best_aic}",
            f"- BIC 선택 시차: {best_bic}",
            f"- HQIC 선택 시차: {best_hqic}",
            f"- 우선 선택 시차: {selected}",
            "",
            reason,
            "",
            "다음 단계의 잔차 자기상관 LM 검정에서 자기상관이 남으면 시차를 늘려 재추정합니다.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Saved: {RESULT_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
