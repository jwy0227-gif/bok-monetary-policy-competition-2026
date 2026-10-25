from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IRF_PATH = ROOT / "reports" / "irf_oil_shock_cholesky_var3.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "simulation"

OIL_OUT = REPORT_DIR / "custom_oil_path_preupdate.csv"
SIM_OUT = REPORT_DIR / "custom_oil_path_preupdate_expected_inflation.csv"
SUMMARY_OUT = REPORT_DIR / "custom_oil_path_preupdate_summary.csv"
REPORT_OUT = REPORT_DIR / "custom_oil_path_preupdate_simulation.md"
FIGURE_OUT = FIGURE_DIR / "custom_oil_path_preupdate_expected_inflation.svg"

HORIZON = 24
OIL_SHOCK_SIGMA = 8.045945115677503
BASELINE_EXPECTED_INFLATION = 2.6
ANCHORING_THRESHOLD = 3.0
BASE_OIL_202602 = 68.51


def oil_price_for_month(date: pd.Timestamp) -> float:
    if date == pd.Timestamp("2026-03-01"):
        return 126.71
    if date == pd.Timestamp("2026-04-01"):
        return 105.30
    if date == pd.Timestamp("2026-05-01"):
        return 105.00
    if date == pd.Timestamp("2026-06-01"):
        return 100.00
    if pd.Timestamp("2026-07-01") <= date <= pd.Timestamp("2026-09-01"):
        return 90.00
    if pd.Timestamp("2026-10-01") <= date <= pd.Timestamp("2026-12-01"):
        return 87.00
    return 87.00


def build_oil_path() -> pd.DataFrame:
    months = pd.date_range("2026-03-01", periods=HORIZON + 1, freq="MS")
    prices = [oil_price_for_month(month) for month in months]
    prevs = [BASE_OIL_202602] + prices[:-1]
    rows = []
    for horizon, (month, price, prev) in enumerate(zip(months, prices, prevs, strict=True)):
        dln = 100 * np.log(price / prev)
        rows.append(
            {
                "horizon": horizon,
                "date": month,
                "oil_price_usd_bbl": price,
                "previous_oil_price_usd_bbl": prev,
                "dln_oil_pct": dln,
                "shock_size_sigma": dln / OIL_SHOCK_SIGMA,
            }
        )
    return pd.DataFrame(rows)


def simulate(oil_path: pd.DataFrame) -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp = irf[irf["response"] == "expected_inflation"].sort_values("horizon").set_index("horizon")
    response_irf = exp["irf"].reindex(range(HORIZON + 1)).to_numpy()
    lower_irf = exp["lower_95"].reindex(range(HORIZON + 1)).to_numpy()
    upper_irf = exp["upper_95"].reindex(range(HORIZON + 1)).to_numpy()
    shocks = oil_path["shock_size_sigma"].to_numpy()

    rows = []
    for h in range(HORIZON + 1):
        response = lower = upper = 0.0
        for j, shock in enumerate(shocks):
            if h - j >= 0:
                response += shock * response_irf[h - j]
                lo = shock * lower_irf[h - j]
                hi = shock * upper_irf[h - j]
                lower += min(lo, hi)
                upper += max(lo, hi)
        rows.append(
            {
                "horizon": h,
                "date": pd.Timestamp("2026-03-01") + pd.DateOffset(months=h),
                "expected_inflation_response_pp": response,
                "lower_95_response_pp": lower,
                "upper_95_response_pp": upper,
                "expected_inflation_level_pct": BASELINE_EXPECTED_INFLATION + response,
                "lower_95_level_pct": BASELINE_EXPECTED_INFLATION + lower,
                "upper_95_level_pct": BASELINE_EXPECTED_INFLATION + upper,
                "above_threshold": BASELINE_EXPECTED_INFLATION + response >= ANCHORING_THRESHOLD,
            }
        )
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    max_row = sim.loc[sim["expected_inflation_level_pct"].idxmax()]
    threshold_rows = sim[sim["above_threshold"]]
    return pd.DataFrame(
        [
            {
                "max_level_pct": float(max_row["expected_inflation_level_pct"]),
                "max_level_lower_95_pct": float(max_row["lower_95_level_pct"]),
                "max_level_upper_95_pct": float(max_row["upper_95_level_pct"]),
                "max_response_pp": float(max_row["expected_inflation_response_pp"]),
                "max_horizon": int(max_row["horizon"]),
                "first_threshold_horizon": int(threshold_rows["horizon"].iloc[0]) if not threshold_rows.empty else pd.NA,
                "level_t": float(sim.loc[sim["horizon"] == 0, "expected_inflation_level_pct"].iloc[0]),
                "level_t6": float(sim.loc[sim["horizon"] == 6, "expected_inflation_level_pct"].iloc[0]),
                "level_t12": float(sim.loc[sim["horizon"] == 12, "expected_inflation_level_pct"].iloc[0]),
                "level_t24": float(sim.loc[sim["horizon"] == 24, "expected_inflation_level_pct"].iloc[0]),
            }
        ]
    )


def svg_plot(sim: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 980, 500
    left, right, top, bottom = 82, 32, 64, 60
    plot_width, plot_height = width - left - right, height - top - bottom
    ymin = min(float(sim["lower_95_level_pct"].min()), BASELINE_EXPECTED_INFLATION, ANCHORING_THRESHOLD)
    ymax = max(float(sim["upper_95_level_pct"].max()), BASELINE_EXPECTED_INFLATION, ANCHORING_THRESHOLD)
    pad = (ymax - ymin) * 0.10
    ymin -= pad
    ymax += pad

    def x_pos(h: int) -> float:
        return left + h * plot_width / HORIZON

    def y_pos(v: float) -> float:
        return top + plot_height - (v - ymin) * plot_height / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd}.base{stroke:#333;stroke-width:1.2}.thr{stroke:#111;stroke-width:1.6;stroke-dasharray:7 5}.band{fill:#0072B2;fill-opacity:.16}.series{fill:none;stroke:#0072B2;stroke-width:2.8}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">Custom oil path: expected inflation level</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#555">Pre-update VAR(3), baseline 2.6%, threshold 3.0%</text>',
    ]
    for tick in range(6):
        y = top + tick * plot_height / 5
        val = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" font-size="11" text-anchor="end">{val:.2f}</text>')
    base_y, thr_y = y_pos(BASELINE_EXPECTED_INFLATION), y_pos(ANCHORING_THRESHOLD)
    lines.append(f'<line class="base" x1="{left}" y1="{base_y:.2f}" x2="{width-right}" y2="{base_y:.2f}"/>')
    lines.append(f'<line class="thr" x1="{left}" y1="{thr_y:.2f}" x2="{width-right}" y2="{thr_y:.2f}"/>')
    lines.append(f'<text x="{width-right-4}" y="{thr_y-7:.2f}" font-size="12" text-anchor="end">anchoring threshold: 3.0%</text>')
    for h in [0, 3, 6, 9, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height-28}" font-size="11" text-anchor="middle">T+{h}</text>')
    upper = [(x_pos(int(r.horizon)), y_pos(float(r.upper_95_level_pct))) for r in sim.itertuples()]
    lower = [(x_pos(int(r.horizon)), y_pos(float(r.lower_95_level_pct))) for r in sim.iloc[::-1].itertuples()]
    band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
    points = " ".join(f"{x_pos(int(r.horizon)):.2f},{y_pos(float(r.expected_inflation_level_pct)):.2f}" for r in sim.itertuples())
    lines.append(f'<polygon class="band" points="{band}"/>')
    lines.append(f'<polyline class="series" points="{points}"/>')
    lines.append("</svg>")
    FIGURE_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_report(oil_path: pd.DataFrame, sim: pd.DataFrame, summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    threshold = "도달 안 함" if pd.isna(row["first_threshold_horizon"]) else f"T+{int(row['first_threshold_horizon'])}"
    lines = [
        "# 사용자 지정 유가 경로: 업데이트 전 모형 기대인플레이션 동향",
        "",
        "- 사용 모형: 2026년 2월까지 추정한 업데이트 전 VAR(3)",
        "- 기준 기대인플레이션: 2026년 2월 2.6%",
        "- 앵커링 임계선: 3.0%",
        "",
        "## 유가 경로",
        "",
        "| 시점 | 유가 | 전월 대비 변동률 | 충격 크기 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, r in oil_path.iloc[:10].iterrows():
        lines.append(f"| T+{int(r['horizon'])} ({r['date'].strftime('%Y-%m')}) | {r['oil_price_usd_bbl']:.2f} | {r['dln_oil_pct']:.4f}% | {r['shock_size_sigma']:.4f}σ |")
    lines.extend(
        [
            "",
            "## 기대인플레이션 결과",
            "",
            f"- 최대 수준: {row['max_level_pct']:.4f}% "
            f"[95% CI: {row['max_level_lower_95_pct']:.4f}%, {row['max_level_upper_95_pct']:.4f}%]",
            f"- 최대 시점: T+{int(row['max_horizon'])}",
            f"- 3.0% 임계선 최초 도달: {threshold}",
            "",
            "| 시점 | 기대인플레이션 수준 | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for h in [0, 1, 2, 3, 4, 5, 6, 12, 24]:
        r = sim.loc[sim["horizon"] == h].iloc[0]
        lines.append(
            f"| T+{h} | {r['expected_inflation_level_pct']:.4f}% | "
            f"[{r['lower_95_level_pct']:.4f}%, {r['upper_95_level_pct']:.4f}%] |"
        )
    lines.extend(
        [
            "",
            "## 파일",
            "",
            f"- 유가 경로: `{OIL_OUT.relative_to(ROOT).as_posix()}`",
            f"- 시뮬레이션: `{SIM_OUT.relative_to(ROOT).as_posix()}`",
            f"- 요약: `{SUMMARY_OUT.relative_to(ROOT).as_posix()}`",
            f"- 그래프: `{FIGURE_OUT.relative_to(ROOT).as_posix()}`",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    oil_path = build_oil_path()
    sim = simulate(oil_path)
    summary = summarize(sim)
    oil_path.to_csv(OIL_OUT, index=False, encoding="utf-8-sig")
    sim.to_csv(SIM_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    svg_plot(sim)
    write_report(oil_path, sim, summary)
    print(f"Saved: {REPORT_OUT}")
    print(f"Saved: {FIGURE_OUT}")
    print(f"Saved: {SIM_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
