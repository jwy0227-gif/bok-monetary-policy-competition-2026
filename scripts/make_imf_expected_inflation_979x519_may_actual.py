from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "bok_style_imf"
MPL_CONFIG_DIR = ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


IRF_PATH = REPORT_DIR / "irf_oil_shock_cholesky_var3.csv"
PATHS_CSV = FIGURE_DIR / "imf_scenario_paths_may_actual.csv"
SUMMARY_CSV = FIGURE_DIR / "imf_scenario_table_may_actual.csv"

WIDTH_CM = 9.79
HEIGHT_CM = 5.19
DPI = 600
HORIZON = 24
START_DATE = pd.Timestamp("2026-03-01")
BASE_OIL_202602 = 68.51
BASELINE_EXPECTED_INFLATION = 2.6
OIL_SHOCK_SIGMA = 8.045945115677503

OUT = FIGURE_DIR / "imf_expected_inflation_paths_may_actual_979x519_600dpi.png"
OUT_NO_TITLE = FIGURE_DIR / "imf_expected_inflation_paths_may_actual_979x519_notitle_600dpi.png"
OUT_LARGE_LEGEND = FIGURE_DIR / "imf_expected_inflation_paths_may_actual_979x519_large_legend_600dpi.png"
OUT_LARGE_LEGEND_NO_TITLE = FIGURE_DIR / "imf_expected_inflation_paths_may_actual_979x519_large_legend_notitle_600dpi.png"

SCENARIOS = [
    {
        "scenario": "기준 시나리오",
        "label": "기준시나리오",
        "oil_2026q2": 100.0,
        "oil_2026q3": 90.0,
        "oil_2026q4": 87.0,
        "oil_2027": 87.0,
        "color": "#1f4e79",
    },
    {
        "scenario": "고유가 장기화 시나리오",
        "label": "고유가 장기화",
        "oil_2026q2": 105.0,
        "oil_2026q3": 105.0,
        "oil_2026q4": 105.0,
        "oil_2027": 105.0,
        "color": "#c55a11",
    },
    {
        "scenario": "유가 안정 시나리오",
        "label": "유가 안정",
        "oil_2026q2": 95.0,
        "oil_2026q3": 85.0,
        "oil_2026q4": 80.0,
        "oil_2027": 80.0,
        "color": "#008c68",
    },
]

ACTUAL_OIL = {
    pd.Timestamp("2026-03-01"): 126.71,
    pd.Timestamp("2026-04-01"): 105.30,
    pd.Timestamp("2026-05-01"): 105.00,
}


def cm_to_inch(value: float) -> float:
    return value / 2.54


def set_korean_font() -> None:
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def scenario_price(spec: dict[str, object], month: pd.Timestamp) -> float:
    if month in ACTUAL_OIL:
        return ACTUAL_OIL[month]
    if month == pd.Timestamp("2026-06-01"):
        return float(spec["oil_2026q2"])
    if pd.Timestamp("2026-07-01") <= month <= pd.Timestamp("2026-09-01"):
        return float(spec["oil_2026q3"])
    if pd.Timestamp("2026-10-01") <= month <= pd.Timestamp("2026-12-01"):
        return float(spec["oil_2026q4"])
    return float(spec["oil_2027"])


def build_oil_paths() -> pd.DataFrame:
    months = pd.date_range(START_DATE, periods=HORIZON + 1, freq="MS")
    rows = []
    for spec in SCENARIOS:
        prices = [scenario_price(spec, month) for month in months]
        prevs = [BASE_OIL_202602] + prices[:-1]
        for h, (month, price, prev) in enumerate(zip(months, prices, prevs, strict=True)):
            dln = 100 * np.log(price / prev)
            rows.append(
                {
                    "scenario": spec["scenario"],
                    "horizon": h,
                    "date": month,
                    "oil_price_usd_bbl": price,
                    "dln_oil_pct": dln,
                    "shock_size_sigma": dln / OIL_SHOCK_SIGMA,
                }
            )
    return pd.DataFrame(rows)


def simulate(paths: pd.DataFrame) -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp = irf[irf["response"] == "expected_inflation"].sort_values("horizon").set_index("horizon")
    exp_irf = exp["irf"].reindex(range(HORIZON + 1)).to_numpy()
    rows = []
    for spec in SCENARIOS:
        scenario = str(spec["scenario"])
        shocks = paths[paths["scenario"] == scenario].sort_values("horizon")["shock_size_sigma"].to_numpy()
        for h in range(HORIZON + 1):
            response = 0.0
            for j, shock in enumerate(shocks):
                if h - j >= 0:
                    response += shock * exp_irf[h - j]
            rows.append(
                {
                    "scenario": scenario,
                    "horizon": h,
                    "date": START_DATE + pd.DateOffset(months=h),
                    "expected_inflation_response_pp": response,
                    "expected_inflation_level_pct": BASELINE_EXPECTED_INFLATION + response,
                }
            )
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        scenario = str(spec["scenario"])
        sub = sim[sim["scenario"] == scenario]
        max_row = sub.loc[sub["expected_inflation_response_pp"].idxmax()]
        rows.append(
            {
                "시나리오": spec["label"],
                "최대 반응(%p)": float(max_row["expected_inflation_response_pp"]),
                "최대 반응 시점": f"T+{int(max_row['horizon'])}",
                "최대 기대인플레이션 수준(%)": float(max_row["expected_inflation_level_pct"]),
            }
        )
    return pd.DataFrame(rows)


def draw(sim: pd.DataFrame, output: Path, include_title: bool, large_legend: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    line_width = 1.45 if not large_legend else 1.35
    for spec in SCENARIOS:
        sub = sim[sim["scenario"] == spec["scenario"]].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["expected_inflation_level_pct"],
            color=str(spec["color"]),
            linewidth=line_width,
            label=str(spec["label"]),
        )

    if include_title:
        ax.set_title("유가 시나리오별 기대인플레이션 경로", loc="left", fontsize=8.4, weight="bold", pad=5)

    axis_font = 7.0 if not large_legend else 7.2
    tick_font = 6.2 if not large_legend else 6.3
    legend_font = 5.9 if not large_legend else 6.9

    ax.set_xlabel("충격 후 경과 개월", fontsize=axis_font, labelpad=7)
    ax.set_ylabel("기대인플레이션 수준(%)", fontsize=axis_font, labelpad=2)
    ax.set_xlim(0, 24.9)
    ax.set_ylim(2.55, 3.04)
    ax.set_xticks([0, 3, 6, 9, 12, 18, 24])
    ax.set_xticklabels(["T", "T+3", "T+6", "T+9", "T+12", "T+18", "T+24"], fontsize=tick_font)
    ax.set_yticks([2.6, 2.7, 2.8, 2.9, 3.0])
    ax.tick_params(axis="y", labelsize=tick_font, length=2.2, pad=1.5)
    ax.grid(True, color="#e6e6e6", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.legend(
        frameon=False,
        fontsize=legend_font,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34 if large_legend else -0.31),
        ncol=3,
        handlelength=1.25,
        columnspacing=0.75,
        borderaxespad=0,
    )

    top = 0.88 if include_title else 0.96
    bottom = 0.43 if large_legend else 0.39
    fig.subplots_adjust(left=0.15, right=0.98, top=top, bottom=bottom)
    fig.savefig(output, dpi=DPI, facecolor="white")
    plt.close(fig)


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    paths = build_oil_paths()
    sim = simulate(paths)
    summary = summarize(sim)
    sim.to_csv(PATHS_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    draw(sim, OUT, include_title=True)
    draw(sim, OUT_NO_TITLE, include_title=False)
    draw(sim, OUT_LARGE_LEGEND, include_title=True, large_legend=True)
    draw(sim, OUT_LARGE_LEGEND_NO_TITLE, include_title=False, large_legend=True)
    for path in [PATHS_CSV, SUMMARY_CSV, OUT, OUT_NO_TITLE, OUT_LARGE_LEGEND, OUT_LARGE_LEGEND_NO_TITLE]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
