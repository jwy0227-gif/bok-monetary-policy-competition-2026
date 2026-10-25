from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports" / "figures" / "bok_style_imf"
DATA_PATH = ROOT / "reports" / "scenario_quarterly_2026_oil_paths_main_model.csv"
MPL_CONFIG_DIR = ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


OUT = FIGURE_DIR / "imf_oil_scenario_paths_979x519_600dpi.png"
OUT_NO_TITLE = FIGURE_DIR / "imf_oil_scenario_paths_979x519_notitle_600dpi.png"
OUT_LARGE_LABEL = FIGURE_DIR / "imf_oil_scenario_paths_979x519_large_legend_600dpi.png"
OUT_LARGE_LABEL_NO_TITLE = FIGURE_DIR / "imf_oil_scenario_paths_979x519_large_legend_notitle_600dpi.png"

WIDTH_CM = 9.79
HEIGHT_CM = 5.19
DPI = 600

COLORS = {
    "기준 시나리오": "#1f4e79",
    "고유가 장기화 시나리오": "#c55a11",
    "유가 안정 시나리오": "#008c68",
}
LABELS = {
    "기준 시나리오": "IMF 기준",
    "고유가 장기화 시나리오": "고유가 장기화",
    "유가 안정 시나리오": "유가 안정",
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


def draw(output: Path, include_title: bool, large_legend: bool = False) -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for scenario, color in COLORS.items():
        sub = data[data["scenario"] == scenario].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["oil_price_usd_bbl"],
            color=color,
            linewidth=1.45,
            label=LABELS[scenario],
        )

    if include_title:
        ax.set_title("시나리오별 유가 전망", loc="left", fontsize=8.6, weight="bold", pad=5)

    axis_font = 7.0 if not large_legend else 7.4
    tick_font = 6.2 if not large_legend else 6.5
    legend_font = 5.8 if not large_legend else 7.1
    ax.set_xlabel("전망 시점", fontsize=axis_font, labelpad=2)
    ax.set_ylabel("달러/배럴", fontsize=axis_font, labelpad=2)
    ax.set_xlim(0, 24)
    ax.set_ylim(75, 132)
    ax.set_xticks([0, 3, 6, 9, 12, 18, 24])
    ax.set_xticklabels(["2026.3", "6", "9", "12", "2027.3", "9", "2028.3"], fontsize=tick_font)
    ax.set_yticks([80, 90, 100, 110, 120, 130])
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
        bbox_to_anchor=(0.5, -0.23 if large_legend else -0.20),
        ncol=3,
        handlelength=1.3,
        columnspacing=0.75 if large_legend else 0.9,
        borderaxespad=0,
    )

    top = 0.90 if include_title else 0.96
    bottom = 0.37 if large_legend else 0.31
    fig.subplots_adjust(left=0.13, right=0.98, top=top, bottom=bottom)
    fig.savefig(output, dpi=DPI, facecolor="white")
    plt.close(fig)


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    draw(OUT, include_title=True)
    draw(OUT_NO_TITLE, include_title=False)
    draw(OUT_LARGE_LABEL, include_title=True, large_legend=True)
    draw(OUT_LARGE_LABEL_NO_TITLE, include_title=False, large_legend=True)
    print(OUT)
    print(OUT_NO_TITLE)
    print(OUT_LARGE_LABEL)
    print(OUT_LARGE_LABEL_NO_TITLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
