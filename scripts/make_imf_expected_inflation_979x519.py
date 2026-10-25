from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports" / "figures" / "bok_style_imf"
DATA_PATH = ROOT / "reports" / "figures" / "bok_style_imf" / "imf_scenario_paths.csv"
MPL_CONFIG_DIR = ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


WIDTH_CM = 9.79
HEIGHT_CM = 5.19
DPI = 600
REFERENCE_LINE = 3.0

OUT = FIGURE_DIR / "imf_expected_inflation_paths_979x519_600dpi.png"
OUT_NO_TITLE = FIGURE_DIR / "imf_expected_inflation_paths_979x519_notitle_600dpi.png"
OUT_LARGE_LEGEND = FIGURE_DIR / "imf_expected_inflation_paths_979x519_large_legend_600dpi.png"
OUT_LARGE_LEGEND_NO_TITLE = FIGURE_DIR / "imf_expected_inflation_paths_979x519_large_legend_notitle_600dpi.png"

SCENARIOS = [
    ("기준 시나리오", "기준시나리오", "#1f4e79"),
    ("고유가 장기화 시나리오", "고유가 장기화", "#c55a11"),
    ("유가 안정 시나리오", "유가 안정", "#008c68"),
]


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
    data = pd.read_csv(DATA_PATH)
    fig, ax = plt.subplots(figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    line_width = 1.45 if not large_legend else 1.35
    for scenario, label, color in SCENARIOS:
        sub = data[data["scenario"] == scenario].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["expected_inflation_level_pct"],
            color=color,
            linewidth=line_width,
            label=label,
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
    draw(OUT, include_title=True)
    draw(OUT_NO_TITLE, include_title=False)
    draw(OUT_LARGE_LEGEND, include_title=True, large_legend=True)
    draw(OUT_LARGE_LEGEND_NO_TITLE, include_title=False, large_legend=True)
    for path in [OUT, OUT_NO_TITLE, OUT_LARGE_LEGEND, OUT_LARGE_LEGEND_NO_TITLE]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
