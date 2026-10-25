from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "bok_style"
MPL_CONFIG_DIR = ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


IRF_PATH = REPORT_DIR / "irf_oil_shock_cholesky_var3.csv"
PATHS_CSV = FIGURE_DIR / "hormuz_scenario_paths.csv"

HORIZON = 24
ANCHORING_REFERENCE = 3.0
DPI = 600

SIZES_CM = [
    (7.0, 4.0),
    (8.0, 4.6),
    (9.0, 5.1),
    (10.0, 5.7),
]

SCENARIO_COLORS = {
    "조기 종전": "#1f4e79",
    "봉쇄 장기화": "#8064a2",
    "에너지 시설 타격": "#c0504d",
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


def load_irf() -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp = irf[irf["response"] == "expected_inflation"].sort_values("horizon").copy()
    return exp[(exp["horizon"] >= 0) & (exp["horizon"] <= HORIZON)].reset_index(drop=True)


def load_paths() -> pd.DataFrame:
    return pd.read_csv(PATHS_CSV)


def save_compact_irf(irf: pd.DataFrame, width_cm: float, height_cm: float) -> Path:
    out = FIGURE_DIR / f"irf_1sd_expected_inflation_{width_cm:g}x{height_cm:g}cm_compact_600dpi.png"
    fig, ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    h = irf["horizon"].to_numpy()
    y = irf["irf"].to_numpy()
    max_idx = int(np.argmax(y))
    max_h = int(h[max_idx])
    max_y = float(y[max_idx])

    ax.axhspan(0, max_y * 1.18, xmin=1 / HORIZON, xmax=20 / HORIZON, color="#d9e6f2", alpha=0.35)
    ax.plot(h, y, color="#202020", linewidth=1.5)
    ax.axhline(0, color="#555555", linestyle=(0, (3, 3)), linewidth=0.65)
    ax.scatter([max_h], [max_y], color="#202020", s=16, zorder=5)
    ax.text(max_h + 1.0, max_y + 0.006, "0.06%p", fontsize=6.8, color="#333333")

    ax.set_xlabel("충격 후 경과 개월", fontsize=7.2, labelpad=2)
    ax.set_ylabel("반응(%p)", fontsize=7.2, labelpad=2)
    ax.set_xlim(0, 24)
    ax.set_ylim(min(-0.012, float(y.min()) - 0.006), max(float(y.max()) * 1.35, 0.085))
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["T", "T+4", "T+8", "T+12", "T+16", "T+20", "T+24"], fontsize=6.4)
    ax.tick_params(axis="y", labelsize=6.4, length=2.5, pad=1.5)
    ax.grid(True, color="#e8e8e8", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.20)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out


def save_compact_paths(paths: pd.DataFrame, width_cm: float, height_cm: float) -> Path:
    out = FIGURE_DIR / f"hormuz_scenario_paths_{width_cm:g}x{height_cm:g}cm_compact_600dpi.png"
    fig, ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for scenario, color in SCENARIO_COLORS.items():
        sub = paths[paths["시나리오"] == scenario].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["expected_inflation_level_pct"],
            color=color,
            linewidth=1.45,
            label=scenario,
        )

    ax.axhline(ANCHORING_REFERENCE, color="#555555", linestyle=(0, (4, 3)), linewidth=0.75)
    ax.text(18.2, ANCHORING_REFERENCE + 0.015, "3.0% 참고선", fontsize=6.5, color="#444444")

    ax.set_xlabel("충격 후 경과 개월", fontsize=7.2, labelpad=2)
    ax.set_ylabel("수준(%)", fontsize=7.2, labelpad=2)
    ax.set_xlim(0, 24)
    ax.set_ylim(2.54, 3.12)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["T", "T+4", "T+8", "T+12", "T+16", "T+20", "T+24"], fontsize=6.4)
    ax.tick_params(axis="y", labelsize=6.4, length=2.5, pad=1.5)
    ax.grid(True, color="#e8e8e8", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right", handlelength=1.4, borderaxespad=0.2)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.20)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    irf = load_irf()
    paths = load_paths()
    created = []
    for width_cm, height_cm in SIZES_CM:
        created.append(save_compact_irf(irf, width_cm, height_cm))
        created.append(save_compact_paths(paths, width_cm, height_cm))
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
