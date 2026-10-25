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
SCENARIO_PATH = REPORT_DIR / "scenario_quarterly_2026_expected_inflation_main_model.csv"
SUMMARY_PATH = REPORT_DIR / "scenario_quarterly_2026_expected_inflation_summary_main_model.csv"

TABLE_CSV = FIGURE_DIR / "imf_scenario_table.csv"
TABLE_TSV = FIGURE_DIR / "imf_scenario_table_for_docs.tsv"
TABLE_MD = FIGURE_DIR / "imf_scenario_table_for_docs.md"
TABLE_PNG = FIGURE_DIR / "imf_scenario_table.png"
IRF_PNG = FIGURE_DIR / "imf_irf_1sd_expected_inflation.png"
PATHS_PNG = FIGURE_DIR / "imf_scenario_paths.png"
PATHS_CSV = FIGURE_DIR / "imf_scenario_paths.csv"

BASELINE_EXPECTED_INFLATION = 2.6
REFERENCE_LINE = 3.0
HORIZON = 24
IRF_MAX_FOR_SCALING = 0.06
NOTE = "주: 각 시나리오는 1표준편차 유가 충격에 대한 SVAR 충격반응함수를 충격 규모에 비례해 조정하여 산출. 자료: 저자 추정."

SCENARIO_ORDER = ["기준 시나리오", "고유가 장기화 시나리오", "유가 안정 시나리오"]
SCENARIO_LABELS = {
    "기준 시나리오": "IMF 기준 시나리오",
    "고유가 장기화 시나리오": "고유가 장기화 시나리오",
    "유가 안정 시나리오": "유가 안정 시나리오",
}
SCENARIO_COLORS = {
    "기준 시나리오": "#1f4e79",
    "고유가 장기화 시나리오": "#c55a11",
    "유가 안정 시나리오": "#008c68",
}
INTERPRETATIONS = {
    "기준 시나리오": "기대인플레이션 상승 압력은 존재하나 3%에는 미달",
    "고유가 장기화 시나리오": "기대인플레이션 불안정 위험 확대",
    "유가 안정 시나리오": "유가 하락 전제하 상승 압력 제한",
}


def set_korean_font() -> None:
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        vals = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.4f}"
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_irf() -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp = irf[irf["response"] == "expected_inflation"].sort_values("horizon").copy()
    exp = exp[(exp["horizon"] >= 0) & (exp["horizon"] <= HORIZON)].copy()
    return exp[["horizon", "irf"]].reset_index(drop=True)


def load_paths() -> pd.DataFrame:
    paths = pd.read_csv(SCENARIO_PATH, parse_dates=["date"])
    paths = paths[paths["horizon"].between(0, HORIZON)].copy()
    paths["scenario_label"] = paths["scenario"].map(SCENARIO_LABELS)
    paths = paths[paths["scenario"].isin(SCENARIO_ORDER)].copy()
    return paths


def build_table() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_PATH, parse_dates=["max_response_date", "first_threshold_date"])
    rows = []
    for scenario in SCENARIO_ORDER:
        row = summary[summary["scenario"] == scenario].iloc[0]
        first_threshold = row.get("first_threshold_horizon")
        if pd.isna(first_threshold):
            threshold_text = "도달 안 함"
        else:
            threshold_text = f"T+{int(first_threshold)}"
        rows.append(
            {
                "시나리오": SCENARIO_LABELS[scenario],
                "충격 규모": f"{float(row['max_response_pp']) / IRF_MAX_FOR_SCALING:.2f}배",
                "최대 반응(%p)": float(row["max_response_pp"]),
                "최대 반응 시점": f"T+{int(row['max_response_horizon'])}",
                "최대 기대인플레이션 수준(%)": float(row["max_level_pct"]),
                "3% 참고선 최초 도달": threshold_text,
                "해석": INTERPRETATIONS[scenario],
            }
        )
    return pd.DataFrame(rows)


def save_table_png(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15.0, 4.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    display = table.copy()
    display["최대 반응(%p)"] = display["최대 반응(%p)"].map(lambda x: f"{x:.4f}")
    display["최대 기대인플레이션 수준(%)"] = display["최대 기대인플레이션 수준(%)"].map(lambda x: f"{x:.4f}")
    ax.set_title("표 1. IMF 유가 시나리오별 기대인플레이션 반응", loc="left", fontsize=14, weight="bold", pad=12)
    mpl_table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0, 0.30, 1, 0.58],
        colWidths=[0.15, 0.08, 0.10, 0.10, 0.15, 0.12, 0.30],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9.2)
    for (r, c), cell in mpl_table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#6f6f6f")
        if r == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f7f7f7")
            if c == 6:
                cell.set_text_props(ha="left")
    ax.text(0, 0.12, NOTE, transform=ax.transAxes, fontsize=8.6, color="#555555", ha="left", va="bottom")
    fig.savefig(TABLE_PNG, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_irf_graph(irf: pd.DataFrame) -> None:
    h = irf["horizon"].to_numpy()
    y = irf["irf"].to_numpy()
    max_idx = int(np.argmax(y))
    max_h = int(h[max_idx])
    max_y = float(y[max_idx])
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhspan(0, max_y * 1.18, xmin=1 / HORIZON, xmax=20 / HORIZON, color="#d9e6f2", alpha=0.35, label="유의한 반응 기간(T+1~T+20)")
    ax.plot(h, y, color="#202020", linewidth=2.2, label="IRF 추정치")
    ax.axhline(0, color="#555555", linestyle=(0, (3, 3)), linewidth=0.9)
    ax.scatter([max_h], [max_y], color="#202020", s=42, zorder=5)
    ax.annotate(
        "최대 0.06%p",
        xy=(max_h, max_y),
        xytext=(max_h + 3.2, max_y + 0.014),
        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.75, "shrinkA": 2, "shrinkB": 5},
        fontsize=9.5,
        color="#333333",
    )
    ax.set_title("그림 1. 유가 충격 1표준편차에 대한 기대인플레이션 반응", loc="left", fontsize=14, weight="bold", pad=11)
    ax.set_xlabel("충격 후 경과 개월")
    ax.set_ylabel("기대인플레이션 반응(%p)")
    ax.set_xlim(0, 24)
    ax.set_ylim(min(-0.015, float(y.min()) - 0.01), max(float(y.max()) * 1.42, 0.09))
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["T", "T+4", "T+8", "T+12", "T+16", "T+20", "T+24"])
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    ax.text(0, -0.30, "주: 신뢰구간은 별도 추정 결과 이용 시 추가.\n" + NOTE, transform=ax.transAxes, fontsize=8.5, color="#555555", ha="left", va="top")
    fig.savefig(IRF_PNG, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_paths_graph(paths: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for scenario in SCENARIO_ORDER:
        sub = paths[paths["scenario"] == scenario].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["expected_inflation_level_pct"],
            linewidth=2.2,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
    ax.axhline(REFERENCE_LINE, color="#555555", linestyle=(0, (4, 3)), linewidth=1.1)
    ax.text(24.25, REFERENCE_LINE + 0.006, "3.0% 참고선", fontsize=9.7, color="#444444", va="bottom")
    ax.set_title("그림 2. IMF 유가 시나리오별 기대인플레이션 경로", loc="left", fontsize=14, weight="bold", pad=11)
    ax.set_xlabel("충격 후 경과 개월")
    ax.set_ylabel("기대인플레이션 수준(%)")
    ax.set_ylim(2.48, 3.06)
    ax.set_xlim(0, 25.4)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["T", "T+4", "T+8", "T+12", "T+16", "T+20", "T+24"])
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.54), borderaxespad=0)
    ax.text(0, -0.25, NOTE, transform=ax.transAxes, fontsize=8.5, color="#555555", ha="left", va="top")
    fig.savefig(PATHS_PNG, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    irf = load_irf()
    paths = load_paths()
    table = build_table()
    table.to_csv(TABLE_CSV, index=False, encoding="utf-8-sig")
    table.to_csv(TABLE_TSV, index=False, sep="\t", encoding="utf-8-sig")
    TABLE_MD.write_text(dataframe_to_markdown(table) + "\n", encoding="utf-8-sig")
    paths.to_csv(PATHS_CSV, index=False, encoding="utf-8-sig")
    save_table_png(table)
    save_irf_graph(irf)
    save_paths_graph(paths)
    for path in [TABLE_CSV, TABLE_TSV, TABLE_MD, TABLE_PNG, IRF_PNG, PATHS_PNG, PATHS_CSV]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
