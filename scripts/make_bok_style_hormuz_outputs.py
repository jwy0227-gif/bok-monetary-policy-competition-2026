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

TABLE_CSV = FIGURE_DIR / "hormuz_scenario_table.csv"
TABLE_TSV = FIGURE_DIR / "hormuz_scenario_table_for_docs.tsv"
TABLE_MD = FIGURE_DIR / "hormuz_scenario_table_for_docs.md"
TABLE_PNG = FIGURE_DIR / "hormuz_scenario_table.png"
IRF_PNG = FIGURE_DIR / "irf_1sd_expected_inflation.png"
PATHS_PNG = FIGURE_DIR / "hormuz_scenario_paths.png"
PATHS_CSV = FIGURE_DIR / "hormuz_scenario_paths.csv"

BASELINE_EXPECTED_INFLATION = 2.6
ANCHORING_THRESHOLD = 3.0
HORIZON = 24

SCENARIOS = [
    {
        "scenario": "조기 종전",
        "max_response": 0.2747,
        "max_level": 2.8747,
        "threshold": "도달 안 함",
        "interpretation": "기대인플레이션 상승 압력은 존재하나 3%에는 미달",
        "color": "#1f4e79",
    },
    {
        "scenario": "봉쇄 장기화",
        "max_response": 0.3070,
        "max_level": 2.9070,
        "threshold": "도달 안 함",
        "interpretation": "기대인플레이션 불안정 위험 확대",
        "color": "#8064a2",
    },
    {
        "scenario": "에너지 시설 타격",
        "max_response": 0.4568,
        "max_level": 3.0568,
        "threshold": "T+4",
        "interpretation": "3% 상회, 기대인플레이션 불안정 위험 현실화",
        "color": "#c0504d",
    },
]

NOTE = "주: 각 시나리오는 1표준편차 유가 충격에 대한 SVAR 충격반응함수를 충격 규모에 비례해 조정하여 산출. 자료: 저자 추정."
DPI_STANDARD = 350
DPI_HIGH_RES = 600
REPORT_WIDTH_CM = [13, 15, 17]


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


def load_baseline_irf() -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp = irf[irf["response"] == "expected_inflation"].sort_values("horizon").copy()
    exp = exp[(exp["horizon"] >= 0) & (exp["horizon"] <= HORIZON)].copy()
    return exp[["horizon", "irf"]].reset_index(drop=True)


def build_table(irf_max_for_scaling: float = 0.06) -> pd.DataFrame:
    rows = []
    for spec in SCENARIOS:
        rows.append(
            {
                "시나리오": spec["scenario"],
                "충격 규모": f"{spec['max_response'] / irf_max_for_scaling:.2f}배",
                "최대 반응(%p)": spec["max_response"],
                "최대 반응 시점": "T+4",
                "최대 기대인플레이션 수준(%)": spec["max_level"],
                "임계선 최초 도달": spec["threshold"],
                "해석": spec["interpretation"],
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_table_png(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15.0, 4.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    display_table = table.copy()
    display_table["최대 반응(%p)"] = display_table["최대 반응(%p)"].map(lambda x: f"{x:.4f}")
    display_table["최대 기대인플레이션 수준(%)"] = display_table["최대 기대인플레이션 수준(%)"].map(lambda x: f"{x:.4f}")

    ax.set_title("표 1. 호르무즈 충격 시나리오별 기대인플레이션 반응", loc="left", fontsize=14, weight="bold", pad=12)
    mpl_table = ax.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0, 0.30, 1, 0.58],
        colWidths=[0.11, 0.08, 0.10, 0.10, 0.15, 0.12, 0.34],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9.2)
    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#6f6f6f")
        if row == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f7f7f7")
            if col == 6:
                cell.set_text_props(ha="left")

    ax.text(0, 0.12, NOTE, transform=ax.transAxes, fontsize=8.6, color="#555555", ha="left", va="bottom")
    fig.savefig(TABLE_PNG, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_irf_graph(
    irf: pd.DataFrame,
    output_path: Path = IRF_PNG,
    figsize: tuple[float, float] = (10.6, 6.2),
    dpi: int = DPI_STANDARD,
) -> None:
    h = irf["horizon"].to_numpy()
    y = irf["irf"].to_numpy()
    max_idx = int(np.argmax(y))
    max_h = int(h[max_idx])
    max_y = float(y[max_idx])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhspan(0, max(y) * 1.18, xmin=1 / HORIZON, xmax=20 / HORIZON, color="#d9e6f2", alpha=0.35, label="유의한 반응 기간(T+1~T+20)")
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
    ax.set_ylim(min(-0.015, float(y.min()) - 0.01), max(float(y.max()) * 1.42, 0.09))
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["T", "T+4", "T+8", "T+12", "T+16", "T+20", "T+24"])
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    ax.text(
        0,
        -0.30,
        "주: 신뢰구간은 별도 추정 결과 이용 시 추가. " + NOTE,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
        ha="left",
        va="top",
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_scenario_paths(irf: pd.DataFrame) -> pd.DataFrame:
    base_response = irf["irf"].to_numpy(dtype=float)
    base_max = float(base_response.max())
    rows = []
    for spec in SCENARIOS:
        scale = spec["max_response"] / base_max
        response = base_response * scale
        level = BASELINE_EXPECTED_INFLATION + response
        # Align exactly with the requested headline numbers at the maximum.
        max_idx = int(np.argmax(level))
        level[max_idx] = spec["max_level"]
        response[max_idx] = spec["max_level"] - BASELINE_EXPECTED_INFLATION
        for h, resp, lev in zip(range(HORIZON + 1), response, level, strict=True):
            rows.append(
                {
                    "시나리오": spec["scenario"],
                    "horizon": h,
                    "response_pp": resp,
                    "expected_inflation_level_pct": lev,
                }
            )
    return pd.DataFrame(rows)


def save_scenario_graph(
    paths: pd.DataFrame,
    output_path: Path = PATHS_PNG,
    figsize: tuple[float, float] = (10.8, 6.3),
    dpi: int = DPI_STANDARD,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spec in SCENARIOS:
        sub = paths[paths["시나리오"] == spec["scenario"]].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["expected_inflation_level_pct"],
            linewidth=2.2,
            color=spec["color"],
            label=spec["scenario"],
        )

    ax.axhline(ANCHORING_THRESHOLD, color="#555555", linestyle=(0, (4, 3)), linewidth=1.1)
    ax.text(24.25, ANCHORING_THRESHOLD + 0.006, "3.0% 참고선", fontsize=9.7, color="#444444", va="bottom")

    hit = paths[(paths["시나리오"] == "에너지 시설 타격") & (paths["horizon"] == 4)].iloc[0]
    ax.scatter([4], [hit["expected_inflation_level_pct"]], color="#c0504d", s=46, zorder=5)
    ax.annotate(
        "3% 상회",
        xy=(4, hit["expected_inflation_level_pct"]),
        xytext=(6.4, hit["expected_inflation_level_pct"] + 0.075),
        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.75, "shrinkA": 2, "shrinkB": 5},
        fontsize=9.5,
    )

    ax.set_title("그림 2. 호르무즈 시나리오별 기대인플레이션 경로", loc="left", fontsize=14, weight="bold", pad=11)
    ax.set_xlabel("충격 후 경과 개월")
    ax.set_ylabel("기대인플레이션 수준(%)")
    ax.set_ylim(2.48, 3.22)
    ax.set_xlim(0, 25.4)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["T", "T+4", "T+8", "T+12", "T+16", "T+20", "T+24"])
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.54), borderaxespad=0)
    ax.text(0, -0.25, NOTE, transform=ax.transAxes, fontsize=8.5, color="#555555", ha="left", va="top")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_report_width_variants(irf: pd.DataFrame, paths: pd.DataFrame) -> list[Path]:
    created: list[Path] = []
    for width_cm in REPORT_WIDTH_CM:
        width_in = cm_to_inch(width_cm)
        irf_height_in = width_in * 0.58
        paths_height_in = width_in * 0.58

        irf_out = FIGURE_DIR / f"irf_1sd_expected_inflation_{width_cm}cm_600dpi.png"
        paths_out = FIGURE_DIR / f"hormuz_scenario_paths_{width_cm}cm_600dpi.png"

        save_irf_graph(irf, output_path=irf_out, figsize=(width_in, irf_height_in), dpi=DPI_HIGH_RES)
        save_scenario_graph(paths, output_path=paths_out, figsize=(width_in, paths_height_in), dpi=DPI_HIGH_RES)

        created.extend([irf_out, paths_out])
    return created


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()

    irf = load_baseline_irf()
    table = build_table(irf_max_for_scaling=0.06)
    paths = build_scenario_paths(irf)

    table.to_csv(TABLE_CSV, index=False, encoding="utf-8-sig")
    table.to_csv(TABLE_TSV, index=False, sep="\t", encoding="utf-8-sig")
    TABLE_MD.write_text(dataframe_to_markdown(table) + "\n", encoding="utf-8-sig")
    paths.to_csv(PATHS_CSV, index=False, encoding="utf-8-sig")
    save_table_png(table)
    save_irf_graph(irf)
    save_scenario_graph(paths)
    width_variant_paths = save_report_width_variants(irf, paths)

    for path in [TABLE_CSV, TABLE_TSV, TABLE_MD, TABLE_PNG, IRF_PNG, PATHS_PNG, PATHS_CSV, *width_variant_paths]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
