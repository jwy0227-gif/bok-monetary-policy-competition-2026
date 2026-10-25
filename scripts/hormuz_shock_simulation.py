from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_HIST_PATH = ROOT / "data" / "raw" / "ecos_monthly_201001_202602.csv"
RAW_NEW_PATH = ROOT / "data" / "raw" / "ecos_monthly_202603_202604.csv"
IRF_PATH = ROOT / "reports" / "irf_oil_shock_cholesky_var3.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "simulation"

SHOCK_PATH = REPORT_DIR / "hormuz_oil_shock_path.csv"
SIM_PATH = REPORT_DIR / "hormuz_expected_inflation_simulation.csv"
REPORT_PATH = REPORT_DIR / "hormuz_shock_simulation.md"
FIGURE_PATH = FIGURE_DIR / "hormuz_expected_inflation_scenarios.svg"

OIL_SHOCK_SIGMA = 8.045945115677503
KDI_HIGH_OIL_PRICE = 105.0
HORIZON = 24


def build_oil_path() -> pd.DataFrame:
    hist = pd.read_csv(RAW_HIST_PATH, parse_dates=["date"])
    new = pd.read_csv(RAW_NEW_PATH, parse_dates=["date"])

    feb_2026 = float(hist.loc[hist["date"] == pd.Timestamp("2026-02-01"), "dubai_oil_usd_bbl"].iloc[0])
    mar_2026 = float(new.loc[new["date"] == pd.Timestamp("2026-03-01"), "dubai_oil_usd_bbl"].iloc[0])
    apr_2026 = float(new.loc[new["date"] == pd.Timestamp("2026-04-01"), "dubai_oil_usd_bbl"].iloc[0])

    months = pd.date_range("2026-03-01", periods=10, freq="MS")
    oil_prices = [mar_2026, apr_2026] + [KDI_HIGH_OIL_PRICE] * 8
    prev_prices = [feb_2026] + oil_prices[:-1]
    labels = ["실제 ECOS", "실제 ECOS"] + ["KDI 고유가 장기화 전제"] * 8

    rows = []
    for idx, (date, prev, price, label) in enumerate(zip(months, prev_prices, oil_prices, labels, strict=True)):
        dln_oil = 100 * np.log(price / prev)
        rows.append(
            {
                "shock_index": idx,
                "date": date,
                "scenario_source": label,
                "oil_price_usd_bbl": price,
                "previous_oil_price_usd_bbl": prev,
                "dln_oil_pct": dln_oil,
                "shock_size_sigma": dln_oil / OIL_SHOCK_SIGMA,
            }
        )
    return pd.DataFrame(rows)


def simulate_expected_inflation(shock_path: pd.DataFrame) -> pd.DataFrame:
    irf = pd.read_csv(IRF_PATH)
    exp_irf = (
        irf[irf["response"] == "expected_inflation"]
        .sort_values("horizon")
        .set_index("horizon")["irf"]
        .reindex(range(HORIZON + 1))
        .to_numpy()
    )
    exp_lower = (
        irf[irf["response"] == "expected_inflation"]
        .sort_values("horizon")
        .set_index("horizon")["lower_95"]
        .reindex(range(HORIZON + 1))
        .to_numpy()
    )
    exp_upper = (
        irf[irf["response"] == "expected_inflation"]
        .sort_values("horizon")
        .set_index("horizon")["upper_95"]
        .reindex(range(HORIZON + 1))
        .to_numpy()
    )

    march_shock = float(shock_path.loc[shock_path["date"] == pd.Timestamp("2026-03-01"), "shock_size_sigma"].iloc[0])
    shock_multipliers = shock_path["shock_size_sigma"].to_numpy()

    rows = []
    for horizon in range(HORIZON + 1):
        one_off = march_shock * exp_irf[horizon]
        one_off_lower = march_shock * exp_lower[horizon]
        one_off_upper = march_shock * exp_upper[horizon]

        persistent = 0.0
        for shock_idx, multiplier in enumerate(shock_multipliers):
            if horizon - shock_idx >= 0:
                persistent += multiplier * exp_irf[horizon - shock_idx]

        rows.append(
            {
                "horizon": horizon,
                "date": pd.Timestamp("2026-03-01") + pd.DateOffset(months=horizon),
                "one_off_expected_inflation_pp": one_off,
                "one_off_lower_95_pp": min(one_off_lower, one_off_upper),
                "one_off_upper_95_pp": max(one_off_lower, one_off_upper),
                "persistent_expected_inflation_pp": persistent,
            }
        )
    return pd.DataFrame(rows)


def svg_plot(sim: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 520
    left = 82
    right = 32
    top = 64
    bottom = 62
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = pd.concat(
        [
            sim["one_off_expected_inflation_pp"],
            sim["one_off_lower_95_pp"],
            sim["one_off_upper_95_pp"],
            sim["persistent_expected_inflation_pp"],
        ]
    )
    ymin = min(0.0, float(values.min()))
    ymax = max(0.0, float(values.max()))
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
    ymin -= pad
    ymax += pad

    def x_pos(horizon: int) -> float:
        return left + horizon * plot_width / HORIZON

    def y_pos(value: float) -> float:
        return top + plot_height - (value - ymin) * plot_height / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}.zero{stroke:#333;stroke-width:1.2}.one{fill:none;stroke:#0072B2;stroke-width:2.7}.persist{fill:none;stroke:#D55E00;stroke-width:2.7}.band{fill:#0072B2;fill-opacity:.16;stroke:none}</style>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">Hormuz oil shock simulation</text>',
        f'<text x="{left}" y="50" font-size="12" fill="#555">Expected inflation response, one-off shock vs. persistent KDI high-oil scenario</text>',
    ]

    for tick in range(6):
        y = top + tick * plot_height / 5
        value = ymax - tick * (ymax - ymin) / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.2f}</text>')

    zero_y = y_pos(0.0)
    lines.append(f'<line class="zero" x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}"/>')

    upper = [(x_pos(int(row.horizon)), y_pos(float(row.one_off_upper_95_pp))) for row in sim.itertuples()]
    lower = [(x_pos(int(row.horizon)), y_pos(float(row.one_off_lower_95_pp))) for row in sim.iloc[::-1].itertuples()]
    band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
    one = " ".join(
        f"{x_pos(int(row.horizon)):.2f},{y_pos(float(row.one_off_expected_inflation_pp)):.2f}" for row in sim.itertuples()
    )
    persist = " ".join(
        f"{x_pos(int(row.horizon)):.2f},{y_pos(float(row.persistent_expected_inflation_pp)):.2f}" for row in sim.itertuples()
    )
    lines.append(f'<polygon class="band" points="{band}"/>')
    lines.append(f'<polyline class="one" points="{one}"/>')
    lines.append(f'<polyline class="persist" points="{persist}"/>')

    for h in [0, 3, 6, 9, 12, 18, 24]:
        x = x_pos(h)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 28}" font-size="11" text-anchor="middle">T+{h}</text>')

    lines.extend(
        [
            f'<line x1="{left + 15}" y1="{height - 17}" x2="{left + 55}" y2="{height - 17}" stroke="#0072B2" stroke-width="3"/>',
            f'<text x="{left + 63}" y="{height - 13}" font-size="12">1회성 충격</text>',
            f'<line x1="{left + 170}" y1="{height - 17}" x2="{left + 210}" y2="{height - 17}" stroke="#D55E00" stroke-width="3"/>',
            f'<text x="{left + 218}" y="{height - 13}" font-size="12">지속적 충격</text>',
            "</svg>",
        ]
    )
    FIGURE_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_report(shock_path: pd.DataFrame, sim: pd.DataFrame) -> None:
    march = shock_path.iloc[0]
    max_one = sim.loc[sim["one_off_expected_inflation_pp"].idxmax()]
    max_persist = sim.loc[sim["persistent_expected_inflation_pp"].idxmax()]

    lines = [
        "# 호르무즈 충격 시뮬레이션",
        "",
        "## 시나리오 설정",
        "",
        "- 2026년 3월, 4월 유가는 ECOS 실제 월평균 두바이유를 사용했습니다.",
        "- 2026년 5월 이후는 KDI 고유가 장기화 시나리오의 4월 평균 수준, 배럴당 105달러 유지 전제를 사용했습니다.",
        "- KDI 출처: https://www.kdi.re.kr/research/analysisView?art_no=3776",
        "",
        "## 7-1. 충격 크기",
        "",
        f"- IRF의 유가 1표준편차 구조충격: {OIL_SHOCK_SIGMA:.4f}%p",
        f"- 2026년 3월 실제 유가 변동률: {march['dln_oil_pct']:.4f}%",
        f"- 2026년 3월 충격 크기: {march['shock_size_sigma']:.4f}σ",
        "",
        "| 시점 | 유가 전제 | 전월 대비 변동률 | 충격 크기 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in shock_path.iterrows():
        lines.append(
            f"| {row['date'].strftime('%Y-%m')} ({row['scenario_source']}) | "
            f"{row['oil_price_usd_bbl']:.2f} | {row['dln_oil_pct']:.4f}% | {row['shock_size_sigma']:.4f}σ |"
        )

    lines.extend(
        [
            "",
            "## 7-2~7-4. 기대인플레이션 반응",
            "",
            "| 시나리오 | 최대 반응 | 최대 시점 |",
            "| --- | ---: | ---: |",
            f"| 1회성 충격 | {max_one['one_off_expected_inflation_pp']:.4f}%p | T+{int(max_one['horizon'])} |",
            f"| 지속적 충격 | {max_persist['persistent_expected_inflation_pp']:.4f}%p | T+{int(max_persist['horizon'])} |",
            "",
            "## 파일",
            "",
            f"- 유가 충격 경로: `{SHOCK_PATH.relative_to(ROOT).as_posix()}`",
            f"- 기대인플레이션 시뮬레이션: `{SIM_PATH.relative_to(ROOT).as_posix()}`",
            f"- 비교 그래프: `{FIGURE_PATH.relative_to(ROOT).as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shock_path = build_oil_path()
    sim = simulate_expected_inflation(shock_path)
    shock_path.to_csv(SHOCK_PATH, index=False, encoding="utf-8-sig")
    sim.to_csv(SIM_PATH, index=False, encoding="utf-8-sig")
    svg_plot(sim)
    write_report(shock_path, sim)
    print(f"Saved: {SHOCK_PATH}")
    print(f"Saved: {SIM_PATH}")
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
