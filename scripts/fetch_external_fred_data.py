from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_OUT = ROOT / "data" / "raw" / "data_workbook_fred_raw_2010.csv"
PANEL_OUT = ROOT / "data" / "processed" / "data_workbook_bok_panel_2010.csv"
STATUS_OUT = ROOT / "data" / "processed" / "data_workbook_collection_status.csv"
UNRESOLVED_OUT = ROOT / "data" / "processed" / "data_workbook_unresolved_features.csv"


SERIES = [
    {
        "feature": "미국 근원PCE(전년동월비)",
        "variable": "us_core_pce_yoy",
        "fred_id": "PCEPILFE",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCEPILFE",
        "source": "FRED: Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)",
        "transform": "yoy",
    }
]


def fetch_csv(url: str) -> pd.DataFrame:
    with urlopen(url, timeout=30) as response:
        return pd.read_csv(response)


def main() -> int:
    raw_frames = []
    panel_rows = []
    for spec in SERIES:
        df = fetch_csv(spec["url"])
        value_col = spec["fred_id"]
        date_col = "DATE" if "DATE" in df.columns else "observation_date"
        df["DATE"] = pd.to_datetime(df[date_col])
        df = df[df["DATE"] >= "2010-01-01"].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df["value"] = df[value_col].pct_change(12, fill_method=None) * 100
        raw = df[["DATE", value_col]].rename(columns={"DATE": "date", value_col: "raw_value"})
        raw["feature"] = spec["feature"]
        raw["variable"] = spec["variable"]
        raw["source"] = spec["source"]
        raw_frames.append(raw)
        for _, row in df.dropna(subset=["value"]).iterrows():
            panel_rows.append(
                {
                    "date": row["DATE"].strftime("%Y-%m-%d"),
                    "period": row["DATE"].strftime("%Y%m"),
                    "frequency": "M",
                    "feature": spec["feature"],
                    "variable": spec["variable"],
                    "value": row["value"],
                    "source": spec["source"],
                    "stat_code": spec["fred_id"],
                    "item_codes": "",
                    "transform": spec["transform"],
                    "note": "External source, not BOK ECOS",
                }
            )

    if raw_frames:
        pd.concat(raw_frames, ignore_index=True).to_csv(RAW_OUT, index=False, encoding="utf-8-sig")
    panel = pd.read_csv(PANEL_OUT)
    panel = panel[~panel["feature"].isin([row["feature"] for row in panel_rows])]
    panel = pd.concat([panel, pd.DataFrame(panel_rows)], ignore_index=True)
    panel.sort_values(["feature", "date"]).to_csv(PANEL_OUT, index=False, encoding="utf-8-sig")

    status = pd.read_csv(STATUS_OUT)
    for spec in SERIES:
        status = status[status["feature"] != spec["feature"]]
        status = pd.concat(
            [
                status,
                pd.DataFrame(
                    [
                        {
                            "feature": spec["feature"],
                            "status": "EXTERNAL_OK",
                            "variable": spec["variable"],
                            "source": spec["source"],
                            "detail": f"rows={sum(1 for row in panel_rows if row['feature'] == spec['feature'])}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    status.sort_values(["status", "feature"]).to_csv(STATUS_OUT, index=False, encoding="utf-8-sig")
    status[status["status"].isin(["NOT_COLLECTED", "BOK_ERROR", "NO_BOK_ROWS", "CALC_NO_ROWS"])].to_csv(
        UNRESOLVED_OUT, index=False, encoding="utf-8-sig"
    )
    print(RAW_OUT)
    print(PANEL_OUT)
    print(STATUS_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
