from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "raw" / "data_workbook_features.json"
STATUS = ROOT / "data" / "processed" / "data_workbook_collection_status.csv"
OUT = ROOT / "data" / "processed" / "data_workbook_unresolved_features_with_sources.csv"


def main() -> int:
    sheets = json.loads(FEATURES.read_text(encoding="utf-8"))
    meta = {}
    for sheet in sheets:
        for feature in sheet["features"]:
            values = feature["values"]
            meta[feature["name"]] = {
                "sheet": sheet["sheet"],
                "unit": feature["unit"],
                "workbook_source": values.get("출처"),
                "workbook_note": values.get("공표주기·시기 / 비고"),
            }
    status = pd.read_csv(STATUS)
    unresolved = status[status["status"].isin(["NOT_COLLECTED", "BOK_ERROR", "NO_BOK_ROWS", "CALC_NO_ROWS"])].copy()
    for column in ["sheet", "unit", "workbook_source", "workbook_note"]:
        unresolved[column] = unresolved["feature"].map(lambda name: meta.get(name, {}).get(column))
    unresolved = unresolved[
        ["sheet", "feature", "unit", "status", "workbook_source", "workbook_note", "detail"]
    ].sort_values(["sheet", "feature"])
    unresolved.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
