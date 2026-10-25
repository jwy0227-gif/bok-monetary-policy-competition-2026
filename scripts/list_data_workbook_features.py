from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "raw" / "data_workbook_features.json"


def main() -> int:
    data = json.loads(FEATURES.read_text(encoding="utf-8"))
    for sheet in data:
        for feature in sheet["features"]:
            values = feature["values"]
            parts = [
                sheet["sheet"],
                feature["name"],
                feature["unit"] or "",
                f"source={values.get('출처') or ''}",
                f"note={values.get('공표주기·시기 / 비고') or ''}",
            ]
            print("\t".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
