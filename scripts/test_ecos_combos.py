from __future__ import annotations

import json
import os
from urllib.parse import quote
from urllib.request import urlopen


BASE_URL = "https://ecos.bok.or.kr/api"


def url(service: str, key: str, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts)
    return f"{BASE_URL}/{service}/{key}/json/kr/1/5/{escaped}"


def test(stat_code: str, cycle: str, start: str, end: str, combos: list[tuple[str, ...]]) -> None:
    key = os.environ["BOK_API_KEY"]
    for combo in combos:
        target = url("StatisticSearch", key, stat_code, cycle, start, end, *combo)
        with urlopen(target, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "RESULT" in payload:
            result = payload["RESULT"]
            print(combo, result.get("CODE"), result.get("MESSAGE"))
        else:
            rows = payload.get("StatisticSearch", {}).get("row", [])
            if isinstance(rows, dict):
                rows = [rows]
            print(combo, len(rows), rows[:2])


def main() -> int:
    test(
        "901Y033",
        "M",
        "201001",
        "201003",
        [("AB00",), ("1",), ("AB00", "1"), ("1", "AB00")],
    )
    test(
        "731Y006",
        "M",
        "201001",
        "201003",
        [("0000003",), ("0000200", "0000003"), ("0000003", "0000200")],
    )
    test(
        "512Y007",
        "M",
        "201001",
        "201003",
        [("AA",), ("AA", "99988"), ("99988", "AA")],
    )
    test(
        "901Y121",
        "M",
        "201001",
        "201003",
        [("T002",), ("T002", "US"), ("US", "T002")],
    )
    test(
        "901Y120",
        "M",
        "201001",
        "201003",
        [("AMT",), ("AMT", "020101"), ("020101", "AMT")],
    )
    test(
        "901Y015",
        "M",
        "201001",
        "201003",
        [("7",), ("7", "2040000"), ("2040000", "7")],
    )
    test(
        "901Y055",
        "M",
        "201001",
        "201003",
        [("S22CC",), ("S22CC", "VO"), ("VO", "S22CC")],
    )
    test(
        "403Y001",
        "M",
        "201001",
        "201003",
        [("*AA",), ("AA",), ("T000",), ("I00",)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
