from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "raw" / "data_workbook_features.json"
RAW_OUT = ROOT / "data" / "raw" / "data_workbook_bok_raw_2010.csv"
PANEL_OUT = ROOT / "data" / "processed" / "data_workbook_bok_panel_2010.csv"
STATUS_OUT = ROOT / "data" / "processed" / "data_workbook_collection_status.csv"
UNRESOLVED_OUT = ROOT / "data" / "processed" / "data_workbook_unresolved_features.csv"

BASE_URL = "https://ecos.bok.or.kr/api"
START = {"M": "201001", "Q": "2010Q1", "A": "2010", "D": "20100101"}
END = {"M": "202606", "Q": "2026Q2", "A": "2026", "D": "20260624"}


@dataclass(frozen=True)
class Spec:
    feature: str
    variable: str
    stat_code: str
    cycle: str
    item_codes: tuple[str, ...]
    transform: str = "level"
    scale: float = 1.0
    source_note: str = "BOK ECOS"
    fallback_note: str = ""


SPECS = [
    Spec("한국은행 기준금리", "bok_base_rate", "722Y001", "M", ("0101000",)),
    Spec("미국 연준 정책금리(상단)", "us_policy_rate_upper", "902Y006", "M", ("US",)),
    Spec("콜금리(익일물)", "call_rate_overnight", "721Y001", "M", ("1010000",)),
    Spec("본원통화", "monetary_base_tril_krw", "102Y002", "M", ("ABA1",), scale=1 / 1000),
    Spec("M1(협의통화)", "m1_tril_krw", "161Y002", "M", ("BBLA00",), scale=1 / 1000),
    Spec("M2(광의통화)", "m2_tril_krw", "161Y006", "M", ("BBHA00",), scale=1 / 1000),
    Spec("Lf(금융기관유동성)", "lf_tril_krw", "171Y004", "M", ("LA00000",), scale=1 / 1000),
    Spec("소비자물가(전년동월비)", "cpi_yoy", "901Y009", "M", ("0",), transform="yoy"),
    Spec("생활물가(전년동월비)", "living_cpi_yoy", "901Y010", "M", ("110",), transform="yoy"),
    Spec("근원물가(식료품·에너지제외)", "core_cpi_ex_food_energy_yoy", "901Y010", "M", ("DB",), transform="yoy"),
    Spec("생산자물가(전년동월비)", "ppi_yoy", "404Y014", "M", ("*AA",), transform="yoy"),
    Spec("기대인플레이션(1년)", "expected_inflation_1y", "511Y003", "M", ("FMB",)),
    Spec("기대인플레이션-일반인(3년)", "expected_inflation_3y", "511Y003", "M", ("FMC",)),
    Spec("석유류(전년동월비)", "cpi_petroleum_yoy", "901Y010", "M", ("2125",), transform="yoy"),
    Spec("공업제품(석유류제외,전년동월비)", "cpi_industrial_products_yoy", "901Y010", "M", ("212",), transform="yoy", fallback_note="석유류 제외가 아닌 공업제품 전체 지수"),
    Spec("농축수산물(전년동월비)", "cpi_agri_livestock_fishery_yoy", "901Y010", "M", ("211",), transform="yoy"),
    Spec("가공식품(전년동월비)", "cpi_processed_food_yoy", "901Y010", "M", ("2121",), transform="yoy"),
    Spec("개인서비스(전년동월비)", "cpi_personal_services_yoy", "901Y010", "M", ("223",), transform="yoy"),
    Spec("공공요금(전년동월비)", "cpi_public_services_yoy", "901Y010", "M", ("222",), transform="yoy", fallback_note="공공요금 대신 공공서비스 CPI"),
    Spec("실질 GDP(전기대비)", "real_gdp_qoq", "200Y102", "Q", ("10111",), fallback_note="BOK 주요지표 분기지표"),
    Spec("실질 GDP(전년동기대비)", "real_gdp_yoy", "200Y102", "Q", ("10211",), fallback_note="BOK 주요지표 분기지표"),
    Spec("실질 GNI(전기대비)", "real_gni_qoq", "200Y102", "Q", ("20112",), fallback_note="BOK 주요지표 분기지표"),
    Spec("제조업생산지수(전년동월비)", "manufacturing_production_yoy", "901Y033", "M", ("AB00", "1"), transform="yoy", fallback_note="광공업 생산지수 원계열"),
    Spec("소매판매액지수(전년동월비)", "retail_sales_yoy", "901Y100", "M", ("G0", "T2"), transform="yoy", fallback_note="불변지수"),
    Spec("서비스업생산지수(전년동월비)", "service_production_yoy", "901Y038", "M", ("I51A", "2"), transform="yoy", fallback_note="불변지수"),
    Spec("설비투자지수(전년동월비)", "facility_investment_yoy", "901Y066", "M", ("I15A",), transform="yoy"),
    Spec("건설기성액(전년동월비)", "construction_completed_yoy", "901Y104", "M", ("I48A", "I37B"), transform="yoy", fallback_note="불변금액"),
    Spec("제조업평균가동률", "manufacturing_avg_operation_rate", "901Y025", "M", ("I31A",)),
    Spec("소비자심리지수(CCSI)", "ccsi", "511Y002", "M", ("FME",)),
    Spec("금리수준전망CSI", "interest_rate_outlook_csi", "511Y002", "M", ("FMBG", "99988")),
    Spec("주택가격전망CSI(수도권)", "housing_price_outlook_csi", "511Y002", "M", ("FMFB", "99988"), fallback_note="전국 소비자동향조사 항목"),
    Spec("기업경기실사지수(BSI)", "business_survey_index_all_industry", "512Y007", "M", ("AA", "99988")),
    Spec("경상수지", "current_account_100m_usd", "301Y013", "M", ("000000",), scale=1 / 100),
    Spec("상품수지", "goods_account_100m_usd", "301Y013", "M", ("100000",), scale=1 / 100),
    Spec("서비스수지", "services_account_100m_usd", "301Y013", "M", ("200000",), scale=1 / 100),
    Spec("수출(전년동월비)", "customs_exports_yoy", "901Y118", "M", ("T002",), transform="yoy", fallback_note="통관 수출금액 기준"),
    Spec("수입(전년동월비)", "customs_imports_yoy", "901Y118", "M", ("T004",), transform="yoy", fallback_note="통관 수입금액 기준"),
    Spec("대미 수출(전년동월비)", "exports_to_us_yoy", "901Y121", "M", ("T002", "US"), transform="yoy", fallback_note="국가별 통관 수출금액 기준"),
    Spec("국고채(3년)", "treasury_3y", "721Y001", "M", ("5020000",)),
    Spec("국고채(10년)", "treasury_10y", "721Y001", "M", ("5050000",)),
    Spec("회사채(3년,AA-)", "corp_bond_3y_aa_minus", "721Y001", "M", ("7020000",)),
    Spec("CD(91일)", "cd_91d", "721Y001", "M", ("2010000",)),
    Spec("CP(91일)", "cp_91d", "721Y001", "M", ("4020000",)),
    Spec("코스피", "kospi_close", "901Y014", "M", ("1070000",), fallback_note="월말 종가"),
    Spec("코스닥", "kosdaq_close", "901Y014", "M", ("2090000",), fallback_note="월말 종가"),
    Spec("외국인 주식순매수", "foreign_stock_net_buy_tril_krw", "901Y055", "M", ("S22CC", "VA"), scale=1 / 1_000_000, fallback_note="투자자별 주식거래, 거래대금 순매수"),
    Spec("원/달러 환율(기말)", "krw_usd_eom", "731Y006", "M", ("0000003", "0000200")),
    Spec("원/엔 환율(100엔)", "krw_jpy_100_eom", "731Y006", "M", ("0000006", "0000200")),
    Spec("원/위안 환율", "krw_cny_eom", "731Y006", "M", ("0000010", "0000200")),
    Spec("가계신용 잔액", "household_credit_tril_krw", "151Y001", "Q", ("1000000",), scale=1 / 1000),
    Spec("가계신용(전분기대비 증감)", "household_credit_qoq_change_tril_krw", "151Y001", "Q", ("1000000",), transform="diff", scale=1 / 1000),
    Spec("주택담보대출 잔액", "housing_related_loans_tril_krw", "151Y004", "Q", ("11000A0",), scale=1 / 1000, fallback_note="분기 가계신용 용도별 주택관련대출"),
    Spec("전국 주택매매가격(전월비)", "kb_housing_price_national_mom", "901Y062", "M", ("P63A",), transform="mom", fallback_note="BOK의 KB 주택매매가격지수"),
    Spec("서울 아파트매매가격(전월비)", "kb_apartment_price_seoul_mom", "901Y062", "M", ("P63ACA",), transform="mom", fallback_note="BOK의 KB 주택매매가격지수"),
    Spec("전세가격지수(전월비)", "kb_jeonse_price_national_mom", "901Y063", "M", ("P64A",), transform="mom", fallback_note="BOK의 KB 주택전세가격지수"),
    Spec("외환보유액", "fx_reserves_100m_usd", "732Y001", "M", ("99",), scale=1 / 100000),
    Spec("미국 성장률(전기연율)", "us_gdp_growth", "902Y015", "Q", ("USA",)),
    Spec("일본 성장률(전기대비)", "japan_gdp_growth", "902Y015", "Q", ("JPN",)),
    Spec("중국 성장률(전년동기)", "china_gdp_growth", "902Y015", "Q", ("CHN",)),
    Spec("미국 정책금리(상단)", "global_us_policy_rate", "902Y006", "M", ("US",)),
    Spec("유로 정책금리", "euro_policy_rate", "902Y006", "M", ("XM",)),
    Spec("일본 정책금리", "japan_policy_rate", "902Y006", "M", ("JP",)),
    Spec("두바이유(기말)", "dubai_oil_usd_bbl", "902Y003", "M", ("010102",), fallback_note="ECOS 월별 국제상품가격"),
    Spec("브렌트유(기말)", "brent_oil_usd_bbl", "902Y003", "M", ("010103",), fallback_note="ECOS 월별 국제상품가격"),
    Spec("WTI(기말)", "wti_oil_usd_bbl", "902Y003", "M", ("010101",), fallback_note="ECOS 월별 국제상품가격"),
    Spec("채권거래 합계(거래대금)", "bond_total_trading_value_tril_krw", "901Y120", "M", ("AMT", "020101"), scale=1 / 1_000_000_000_000, fallback_note="BOK 추가수집: 채권시장별 거래"),
    Spec("회사채 거래대금", "corporate_bond_trading_value_tril_krw", "901Y015", "M", ("7", "2040000"), scale=1 / 1_000_000_000_000, fallback_note="BOK 추가수집: 채권종류별 거래"),
    Spec("수출금액지수(전년동월비)", "export_value_index_yoy", "403Y001", "M", ("*AA",), transform="yoy", fallback_note="BOK 추가수집: 무역지수 총지수"),
    Spec("수출물량지수(전년동월비)", "export_volume_index_yoy", "403Y002", "M", ("*AA",), transform="yoy", fallback_note="BOK 추가수집: 무역지수 총지수"),
    Spec("수입금액지수(전년동월비)", "import_value_index_yoy", "403Y003", "M", ("*AA",), transform="yoy", fallback_note="BOK 추가수집: 무역지수 총지수"),
    Spec("수입물량지수(전년동월비)", "import_volume_index_yoy", "403Y004", "M", ("*AA",), transform="yoy", fallback_note="BOK 추가수집: 무역지수 총지수"),
    Spec("_명목GDP", "nominal_gdp_tril_krw", "200Y107", "Q", ("10601",), scale=1 / 1000),
    Spec("_총외채", "external_debt_total_100m_usd", "311Y004", "Q", ("A000000",), scale=1 / 100),
    Spec("_단기외채", "external_debt_short_100m_usd", "311Y004", "Q", ("A500000",), scale=1 / 100),
]


DERIVED = [
    ("내외 금리차(한-미)", "kor_us_policy_rate_gap", "bok_base_rate", "us_policy_rate_upper", "sub"),
    ("M2증가율(전년동월비)", "m2_yoy", "m2_tril_krw", "", "yoy"),
    ("장단기금리차(10년-3년)", "yield_spread_10y_3y", "treasury_10y", "treasury_3y", "sub"),
    ("국고3년-기준금리 스프레드", "treasury_3y_base_rate_spread", "treasury_3y", "bok_base_rate", "sub"),
    ("회사채 스프레드(AA--국고3년)", "corp_bond_aa_minus_spread", "corp_bond_3y_aa_minus", "treasury_3y", "sub"),
    ("명목GDP대비 가계신용 비율", "household_credit_to_nominal_gdp", "household_credit_tril_krw", "nominal_gdp_tril_krw", "ratio_pct"),
    ("단기외채/총외채 비율", "short_external_debt_ratio", "external_debt_short_100m_usd", "external_debt_total_100m_usd", "ratio_pct"),
    ("주담대 월평균 증감", "housing_related_loans_monthly_avg_change", "housing_related_loans_tril_krw", "", "q_diff_div3"),
]


def get_key() -> str:
    key = os.environ.get("ECOS_API_KEY") or os.environ.get("BOK_API_KEY")
    if not key:
        raise SystemExit("Set ECOS_API_KEY or BOK_API_KEY.")
    return key


def ecos_url(service: str, key: str, start_no: int, end_no: int, *parts: str) -> str:
    escaped = "/".join(quote(str(part), safe="") for part in parts if part is not None)
    return f"{BASE_URL}/{service}/{key}/json/kr/{start_no}/{end_no}/{escaped}"


def fetch_rows(key: str, spec: Spec) -> list[dict[str, Any]]:
    start_no = 1
    rows: list[dict[str, Any]] = []
    while True:
        url = ecos_url(
            "StatisticSearch",
            key,
            start_no,
            start_no + 999,
            spec.stat_code,
            spec.cycle,
            START[spec.cycle],
            END[spec.cycle],
            *spec.item_codes,
        )
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "RESULT" in payload:
            result = payload["RESULT"]
            code = result.get("CODE")
            if code == "INFO-200":
                return rows
            raise RuntimeError(f"{code}: {result.get('MESSAGE')}")
        body = payload.get("StatisticSearch") or {}
        page = body.get("row") or []
        if isinstance(page, dict):
            page = [page]
        rows.extend(page)
        total = int(body.get("list_total_count", len(rows)))
        if start_no + 999 >= total or not page:
            return rows
        start_no += 1000
        time.sleep(0.05)


def period_to_date(time_value: str, cycle: str) -> str:
    if cycle == "M":
        return f"{time_value[:4]}-{time_value[4:6]}-01"
    if cycle == "Q":
        year = int(time_value[:4])
        quarter = int(time_value[-1])
        month = 3 * (quarter - 1) + 1
        return f"{year:04d}-{month:02d}-01"
    if cycle == "D":
        return f"{time_value[:4]}-{time_value[4:6]}-{time_value[6:8]}"
    return f"{time_value}-01-01"


def transform_series(values: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return values
    if transform == "yoy":
        return values.pct_change(12, fill_method=None) * 100
    if transform == "mom":
        return values.pct_change(1, fill_method=None) * 100
    if transform == "qoq":
        return values.pct_change(1, fill_method=None) * 100
    if transform == "q_yoy":
        return values.pct_change(4, fill_method=None) * 100
    if transform == "diff":
        return values.diff()
    raise ValueError(transform)


def load_feature_set() -> set[str]:
    sheets = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return {feature["name"] for sheet in sheets for feature in sheet["features"] if not feature["name"].startswith("※")}


def main() -> int:
    key = get_key()
    raw_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for index, spec in enumerate(SPECS, start=1):
        print(f"[{index}/{len(SPECS)}] {spec.feature}")
        try:
            rows = fetch_rows(key, spec)
        except Exception as exc:
            status_rows.append(
                {
                    "feature": spec.feature,
                    "status": "BOK_ERROR",
                    "variable": spec.variable,
                    "source": spec.source_note,
                    "detail": str(exc),
                }
            )
            continue
        for row in rows:
            raw_rows.append(
                {
                    "feature": spec.feature,
                    "variable": spec.variable,
                    "date": period_to_date(row["TIME"], spec.cycle),
                    "period": row["TIME"],
                    "frequency": spec.cycle,
                    "raw_value": row.get("DATA_VALUE"),
                    "stat_code": row.get("STAT_CODE"),
                    "stat_name": row.get("STAT_NAME"),
                    "item_code1": row.get("ITEM_CODE1"),
                    "item_name1": row.get("ITEM_NAME1"),
                    "item_code2": row.get("ITEM_CODE2"),
                    "item_name2": row.get("ITEM_NAME2"),
                    "unit": row.get("UNIT_NAME"),
                    "source_note": spec.source_note,
                    "fallback_note": spec.fallback_note,
                }
            )
        if not rows:
            status_rows.append(
                {
                    "feature": spec.feature,
                    "status": "NO_BOK_ROWS",
                    "variable": spec.variable,
                    "source": spec.source_note,
                    "detail": spec.fallback_note,
                }
            )
            continue
        frame = pd.DataFrame(
            {
                "date": [period_to_date(row["TIME"], spec.cycle) for row in rows],
                "period": [row["TIME"] for row in rows],
                "frequency": spec.cycle,
                "value": [float(row["DATA_VALUE"]) * spec.scale for row in rows],
            }
        ).sort_values("date")
        frame["value"] = transform_series(frame["value"], spec.transform)
        frame = frame.dropna(subset=["value"])
        for _, row in frame.iterrows():
            panel_rows.append(
                {
                    "date": row["date"],
                    "period": row["period"],
                    "frequency": row["frequency"],
                    "feature": spec.feature,
                    "variable": spec.variable,
                    "value": row["value"],
                    "source": spec.source_note,
                    "stat_code": spec.stat_code,
                    "item_codes": "/".join(spec.item_codes),
                    "transform": spec.transform,
                    "note": spec.fallback_note,
                }
            )
        status_rows.append(
            {
                "feature": spec.feature,
                "status": "BOK_OK",
                "variable": spec.variable,
                "source": spec.source_note,
                "detail": f"rows={len(frame)}, stat={spec.stat_code}, items={'/'.join(spec.item_codes)}; {spec.fallback_note}",
            }
        )
        time.sleep(0.08)

    panel = pd.DataFrame(panel_rows)
    if not panel.empty:
        wide = panel.pivot_table(index=["date", "period", "frequency"], columns="variable", values="value", aggfunc="last").reset_index()
        wide = wide.sort_values(["date", "frequency"])
        for feature, variable, left, right, op in DERIVED:
            values = []
            if op == "sub" and left in wide and right in wide:
                series = wide[left] - wide[right]
            elif op == "yoy" and left in wide:
                series = wide[left].pct_change(12, fill_method=None) * 100
            elif op == "ratio_pct" and left in wide and right in wide:
                series = wide[left] / wide[right] * 100
            elif op == "q_diff_div3" and left in wide:
                series = wide[left].dropna().diff() / 3
            else:
                continue
            for idx, value in series.items():
                if pd.isna(value):
                    continue
                row = wide.loc[idx]
                values.append(
                    {
                        "date": row["date"],
                        "period": row["period"],
                        "frequency": row["frequency"],
                        "feature": feature,
                        "variable": variable,
                        "value": value,
                        "source": "calculated from BOK ECOS",
                        "stat_code": "",
                        "item_codes": "",
                        "transform": op,
                        "note": f"{left}, {right}".rstrip(", "),
                    }
                )
            panel_rows.extend(values)
            status_rows.append(
                {
                    "feature": feature,
                    "status": "CALCULATED_OK" if values else "CALC_NO_ROWS",
                    "variable": variable,
                    "source": "calculated from BOK ECOS",
                    "detail": f"rows={len(values)}",
                }
            )

    feature_set = load_feature_set()
    collected = {row["feature"] for row in status_rows if row["status"] in {"BOK_OK", "CALCULATED_OK"}}
    for feature in sorted(feature_set - collected):
        status_rows.append(
            {
                "feature": feature,
                "status": "NOT_COLLECTED",
                "variable": "",
                "source": "",
                "detail": "BOK 직접 매핑/계산 스펙 없음. 외부 출처 확인 필요.",
            }
        )

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(RAW_OUT, index=False, encoding="utf-8-sig")
    panel_final = pd.DataFrame(panel_rows)
    if not panel_final.empty:
        panel_final = panel_final[~panel_final["feature"].astype(str).str.startswith("_")]
    panel_final.sort_values(["feature", "date"]).to_csv(PANEL_OUT, index=False, encoding="utf-8-sig")
    status = pd.DataFrame(status_rows).sort_values(["status", "feature"])
    status.to_csv(STATUS_OUT, index=False, encoding="utf-8-sig")
    status[status["status"].isin(["NOT_COLLECTED", "BOK_ERROR", "NO_BOK_ROWS", "CALC_NO_ROWS"])].to_csv(
        UNRESOLVED_OUT, index=False, encoding="utf-8-sig"
    )
    print(RAW_OUT)
    print(PANEL_OUT)
    print(STATUS_OUT)
    print(UNRESOLVED_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
