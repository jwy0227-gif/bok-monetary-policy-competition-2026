# Raw ECOS data validation

- 기간: 2010-01-01 ~ 2026-02-01
- 행 개수: 194개
- 기대 행 개수: 194개

| 점검 항목 | 결과 | 확인 내용 |
| --- | --- | --- |
| 4개 변수 모두 같은 행 개수인가 | PASS | 각 변수 194개 관측치 |
| 결측치가 없는가 | PASS | {'dubai_oil_usd_bbl': 0, 'exchange_rate_krw_usd': 0, 'cpi': 0, 'expected_inflation_1y': 0} |
| 날짜가 정확히 매칭되는가 | PASS | 2010-01부터 2026-02까지 월별 연속 |
| 유가: $/bbl, 월평균인가 | PASS | 902Y003 / 010102 / 원유- Dubai / 달러/배럴 |
| 환율: 원/달러, 월평균인가 | PASS | 731Y004 / 0000001 / 원/미국달러(매매기준율) / 원 |
| CPI: 지수(2020=100), 동일 기준연도인가 | PASS | 901Y009 / 0 / 총지수 / 2020=100 |
| 기대인플레이션: %, 서베이 기반인가 | PASS | 511Y003 / FMB / 향후1년 기대인플레이션율 / % |
| 그래프로 그려서 이상치 확인했는가 | PASS | reports/figures/raw_timeseries.svg 생성 |

## Source metadata

| variable | stat_code | item_code | item_name | unit |
| --- | --- | --- | --- | --- |
| dubai_oil_usd_bbl | 902Y003 | 010102 | 원유- Dubai | 달러/배럴 |
| exchange_rate_krw_usd | 731Y004 | 0000001 | 원/미국달러(매매기준율) | 원 |
| cpi | 901Y009 | 0 | 총지수 | 2020=100 |
| expected_inflation_1y | 511Y003 | FMB | 향후1년 기대인플레이션율 | % |
