# PPT 그래프·표 생성 결과

## 생성한 파일

- `inflation_cpi_core_living_2025_07_2026_06.png`
- `inflation_expectations_2025_01_2026_06.png`
- `industrial_ppi_cpi_pressure_2026_01_06.png`
- `monthly_oil_prices_2025_01_2026_06.png`
- `domestic_demand_ccsi_retail_2025_01_2026_06.png`
- `real_activity_investment_construction_2025_01_2026_06.png`
- `external_trade_current_account_2025_01_2026_06.png`
- `financial_market_rates_2025_01_2026_06.png`
- `housing_market_prices_csi_2025_01_2026_06.png`
- `fx_reserves_external_debt_2025_01_2026_06.png`
- `global_policy_rates_2025_01_2026_06.png`
- `latest_key_indicators_table.csv`

## 추가 데이터가 필요한 항목

- **세계경제 성장률/전망 비교**: OECD Data Explorer, OECD Economic Outlook 2026.6, BOK/KDI 경제전망
- **AI 하이퍼스케일러 설비투자**: BIS Annual Economic Report 2026 Graph 11.A, 각사 IR/10-K
- **일별 국제유가·전쟁 이벤트**: Petronet, EIA, IEA OMR. 현재 수집본은 월별 유가만 보유
- **지정학적 위험지수 GPR**: Caldara-Iacoviello GPR: matteoiacoviello.com/gpr.htm
- **OECD 석유재고 커버리지**: EIA STEO, IEA Oil Market Report
- **주요국 CPI 전체 비교**: BLS, Eurostat, 일본 총무성, 국가데이터처
- **FOMC 점도표**: Federal Reserve SEP 원문
- **수출 품목분해: 반도체·자동차·선박 등**: 관세청/무역협회 KITA 품목별 수출입, 산업통상자원부 수출입동향
- **반도체 가격/DRAM spot**: TrendForce, DRAMeXchange 등 유료/부분공개 데이터
- **고용 산업별·연령별 증감**: KOSIS 경제활동인구조사, 고용행정통계
- **건설수주·착공·허가·건설업 BSI**: KOSIS 건설경기동향, 한국은행 BSI 세부, 국토교통부
- **금융불안지수 FSI/FVI**: 한국은행 금융안정 상황 대시보드/금융안정보고서
- **기업대출·연체율·취약차주**: 금융감독원 금융통계정보시스템, BOK 금융안정 DB
- **VKOSPI**: KRX, Investing.com, Stockplus, Bloomberg
- **BIS 명목실효환율**: BIS effective exchange rate statistics
- **주택매매거래량·수도권 전월세 세부**: 한국부동산원 R-ONE, 국토교통부 실거래가
- **IT 특별급여/성과급 CPI 파급**: 한국은행 이슈노트/경제전망보조자료 원문 도표

## 주의

- `cpi_industrial_products_yoy`는 현재 수집본 기준 ECOS 공업제품 전체 지수입니다. PPT 문구의 '석유류제외 공업제품'과 완전히 같지 않으므로, 보도자료 기여도 표 또는 ECOS 세부 품목으로 별도 계산하는 것이 좋습니다.
- 분기 지표는 해당 분기 시작월에 배치된 값입니다.
