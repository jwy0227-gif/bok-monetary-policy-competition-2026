# 촐레스키 IRF: 유가 충격

- 사용 데이터: `data/processed/svar_main_level_stl_cpi_201002_202602.csv`
- 모형: VAR(3)
- 변수 순서: 국제유가 → 환율 → CPI → 기대인플레이션
- CPI: STL 계절조정 CPI의 월간 로그차분
- 유가 1표준편차 충격 크기: 8.0459%p, 약 8.05% 월간 상승
- IRF 기간: 0~24개월
- 신뢰구간: asymptotic standard error 기반 95%

## 핵심 결과 요약

| 충격 → 반응 | 최대 반응치 | 최대 반응 시점 | 95% 유의 기간 |
| --- | ---: | ---: | --- |
| 유가 → 기대인플레이션 | 0.0602%p | T+4개월 | T+1~T+20 |
| 유가 → CPI | 0.1219%p | T+1개월 | T+1 |
| 유가 → 환율 | 0.1267%p | T+3개월 | 없음 |

## 파일

- IRF 전체 표: `reports/irf_oil_shock_cholesky_var3.csv`
- 요약표: `reports/irf_oil_shock_summary_var3.csv`
- 촐레스키 행렬: `reports/cholesky_matrix_var3.csv`
- IRF 그래프: `reports/figures/irf/oil_shock_irf_var3_stl_cpi.svg`

## 해석 메모

IRF는 유가 방정식 잔차의 1표준편차 구조충격을 촐레스키 분해로 식별한 결과입니다. 기대인플레이션 반응은 수준 변수이므로 단위는 %p로 해석합니다.
