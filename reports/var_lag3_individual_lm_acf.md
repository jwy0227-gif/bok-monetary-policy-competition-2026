# VAR(3) 잔차 개별 lag LM 및 ACF/PACF

- 사용 데이터: `data/processed/svar_main_level_201002_202602.csv`
- VAR 시차: 3
- 잔차 관측치: 190개

## 개별 lag별 LM 검정

각 검정은 원래 VAR(3)의 설명변수들을 통제한 뒤, 특정 lag의 잔차항이 현재 잔차를 추가로 설명하는지 보는 시스템 LM 검정입니다. 귀무가설은 '해당 lag의 잔차 자기상관이 없다'입니다.

| lag | LM 통계량 | df | p-value | 판정 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 22.700 | 16 | 0.1220 | 자기상관 뚜렷하지 않음 |
| 2 | 24.227 | 16 | 0.0847 | 자기상관 뚜렷하지 않음 |
| 3 | 21.376 | 16 | 0.1645 | 자기상관 뚜렷하지 않음 |
| 4 | 27.703 | 16 | 0.0343 | 자기상관 있음 |
| 5 | 10.823 | 16 | 0.8203 | 자기상관 뚜렷하지 않음 |
| 6 | 37.186 | 16 | 0.0020 | 자기상관 있음 |

## ACF/PACF 그래프

- ACF: `reports/figures/residual_diagnostics/var_lag3_residual_acf.svg`
- PACF: `reports/figures/residual_diagnostics/var_lag3_residual_pacf.svg`

주황색 막대는 대략적 95% 경계 밖에 있는 lag입니다.

## lag 1~6에서 경계 밖 ACF/PACF

| 변수 | lag | ACF | PACF | 경계 밖 |
| --- | ---: | ---: | ---: | --- |
| CPI | 4 | 0.154 | 0.154 | ACF, PACF |

## 결론

개별 lag 1~6 LM 검정에서는 lag 4, 6에서 5% 기준 잔차 자기상관이 확인됩니다.
