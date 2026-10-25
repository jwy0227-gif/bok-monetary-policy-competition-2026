# SVAR oil shock data collection

국제유가가 기대인플레이션에 미치는 영향을 추정하기 위한 ECOS 데이터 수집 작업입니다.

## 필요한 인증 정보

ECOS Open API는 회원 ID가 아니라 별도 인증키가 필요합니다.

PowerShell에서 아래처럼 인증키를 설정한 뒤 실행합니다.

```powershell
$env:ECOS_API_KEY="여기에_ECOS_API_인증키"
python .\scripts\fetch_ecos_data.py
```

## 수집 대상

기간: 2010.01~2026.02

| 변수 | ECOS 계열 | 비고 |
| --- | --- | --- |
| 국제유가 | `902Y003`, 항목 `010102` | 두바이유 월평균, $/bbl |
| 원/달러 환율 | `731Y004`, 항목 `0000001` | 월평균 환율 |
| CPI | `901Y009`, 항목 `0` | 소비자물가지수, 2020=100 |
| 기대인플레이션 | `511Y003`, 항목 `FMB` | 향후1년 기대인플레이션율 |

## 출력

성공하면 아래 파일이 생성됩니다.

- `data/raw/ecos_monthly_201001_202602.csv`
- `data/raw/ecos_metadata_candidates.csv`
