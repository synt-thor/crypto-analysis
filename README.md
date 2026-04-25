# crypto-analysis

Deribit BTC 선물 **다팩터 신호 엔진 + 백테스트** 프로젝트.
단기 타임프레임(수 시간~수일) 기준으로 "지금 롱/숏/중립 중 무엇인가 + 신뢰도"를 답합니다.

> ⚠️ 확률적 보조 신호입니다. 시장 방향을 확정 예측하지 않으며, 과거 성과가 미래를 보장하지 않습니다.

## 어떻게 동작하나

11개 신호가 각각 `score ∈ [-1(강숏), +1(강롱)]` + `confidence ∈ [0,1]`을 내고,
`engine.fuse()`가 `base_weight × confidence`로 재정규화 후 가중합하여 최종 점수를 만들고,
`decision.decide()`가 임계치(±0.15)와 최소 신뢰도(0.25)로 **LONG / SHORT / NEUTRAL** 결론을 냅니다.

| 신호 | 파일 | 기본 가중치 | 요지 |
|---|---|---|---|
| funding | `signals/funding.py` | 18% | 퍼펫추얼 펀딩 극단치 → 역추세 |
| spot_futures | `signals/spot_futures.py` | 13% | Deribit 퍼프 마크 vs Binance 스팟 프리미엄 |
| option_skew | `signals/option_skew.py` | 12% | 25Δ 리스크리버설 (RR=콜25IV−풋25IV) |
| basis | `signals/basis.py` | 10% | 만기 선물 연율화 베이시스 (콘탱고/백워데이션) |
| oi | `signals/oi.py` | 10% | OI·가격 결합 (신규 포지션 구축 방향) |
| macro | `signals/macro.py` | 9% | DXY/SPY/QQQ/VIX/10Y/Gold 모멘텀 → BTC 상관 |
| iv_skew | `signals/iv_skew.py` | 8% | ATM IV − 실현 변동성 (공포/안도) |
| gex | `signals/gex.py` | 7% | 딜러 감마 편향 프록시 (콜/풋 γ-OI 비중) |
| orderbook | `signals/orderbook.py` | 5% | 퍼펫추얼 상위 10호가 뎁스 임밸런스 |
| onchain | `signals/onchain.py` | 4% | 멤풀·수수료·해시레이트 (단기 비중 작음) |
| news | `signals/news.py` | 4% | 구조화된 뉴스 브리프 → 롱/숏 바이어스 |

옵션계 신호 3종(option_skew, gex, iv_skew)은 `greeks.py`의 Black-Scholes
delta/gamma를 사용해 Deribit 옵션 체인에서 계산합니다 (r=0 근사).

## 구조

```
src/crypto_analysis/
├── collectors/           # Deribit, mempool.space, Binance/Coinbase, yfinance
├── signals/              # 개별 신호 모듈 (위 표)
├── indicators.py         # zscore, squash, 연율화 basis/펀딩 공통 계산
├── storage.py            # Raw JSON + Parquet + DuckDB view 헬퍼
├── engine.py             # 가중치 융합
├── decision.py           # 임계치 → LONG/SHORT/NEUTRAL
└── backtest.py           # 과거 데이터 기반 적중률/에쿼티 커브
scripts/
├── snapshot_now.py       # 현재 시장 한 세트 저장
├── backfill_history.py   # OHLCV·펀딩·macro 백필
└── decide_now.py         # 지금 바로 의사결정 리포트 출력
notebooks/
├── decision_now.ipynb    # 실시간 의사결정 + macro brief + 가중치 오버라이드
└── backtest_results.ipynb# 백테스트 결과 요약·에쿼티 커브
```

## 시작하기

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# A) 웹 UI (가장 쉬움)
streamlit run streamlit_app.py
# → 브라우저 자동 오픈, http://localhost:8501

# B) CLI로 한 번 판단
python scripts/decide_now.py --news-json data/news_brief.json

# C) 90일 백필 → 노트북 백테스트
python scripts/backfill_history.py --days 90
jupyter lab notebooks/backtest_results.ipynb
```

## 웹 UI

`streamlit_app.py`가 기존 신호 엔진을 그대로 호출하는 단일 파일 Streamlit 앱입니다.

**제공 기능**
- 종합 판정 (LONG/SHORT/NEUTRAL) + score + confidence를 큰 카드로 표시
- ST(수 시간)/MT(수 일)/LT(수 주) 타임프레임별 판정 카드
- 12개 신호 기여 표 (score progress bar + 근거)
- 사이드바 가중치 슬라이더 + ST/MT/LT 프리셋 버튼
- 뉴스 브리프 JSON 인라인 편집 (디스크 로드/적용)
- 라이브 API에서 즉시 다운로드해 실행하는 백테스트 (7~90일, hold 4/8/24h)

**캐싱**
- 마켓 인풋 60초, 매크로 패널 5분, 백테스트 1시간
- 우상단 🔄 Refresh 버튼으로 즉시 클리어

### 원격 배포 (Streamlit Community Cloud)

1. GitHub 저장소 생성 후 프로젝트 push (`.gitignore`가 `data/parquet/`, `.venv/` 막는지 확인)
2. <https://share.streamlit.io> 접속, GitHub 연결
3. "New app" → repo 선택, branch `main`, main file `streamlit_app.py`
4. Advanced settings → Python 3.12
5. Deploy → 1~2분 후 `https://<sub>.streamlit.app` URL 발급

퍼블릭 GitHub 저장소만 무료. 프라이빗이 필요하면 Streamlit for Teams 또는 Fly.io.

## 실시간 뉴스 자동 반영 (Gemini)

키를 설정하면 앱이 페이지 로드 시 RSS(CoinDesk, CoinTelegraph, Federal Reserve, BBC World/Politics/Business)를 자동 페치하고 Gemini 2.0 Flash로 점수화해 `news` 신호에 즉시 반영합니다. 트럼프 외교·관세·제재·연준 결정 등 시시각각 변하는 이슈를 사람 개입 없이 단기 의사결정에 즉시 반영하기 위함.

**무료 — Gemini 2.0 Flash 무료 티어 (15 RPM, 일 1500회) + 모든 RSS 무료**

### 1) Gemini API 키 발급 (1분)

<https://aistudio.google.com/app/apikey> 접속 → Google 로그인 → "Create API key" 클릭 → 발급된 키 복사

### 2) Streamlit Cloud Secrets에 등록

배포된 앱에서:
1. 우하단 **Manage app** 클릭
2. **Settings** → **Secrets**
3. 다음 한 줄 입력 후 저장:

```toml
GEMINI_API_KEY = "AIzaSy...여기에붙여넣기"
```

다음 페이지 로드부터 자동으로 LIVE 모드. 앱 재시작 불필요.

### 3) 동작 확인

페이지 상단에 다음 배지가 뜨면 정상:

- 🔴 **LIVE · N events · M분 전 갱신** ← 키 등록 완료, 자동 페치 동작
- ⚠️ **LIVE 페치 실패** ← 키는 있는데 Gemini API 호출 실패 (한도 초과 / 일시 장애)
- 📁 **Disk baseline (Gemini API 키 미설정)** ← 키 없음, 정적 `data/news_brief.json` 사용

### 4) 수동 오버라이드

LIVE 모드에서도 expander 안의 토글 "수동 편집 사용"으로 사이드 by 사이드 비교·강제 교체 가능. 토글 끄면 다시 LIVE.

### 캐시 동작

- LIVE 브리프는 **15분 캐시**. 트레이더 사용량 기준 분당 호출 부하는 제로에 가까움
- 헤더의 🔄 Refresh 버튼이 캐시 클리어 후 강제 재페치 (Gemini API 한 번 더 사용됨)

### 로컬 테스트

```bash
export GEMINI_API_KEY="AIzaSy..."
.venv/bin/python -c "
import os
from crypto_analysis.news_fetcher import build_news_brief
b = build_news_brief(api_key=os.getenv('GEMINI_API_KEY'))
print(f'events: {len(b[\"events\"])}')
for e in b['events'][:5]:
    print(f'  {e[\"bias\"]:7} w={e[\"weight\"]} :: {e[\"headline\"][:80]}')
"
```

## 뉴스·국제정세 레이어 사용법

`signals/news.py`는 구조화된 브리프를 받습니다(의도적으로 자동 스크레이핑 없음 —
뉴스→숫자 변환은 노이즈가 크므로 사람 또는 에이전트가 한 번 검토):

```json
{
  "window_hours": 24,
  "events": [
    {"headline": "Fed signals pause", "bias": "long", "weight": 0.7, "rationale": "liquidity easing"},
    {"headline": "Spot ETF outflow $180M", "bias": "short", "weight": 0.5, "rationale": "demand softening"}
  ],
  "macro_calendar": ["CPI 2026-05-13", "FOMC 2026-05-01"]
}
```

`decide_now.py --news-json path/to/brief.json` 또는 `decision_now.ipynb`의
`news_brief` 셀에 직접 입력. 뉴스 가중치는 기본 5%, 상한 ±0.6 점수, 0.5 신뢰도로 제한됩니다.

## 커스텀 가중치

```python
from crypto_analysis.engine import DEFAULT_WEIGHTS, fuse
w = dict(DEFAULT_WEIGHTS)
w['funding'] = 0.35
w['news'] = 0.0
result = fuse(signals, weights=w)
```

## 한계 — 읽고 가세요

- 단기 신호이며, 중·장기 거시 레짐 전환(규제·블랙스완)에는 취약합니다.
- 뉴스 신호는 노이즈·편향이 커서 의도적으로 저가중. 중요 이벤트가 있는 날은 수동 부스트를 권장.
- 백테스트는 자유 데이터(퍼플릭 API)로 복원 가능한 범위만 포함 — 옵션 IV·on-chain·news는 빠집니다.
- 퍼블릭 API만 쓰므로 주문·포지션 관리는 포함되지 않습니다.
