# 🏙️ 부산 걷기 좋은 도시 만들기 - AI 대화형 평가 시스템

> **"시민의 목소리를 데이터로, 데이터를 정책으로."**
> 
> 본 프로젝트는 생성형 AI(LLM)를 활용하여 시민들과 1:1 심층 인터뷰를 수행하고, 수집된 비정형 데이터를 실시간으로 분석/시각화하는 **지능형 공공디자인 평가 플랫폼**입니다.

---

## 🚀 주요 기능 (Key Features)

### 1. 🤖 AI 하이브리드 인터뷰어 (Hybrid Chatbot)
- **LangGraph 기반 상태 머신**: 단순 챗봇이 아닌, 목표 지향적인(Goal-oriented) 인터뷰를 수행합니다.
- **적응형 프로빙 (Adaptive Probing)**: 시민의 답변에 따라 "왜 그렇게 느끼셨나요?", "구체적으로 어디인가요?"와 같은 심층 질문을 던집니다.
- **역코딩 (Inverse Coding)**: 대화 내용을 실시간으로 분석하여 **장소, 이슈, 해결책, 심각도** 등의 정형 데이터로 자동 변환합니다.

### 2. 📊 실시간 분석 대시보드 (Real-time Dashboard)
- **통합형 UI**: 채팅과 데이터 분석을 하나의 앱(`src/app.py`)에서 탭으로 전환하며 경험할 수 있습니다.
- **3D 의미 연결망 (Semantic Network)**: 수집된 이슈들을 인공지능이 분석하여 3차원 공간에 군집화(Clustering)하여 보여줍니다.
- **토픽 상세 카드**: "안전", "접근성" 등 AI가 자동 분류한 주제별 리포트를 카드 형태로 제공합니다.

### 3. 📈 데이터 과학적 접근
- **NLP & Embedding**: OpenAI의 최신 임베딩 모델을 활용해 텍스트의 의미적 유사도를 계산합니다.
- **K-Means & t-SNE**: 고차원 데이터를 군집화하고 시각화하여 숨겨진 여론의 패턴을 발굴합니다.

---

## 🛠️ 설치 및 실행 (Installation & Run)

### 1. 환경 설정 (Prerequisites)
Python 3.10 이상이 필요합니다.

```bash
# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 입력하세요.

```env
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. 애플리케이션 실행
하나의 명령어로 통합 앱을 실행합니다.

```bash
streamlit run src/app.py
```

브라우저가 자동으로 열리며, **[인터뷰]** 탭과 **[대시보드]** 탭을 오가며 사용할 수 있습니다.

---

## 📂 프로젝트 구조 (Project Structure)

```
📂 test4
├── 📂 src
│   ├── app.py          # 메인 애플리케이션 (Streamlit UI + Chat + Dashboard)
│   ├── bot.py          # 챗봇 로직 (LangGraph, OpenAI LLM)
│   ├── db.py           # 데이터베이스 처리 (SQLite)
│   ├── analysis.py     # 데이터 분석 엔진 (Embedding, Clustering, t-SNE)
│   └── seed_rich.py    # (테스트용) 가상 데이터 생성 스크립트
├── 📜 requirements.txt # 의존성 패키지 목록
└── 📜 설계방식_분석보고서.md # 시스템 설계 이론 및 방법론 보고서
```

---

## 📝 설계 철학
이 시스템은 단순한 설문조사를 넘어, **정성적 데이터(Qualitative Data)의 정량화(Quantification)**를 목표로 합니다. 리커트 척도(1~5점)로는 담을 수 없는 시민의 구체적인 경험과 맥락을 보존하면서도, 통계적으로 유의미한 패턴을 도출해내는 **과학적 행정 도구**입니다.

자세한 내용은 포함된 `설계방식_분석보고서.md`를 참고하세요.
