# 아키텍처 문서 (마이그레이션 가이드)

> **목적**: React + FastAPI 마이그레이션을 위한 코드 구조 및 로직 설명
>
> 이 문서를 읽는 AI 에이전트에게: 이 프로젝트는 Streamlit MVP에서 React(Frontend) + FastAPI(Backend)로 분리될 예정입니다.

---

## 📁 현재 프로젝트 구조

```
test4/
├── src/
│   ├── app.py              # Streamlit UI (→ React로 대체)
│   ├── bot.py              # LangGraph 인터뷰 로직 (→ FastAPI 서비스)
│   ├── db.py               # SQLite 데이터베이스 (→ FastAPI + PostgreSQL)
│   ├── analysis.py         # 시맨틱 분석 (→ FastAPI 서비스)
│   ├── config_loader.py    # YAML 설정 로더 (→ 백엔드 유틸)
│   ├── knowledge_base.py   # RAG/ChromaDB (→ FastAPI 서비스)
│   ├── survey_generator.py # 설문 생성기 (→ FastAPI 서비스)
│   └── main.py             # CLI 버전 (참고용)
├── configs/
│   ├── config.yaml         # 메인 설정
│   ├── topics/             # 주제별 설정
│   └── ui/                 # UI 문자열 (→ i18n)
├── data/
│   └── chroma/             # ChromaDB 벡터 저장소
├── tests/
└── docs/
```

---

## 🔄 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                         CURRENT (Streamlit)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Input                                                     │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────┐    ┌──────────────┐    ┌─────────────┐            │
│   │ app.py  │───▶│   bot.py     │───▶│ knowledge   │            │
│   │ (UI)    │    │ (LangGraph)  │    │ _base.py    │            │
│   └─────────┘    └──────────────┘    │ (RAG)       │            │
│       │                │              └─────────────┘            │
│       │                │                                         │
│       ▼                ▼                                         │
│   ┌─────────┐    ┌──────────────┐                               │
│   │ db.py   │    │ analysis.py  │                               │
│   │(SQLite) │    │ (t-SNE)      │                               │
│   └─────────┘    └──────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      TARGET (React + FastAPI)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────┐         ┌─────────────────────────┐   │
│   │    React Frontend   │  HTTP   │     FastAPI Backend     │   │
│   │                     │◀───────▶│                         │   │
│   │  - Interview Chat   │   API   │  - /api/chat            │   │
│   │  - Dashboard        │         │  - /api/interviews      │   │
│   │  - Config Manager   │         │  - /api/analysis        │   │
│   │  - Knowledge UI     │         │  - /api/knowledge       │   │
│   └─────────────────────┘         │  - /api/config          │   │
│                                   └─────────────────────────┘   │
│                                            │                     │
│                          ┌─────────────────┼─────────────────┐  │
│                          │                 │                 │  │
│                          ▼                 ▼                 ▼  │
│                    ┌──────────┐    ┌────────────┐    ┌───────┐ │
│                    │ PostgreSQL│    │  ChromaDB  │    │ Redis │ │
│                    │ (메인 DB) │    │  (벡터DB)  │    │(세션) │ │
│                    └──────────┘    └────────────┘    └───────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 모듈별 상세 설명

### 1. `bot.py` - 인터뷰 엔진 (핵심)

**역할**: LangGraph 기반 AI 인터뷰어

**주요 클래스**:

```python
# 인터뷰 데이터 스키마 (Pydantic)
class InterviewInfo(BaseModel):
    issue_text: Optional[str]       # 불편 사항 텍스트
    severity_score: Optional[int]   # 심각도 0-4
    primary_category: Optional[str] # 카테고리
    location_bucket: Optional[str]  # 위치
    evidence_span: Optional[str]    # 증거 텍스트

# LLM 응답 스키마
class BotResponse(BaseModel):
    response: str                   # AI 응답 텍스트
    suggested_replies: List[str]    # 추천 답변 버튼
    info_update: InterviewInfo      # 추출된 정보
    current_issue_complete: bool    # 현재 이슈 완료 여부
    new_issue_started: bool         # 새 이슈 시작 여부
    interview_finished: bool        # 인터뷰 종료 여부

# LangGraph 상태
class AgentState(TypedDict):
    messages: List[BaseMessage]     # 대화 히스토리
    info: InterviewInfo             # 현재 수집 정보
    collected_issues: List[dict]    # 완료된 이슈들
    turn_index: int                 # 턴 인덱스
    suggested_replies: List[str]    # 추천 답변
    is_complete: bool               # 완료 여부
    rag_context: str                # RAG 컨텍스트
```

**핵심 클래스 - `ConfigurableInterviewGraph`**:

```python
class ConfigurableInterviewGraph:
    """
    FastAPI 마이그레이션 시 이 클래스를 서비스로 래핑

    주요 메서드:
    - __init__(api_key, topic_id, enable_rag)
    - get_greeting() -> str           # 첫 인사말
    - get_closing() -> str            # 마무리 인사
    - graph.invoke(state) -> state    # 메인 처리 (LangGraph)
    """

    def __init__(self, api_key: str, topic_id: str = None, enable_rag: bool = True):
        # 1. 설정 로드
        self.loader = get_config_loader()
        self.topic_config = self.loader.load_topic_config(topic_id)

        # 2. 인터뷰 모드 결정
        self.interview_mode = self.topic_config.get("interview_mode", "turn_based")
        # "field_based": 반구조화 (추천)
        # "turn_based": 고정 턴 (레거시)

        # 3. RAG 초기화
        if enable_rag:
            self.field_manager = FieldBasedInterviewManager(
                topic_config, language, enable_rag=True
            )

        # 4. LangGraph 빌드
        builder = StateGraph(AgentState)
        builder.add_node("interviewer", self.interviewer_node)
        self.graph = builder.compile()

    def interviewer_node(self, state: AgentState) -> AgentState:
        """
        핵심 로직 - 한 턴 처리

        FastAPI 마이그레이션 시:
        POST /api/chat
        Request: { messages, info, collected_issues, ... }
        Response: { messages, info, collected_issues, suggested_replies, ... }
        """
        # 1. RAG 컨텍스트 검색 (위치 언급 시)
        if self.enable_rag:
            rag_context = self.field_manager.search_context(user_message)

        # 2. 시스템 프롬프트 생성 (수집 상태 포함)
        system_prompt = self.field_manager.build_system_prompt(
            current_info, collected_issues, rag_context
        )

        # 3. LLM 호출 (구조화 출력)
        result: BotResponse = self.structured_llm.invoke(messages)

        # 4. 상태 업데이트 & 반환
        return new_state
```

**FastAPI 변환 예시**:

```python
# backend/routers/chat.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    suggested_replies: List[str]
    info: dict
    is_complete: bool

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. 세션에서 상태 로드 (Redis)
    state = await load_session_state(request.session_id)

    # 2. 메시지 추가
    state["messages"].append(HumanMessage(content=request.message))

    # 3. 그래프 실행
    graph = get_interview_graph()
    result = graph.invoke(state)

    # 4. 세션 상태 저장
    await save_session_state(request.session_id, result)

    # 5. 응답 반환
    return ChatResponse(
        response=result["messages"][-1].content,
        suggested_replies=result["suggested_replies"],
        info=result["info"].model_dump(),
        is_complete=result["is_complete"]
    )
```

---

### 2. `knowledge_base.py` - RAG 시스템

**역할**: ChromaDB 기반 위치/시설 지식 관리

**주요 클래스**:

```python
class LocationInfo(BaseModel):
    """위치 정보 스키마"""
    name: str                       # "부산대학교"
    type: str                       # "대학교"
    region: str                     # "금정구"
    characteristics: List[str]      # ["급경사", "계단 많음"]
    known_issues: List[str]         # ["휠체어 접근성 부족"]
    demographics: Optional[str]     # "대학생, 교직원"
    additional_info: Optional[str]  # "셔틀버스 운행"

class RetrievedContext(BaseModel):
    """RAG 검색 결과"""
    location_name: str
    relevant_info: List[str]        # 관련 문서 내용
    suggested_probes: List[str]     # AI 생성 후속 질문

class KnowledgeBase:
    """
    FastAPI 마이그레이션 시:
    - 싱글톤 → 의존성 주입
    - 파일 저장 → 별도 ChromaDB 서버 고려
    """

    def __init__(self, api_key: str, data_dir: str = None):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=str(data_dir)  # → 프로덕션: ChromaDB 서버
        )

    def add_location(self, info: LocationInfo):
        """위치 정보 추가 (임베딩 자동 생성)"""

    def search(self, query: str, k: int = 3) -> List[Document]:
        """유사도 검색"""

    def get_context_for_location(self, mention: str) -> RetrievedContext:
        """
        핵심 메서드 - 인터뷰 중 호출

        1. 벡터 검색으로 관련 문서 찾기
        2. GPT로 후속 질문 생성
        3. RetrievedContext 반환
        """
```

**FastAPI 엔드포인트**:

```python
# backend/routers/knowledge.py

@router.get("/api/knowledge")
async def list_knowledge():
    """모든 위치 정보 목록"""

@router.post("/api/knowledge")
async def add_knowledge(info: LocationInfo):
    """새 위치 정보 추가"""

@router.delete("/api/knowledge/{name}")
async def delete_knowledge(name: str):
    """위치 정보 삭제"""

@router.post("/api/knowledge/search")
async def search_knowledge(query: str, k: int = 3):
    """유사도 검색 (디버깅/테스트용)"""
```

---

### 3. `db.py` - 데이터베이스

**역할**: 인터뷰 결과 저장

**현재 스키마 (SQLite)**:

```sql
CREATE TABLE interviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,              -- 세션 그룹핑용
    issue_text TEXT,
    severity_score INTEGER,
    primary_category TEXT,
    location_bucket TEXT,
    evidence_span TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**주요 함수**:

```python
def init_db():
    """테이블 생성 (최초 1회)"""

def insert_interview(info: dict) -> int:
    """단일 이슈 저장"""

def insert_multiple_issues(issues: List[dict], session_id: str) -> str:
    """
    다중 이슈 저장 (같은 session_id로 그룹핑)

    FastAPI 마이그레이션 시:
    - SQLite → PostgreSQL
    - 동기 → 비동기 (asyncpg)
    """

def get_all_interviews() -> List[dict]:
    """전체 조회"""

def generate_session_id() -> str:
    """UUID 기반 세션 ID 생성"""
```

**PostgreSQL 마이그레이션 스키마**:

```sql
-- 세션 테이블 분리
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    topic_id VARCHAR(100),
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'in_progress'
);

-- 이슈 테이블
CREATE TABLE issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    issue_text TEXT,
    severity_score INTEGER CHECK (severity_score BETWEEN 0 AND 4),
    primary_category VARCHAR(50),
    location_bucket VARCHAR(100),
    evidence_span TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 인덱스
CREATE INDEX idx_issues_session ON issues(session_id);
CREATE INDEX idx_issues_category ON issues(primary_category);
CREATE INDEX idx_issues_location ON issues(location_bucket);
```

---

### 4. `analysis.py` - 시맨틱 분석

**역할**: t-SNE 클러스터링 + 토픽 라벨링

**주요 클래스**:

```python
class SemanticAnalyzer:
    """
    FastAPI 마이그레이션 시:
    - 동기 처리 → 백그라운드 태스크 (Celery)
    - 대용량 데이터 시 배치 처리
    """

    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def process_and_analyze(
        self,
        df: pd.DataFrame,
        text_column: str = 'issue_text',
        n_clusters: int = 5,
        n_dimensions: int = 3
    ) -> pd.DataFrame:
        """
        메인 파이프라인:
        1. 텍스트 → 임베딩 (OpenAI)
        2. K-Means 클러스터링
        3. t-SNE 차원 축소 (3D)
        4. GPT로 클러스터 라벨 생성

        반환: 원본 df + x, y, z, cluster, topic_label 컬럼
        """
```

**FastAPI 엔드포인트**:

```python
# backend/routers/analysis.py

@router.post("/api/analysis/run")
async def run_analysis(background_tasks: BackgroundTasks):
    """
    분석 시작 (백그라운드)
    Returns: { task_id: "..." }
    """
    task_id = str(uuid4())
    background_tasks.add_task(run_semantic_analysis, task_id)
    return {"task_id": task_id}

@router.get("/api/analysis/status/{task_id}")
async def get_analysis_status(task_id: str):
    """분석 진행 상태 조회"""

@router.get("/api/analysis/result/{task_id}")
async def get_analysis_result(task_id: str):
    """분석 결과 조회 (3D 좌표 + 라벨)"""
```

---

### 5. `config_loader.py` - 설정 관리

**역할**: YAML 설정 파일 로드

**주요 함수/클래스**:

```python
def get_configs_dir() -> Path:
    """configs/ 디렉토리 경로"""

def get_config_loader() -> ConfigLoader:
    """싱글톤 ConfigLoader 인스턴스"""

class ConfigLoader:
    """
    YAML 설정 로더

    FastAPI 마이그레이션 시:
    - 환경 변수 우선 (12-factor app)
    - Pydantic Settings 활용
    """

    def __init__(self):
        self.config = self._load_yaml("config.yaml")
        self.active_topic = self.config.get("active_topic")
        self.language = self.config.get("language", "ko")

    def load_topic_config(self, topic_id: str = None) -> dict:
        """topics/{topic_id}.yaml 로드"""

    def load_ui_strings(self) -> dict:
        """ui/{language}.yaml 로드"""

    def get_localized(self, obj: dict, key: str = None) -> str:
        """다국어 값 추출"""

class TurnManager:
    """턴 기반 인터뷰 관리 (레거시)"""

class DynamicSchemaBuilder:
    """동적 Pydantic 모델 생성 (미사용)"""
```

**Pydantic Settings 마이그레이션**:

```python
# backend/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 환경 변수
    OPENAI_API_KEY: str
    DATABASE_URL: str = "postgresql://..."
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000

    # 앱 설정
    ACTIVE_TOPIC: str = "busan_walkability_v2"
    LANGUAGE: str = "ko"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

### 6. `survey_generator.py` - 설문 생성기

**역할**: 자연어 → YAML 설정 변환

```python
class SurveyConfigGenerator:
    """
    GPT를 사용해 자연어 설명을 YAML 설정으로 변환

    FastAPI 마이그레이션 시:
    POST /api/config/generate
    """

    def generate_config(self, description: str) -> SurveyConfig:
        """
        입력: "카페 고객 만족도 조사를 하고 싶어..."
        출력: SurveyConfig (Pydantic 모델)
        """

    def config_to_yaml(self, config: SurveyConfig) -> str:
        """SurveyConfig → YAML 문자열"""

    def save_config(self, config: SurveyConfig) -> Path:
        """configs/topics/{id}.yaml 저장"""
```

---

### 7. `app.py` - Streamlit UI (React 대체 대상)

**현재 구조**:

```python
# 탭 1: 인터뷰 채팅
with tab1:
    # 세션 상태 관리
    st.session_state.messages      # 대화 히스토리
    st.session_state.interview_info # 현재 수집 정보
    st.session_state.collected_issues # 완료된 이슈들
    st.session_state.bot_graph     # LangGraph 인스턴스

    # 채팅 UI
    for message in messages:
        st.chat_message(role).markdown(content)

    # 입력 처리
    if prompt := st.chat_input():
        result = bot_graph.invoke(state)

# 탭 2: 대시보드
with tab2:
    # 통계 카드
    st.metric("총 인터뷰", len(df))

    # 차트
    st.bar_chart(category_counts)

    # 3D 시각화
    fig = px.scatter_3d(result_df, x='x', y='y', z='z', color='topic_label')
    st.plotly_chart(fig)

# 탭 3: 설문 설정
with tab3:
    # 자연어 입력
    description = st.text_area("설문 설명")

    # AI 생성
    config = generator.generate_config(description)

    # YAML 미리보기 & 저장
    st.code(yaml_content, language='yaml')

# 탭 4: 지식 관리
with tab4:
    # CRUD UI
    # 검색 테스트
```

**React 컴포넌트 구조 제안**:

```
frontend/src/
├── components/
│   ├── Chat/
│   │   ├── ChatWindow.tsx      # 메시지 목록
│   │   ├── ChatInput.tsx       # 입력창 + 추천 버튼
│   │   ├── Message.tsx         # 개별 메시지
│   │   └── SuggestedReplies.tsx
│   ├── Dashboard/
│   │   ├── StatsCards.tsx      # 통계 카드
│   │   ├── CategoryChart.tsx   # 카테고리 분포
│   │   ├── LocationChart.tsx   # 위치 분포
│   │   └── SemanticViewer.tsx  # 3D t-SNE (Three.js/react-three-fiber)
│   ├── Config/
│   │   ├── SurveyGenerator.tsx # 자연어 입력
│   │   ├── YamlPreview.tsx     # YAML 미리보기
│   │   └── TopicSelector.tsx   # 주제 선택
│   └── Knowledge/
│       ├── KnowledgeList.tsx   # 목록
│       ├── KnowledgeForm.tsx   # 추가/수정 폼
│       ├── FileUpload.tsx      # CSV/JSON 업로드
│       └── SearchTest.tsx      # RAG 검색 테스트
├── hooks/
│   ├── useChat.ts              # 채팅 상태 관리
│   ├── useInterview.ts         # 인터뷰 세션 관리
│   └── useAnalysis.ts          # 분석 결과 조회
├── services/
│   └── api.ts                  # FastAPI 호출
├── store/
│   └── interviewStore.ts       # Zustand/Redux
└── types/
    └── index.ts                # 타입 정의
```

---

## 🔌 API 명세 (FastAPI 설계)

```yaml
openapi: 3.0.0
info:
  title: Interview Platform API
  version: 1.0.0

paths:
  # 인터뷰
  /api/sessions:
    post:
      summary: 새 인터뷰 세션 시작
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  session_id: { type: string }
                  greeting: { type: string }

  /api/sessions/{session_id}/chat:
    post:
      summary: 메시지 전송 & 응답 받기
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                message: { type: string }
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  response: { type: string }
                  suggested_replies: { type: array }
                  info: { type: object }
                  is_complete: { type: boolean }

  /api/sessions/{session_id}/complete:
    post:
      summary: 인터뷰 완료 & 저장

  # 인터뷰 결과
  /api/interviews:
    get:
      summary: 전체 인터뷰 목록
      parameters:
        - name: skip
          in: query
          schema: { type: integer, default: 0 }
        - name: limit
          in: query
          schema: { type: integer, default: 100 }

  # 분석
  /api/analysis/run:
    post:
      summary: 시맨틱 분석 시작 (백그라운드)
      responses:
        202:
          content:
            application/json:
              schema:
                type: object
                properties:
                  task_id: { type: string }

  /api/analysis/{task_id}:
    get:
      summary: 분석 결과 조회

  # 지식 베이스
  /api/knowledge:
    get:
      summary: 전체 위치 정보 목록
    post:
      summary: 새 위치 정보 추가

  /api/knowledge/{name}:
    delete:
      summary: 위치 정보 삭제

  /api/knowledge/search:
    post:
      summary: RAG 검색 테스트

  # 설정
  /api/config/topics:
    get:
      summary: 사용 가능한 주제 목록

  /api/config/topics/{topic_id}:
    get:
      summary: 주제 설정 조회

  /api/config/generate:
    post:
      summary: 자연어 → 설정 생성
```

---

## 🗃️ 세션 상태 관리

**현재 (Streamlit)**:
```python
st.session_state.messages
st.session_state.interview_info
st.session_state.collected_issues
st.session_state.is_complete
```

**마이그레이션 (Redis)**:

```python
# backend/core/session.py
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def save_session_state(session_id: str, state: dict):
    """세션 상태 저장 (TTL 24시간)"""
    redis_client.setex(
        f"session:{session_id}",
        86400,  # 24시간
        json.dumps(state, default=str)
    )

async def load_session_state(session_id: str) -> dict:
    """세션 상태 로드"""
    data = redis_client.get(f"session:{session_id}")
    if data:
        return json.loads(data)
    return None

async def delete_session(session_id: str):
    """세션 삭제"""
    redis_client.delete(f"session:{session_id}")
```

---

## 📝 마이그레이션 체크리스트

### Phase 1: 백엔드 (FastAPI)

- [ ] 프로젝트 구조 생성
  ```
  backend/
  ├── app/
  │   ├── main.py
  │   ├── core/
  │   │   ├── config.py
  │   │   └── session.py
  │   ├── routers/
  │   │   ├── chat.py
  │   │   ├── interviews.py
  │   │   ├── analysis.py
  │   │   ├── knowledge.py
  │   │   └── config.py
  │   ├── services/
  │   │   ├── interview_service.py
  │   │   ├── knowledge_service.py
  │   │   └── analysis_service.py
  │   ├── models/
  │   │   └── schemas.py
  │   └── db/
  │       ├── database.py
  │       └── models.py
  ├── tests/
  └── requirements.txt
  ```
- [ ] SQLite → PostgreSQL 마이그레이션
- [ ] Redis 세션 관리
- [ ] bot.py 로직 서비스로 래핑
- [ ] knowledge_base.py 서비스로 래핑
- [ ] analysis.py 백그라운드 태스크로 변환
- [ ] API 문서 자동 생성 (Swagger)
- [ ] CORS 설정
- [ ] 인증/인가 (JWT)

### Phase 2: 프론트엔드 (React)

- [ ] Vite + React + TypeScript 세팅
- [ ] 컴포넌트 구조 생성
- [ ] API 클라이언트 (axios/fetch)
- [ ] 상태 관리 (Zustand/Redux)
- [ ] 채팅 UI
- [ ] 대시보드 (Chart.js/Recharts)
- [ ] 3D 시각화 (Three.js/react-three-fiber)
- [ ] 폼 관리 (React Hook Form)
- [ ] 스타일링 (Tailwind CSS)

### Phase 3: 인프라

- [ ] Docker Compose (API + DB + Redis + Chroma)
- [ ] 환경 분리 (dev/staging/prod)
- [ ] CI/CD (GitHub Actions)
- [ ] 모니터링 (Prometheus + Grafana)
- [ ] 로깅 (ELK Stack)

---

## 🎯 핵심 포인트 요약

1. **bot.py의 `ConfigurableInterviewGraph`** 가 핵심 비즈니스 로직
2. **상태(state)**는 Redis로 관리 (messages, info, collected_issues)
3. **RAG**는 ChromaDB 서버로 분리 고려
4. **분석**은 백그라운드 태스크로 처리
5. **설정**은 YAML → 환경 변수 + DB 조합으로

---

> 이 문서를 기반으로 마이그레이션을 진행하면 됩니다.
> 추가 질문이 있으면 이 파일을 참조하세요.
