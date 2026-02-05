"""
Knowledge Base with RAG (Retrieval Augmented Generation)
지역/시설 정보를 저장하고 검색하여 인터뷰어에게 컨텍스트를 제공합니다.

Vector Store: ChromaDB (로컬 영속성 지원)
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


# --- Knowledge Data Models ---

class LocationInfo(BaseModel):
    """위치/시설 정보"""
    name: str = Field(description="위치/시설 이름")
    type: str = Field(description="유형 (대학교, 역, 공원, 상권 등)")
    region: str = Field(description="지역구")
    characteristics: List[str] = Field(description="주요 특성")
    known_issues: List[str] = Field(default_factory=list, description="알려진 문제점")
    demographics: Optional[str] = Field(None, description="주 이용자층")
    additional_info: Optional[str] = Field(None, description="추가 정보")


class RetrievedContext(BaseModel):
    """검색된 컨텍스트"""
    location_name: str
    relevant_info: List[str]
    suggested_probes: List[str] = Field(description="제안되는 후속 질문들")


# --- Knowledge Base ---

class KnowledgeBase:
    """
    RAG 기반 지식 베이스 (ChromaDB)
    - 지역/시설 정보 저장
    - 유사도 검색
    - 인터뷰 컨텍스트 생성
    """

    def __init__(self, api_key: str = None, data_dir: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self.embeddings = OpenAIEmbeddings()
        self.knowledge_data: Dict[str, LocationInfo] = {}

        # Data directory for Chroma persistence
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data" / "chroma"
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma with persistence
        self.vector_store = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=str(self.data_dir)
        )

        # LLM for generating probing questions
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # Load existing data from Chroma metadata
        self._load_from_chroma()

    def _load_from_chroma(self):
        """Chroma에서 기존 데이터 로드"""
        try:
            collection = self.vector_store._collection
            if collection.count() > 0:
                results = collection.get(include=["metadatas"])
                for metadata in results.get("metadatas", []):
                    if metadata and "name" in metadata:
                        # Reconstruct LocationInfo from metadata
                        info = LocationInfo(
                            name=metadata.get("name", ""),
                            type=metadata.get("type", ""),
                            region=metadata.get("region", ""),
                            characteristics=metadata.get("characteristics", "").split("|") if metadata.get("characteristics") else [],
                            known_issues=metadata.get("known_issues", "").split("|") if metadata.get("known_issues") else [],
                            demographics=metadata.get("demographics"),
                            additional_info=metadata.get("additional_info")
                        )
                        self.knowledge_data[info.name] = info
        except Exception as e:
            print(f"Failed to load from Chroma: {e}")

    def add_location(self, info: LocationInfo):
        """위치/시설 정보 추가"""
        self.knowledge_data[info.name] = info

        # Create document for vector store
        doc_content = self._info_to_text(info)

        # Metadata (Chroma stores this alongside vectors)
        metadata = {
            "name": info.name,
            "type": info.type,
            "region": info.region,
            "characteristics": "|".join(info.characteristics),
            "known_issues": "|".join(info.known_issues),
            "demographics": info.demographics or "",
            "additional_info": info.additional_info or ""
        }

        # Add to Chroma (automatically persisted)
        self.vector_store.add_texts(
            texts=[doc_content],
            metadatas=[metadata],
            ids=[f"loc_{info.name}"]
        )

    def _info_to_text(self, info: LocationInfo) -> str:
        """LocationInfo를 검색 가능한 텍스트로 변환"""
        parts = [
            f"위치: {info.name}",
            f"유형: {info.type}",
            f"지역: {info.region}",
            f"특성: {', '.join(info.characteristics)}"
        ]

        if info.known_issues:
            parts.append(f"알려진 문제: {', '.join(info.known_issues)}")

        if info.demographics:
            parts.append(f"주 이용자: {info.demographics}")

        if info.additional_info:
            parts.append(f"추가정보: {info.additional_info}")

        return "\n".join(parts)

    def search(self, query: str, k: int = 3) -> List[Document]:
        """유사도 검색"""
        return self.vector_store.similarity_search(query, k=k)

    def get_context_for_location(self, location_mention: str) -> Optional[RetrievedContext]:
        """
        위치 언급에 대한 컨텍스트 검색
        인터뷰어가 사용할 수 있는 정보와 후속 질문 제안
        """
        # Search for relevant documents
        docs = self.search(location_mention, k=2)

        if not docs:
            return None

        # Extract relevant info
        relevant_info = []
        for doc in docs:
            relevant_info.append(doc.page_content)

        # Generate probing questions based on context
        suggested_probes = self._generate_probes(location_mention, relevant_info)

        return RetrievedContext(
            location_name=location_mention,
            relevant_info=relevant_info,
            suggested_probes=suggested_probes
        )

    def _generate_probes(self, location: str, context: List[str]) -> List[str]:
        """컨텍스트 기반 후속 질문 생성"""
        context_text = "\n".join(context)

        prompt = f"""다음은 '{location}'에 대한 정보입니다:

{context_text}

이 정보를 바탕으로 인터뷰어가 사용할 수 있는 자연스러운 후속 질문 3개를 생성해주세요.
- 질문은 한국어로
- 정보를 직접 언급하지 말고, 자연스럽게 유도하는 질문
- 응답자의 경험을 끌어내는 질문

예시: "혹시 경사가 있는 구간에서 불편하셨던 적 있으세요?"

질문만 리스트로 출력하세요."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Parse response into list
        lines = response.content.strip().split('\n')
        probes = [line.strip().lstrip('0123456789.-) ') for line in lines if line.strip()]

        return probes[:3]

    def delete_location(self, name: str):
        """위치 삭제"""
        if name in self.knowledge_data:
            del self.knowledge_data[name]
            # Delete from Chroma
            try:
                self.vector_store._collection.delete(ids=[f"loc_{name}"])
            except Exception as e:
                print(f"Failed to delete from Chroma: {e}")

    def save(self):
        """명시적 저장 (Chroma는 자동 저장되지만 호환성 유지)"""
        # Chroma auto-persists, but we can call this for compatibility
        pass

    def load(self, filename: str = "knowledge_base") -> bool:
        """지식 베이스 로드 (Chroma는 자동 로드)"""
        # Chroma automatically loads from persist_directory
        return len(self.knowledge_data) > 0


# --- Sample Data Loader ---

def create_sample_knowledge_base(api_key: str = None) -> KnowledgeBase:
    """샘플 지식 베이스 생성 (부산 지역 예시)"""
    kb = KnowledgeBase(api_key=api_key)

    # 기존 데이터가 있으면 건너뛰기
    if kb.knowledge_data:
        return kb

    sample_locations = [
        LocationInfo(
            name="부산대학교",
            type="대학교",
            region="금정구",
            characteristics=["급경사 캠퍼스", "계단이 많음", "넓은 부지", "산지 지형"],
            known_issues=["경사로 인한 보행 어려움", "겨울철 빙판 위험", "휠체어 접근성 부족"],
            demographics="대학생, 교직원",
            additional_info="해발고도 차이가 큰 캠퍼스, 셔틀버스 운행"
        ),
        LocationInfo(
            name="서면역",
            type="지하철역/상권",
            region="부산진구",
            characteristics=["유동인구 많음", "상업지구", "교통 허브", "지하상가 연결"],
            known_issues=["혼잡한 보행로", "불법 주정차", "야간 취객"],
            demographics="직장인, 쇼핑객, 청년층",
            additional_info="1호선 2호선 환승역, 부산 최대 상권"
        ),
        LocationInfo(
            name="해운대해수욕장",
            type="관광지",
            region="해운대구",
            characteristics=["관광 명소", "해변 산책로", "계절별 인파 변동"],
            known_issues=["여름철 과밀", "모래사장 접근성", "횡단보도 부족"],
            demographics="관광객, 지역주민, 가족단위",
            additional_info="연간 방문객 1000만명 이상"
        ),
        LocationInfo(
            name="광안리해수욕장",
            type="관광지",
            region="수영구",
            characteristics=["광안대교 야경", "카페거리", "젊은층 선호"],
            known_issues=["주차 문제", "야간 소음", "자전거-보행자 충돌"],
            demographics="청년층, 관광객, 데이트족",
            additional_info="광안대교 마라톤 코스"
        ),
        LocationInfo(
            name="감천문화마을",
            type="관광지",
            region="사하구",
            characteristics=["급경사 마을", "좁은 골목", "계단식 구조"],
            known_issues=["고령자 보행 어려움", "경사로 미끄러움", "대중교통 접근성"],
            demographics="관광객, 고령 주민",
            additional_info="한국의 마추픽추, 벽화마을"
        ),
        LocationInfo(
            name="부산역",
            type="교통시설",
            region="동구",
            characteristics=["KTX 정차역", "대중교통 허브", "역세권 개발"],
            known_issues=["복잡한 동선", "노숙자 문제", "택시 호객행위"],
            demographics="여행객, 출장자, 통근자",
            additional_info="부산의 관문, 차이나타운 인접"
        ),
        LocationInfo(
            name="남포동",
            type="상권",
            region="중구",
            characteristics=["전통 상권", "BIFF 광장", "국제시장 인접"],
            known_issues=["노후화된 보도", "관광객 혼잡", "간판 돌출"],
            demographics="관광객, 중장년층",
            additional_info="자갈치시장, 용두산공원 인접"
        ),
        LocationInfo(
            name="센텀시티",
            type="신도시/상권",
            region="해운대구",
            characteristics=["계획도시", "넓은 보도", "현대적 시설"],
            known_issues=["횡단 거리가 김", "그늘 부족", "바람 통로"],
            demographics="직장인, 쇼핑객, 가족단위",
            additional_info="신세계백화점, 영화의전당, BEXCO"
        )
    ]

    for location in sample_locations:
        kb.add_location(location)

    return kb


# --- Global Instance ---

_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(api_key: str = None) -> KnowledgeBase:
    """싱글톤 지식 베이스 인스턴스"""
    global _knowledge_base

    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(api_key=api_key)

    return _knowledge_base


def reset_knowledge_base():
    """지식 베이스 리셋"""
    global _knowledge_base
    _knowledge_base = None
