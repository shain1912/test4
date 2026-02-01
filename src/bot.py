import os
import json
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field

# --- Schema Definitions ---

class InterviewInfo(BaseModel):
    """Structured information extracted from the interview."""
    location: Optional[str] = Field(None, description="Detailed location mentioned")
    urban_element: Optional[str] = Field(None, description="Specific facility or element (e.g., Road, Bus Stop)")
    issue: Optional[str] = Field(None, description="Core problem described")
    solution_type: Optional[str] = Field(None, description="Type of solution (Infrastructure/Policy/Service)")
    solution_detail: Optional[str] = Field(None, description="Details of the proposed solution")
    solution_logic: Optional[str] = Field(None, description="Why the user thinks this solves the problem")
    primary_value: Optional[str] = Field(None, description="Safety/Convenience/Aesthetics/etc")
    willingness_to_pay: Optional[str] = Field(None, description="User's willingness to accept trade-offs (High/Medium/Low)")

class BotResponse(BaseModel):
    """The structured response from the interviewer bot."""
    response: str = Field(description="The natural language response to the user.")
    current_topic: str = Field(description="The specific topic/state of this question (e.g., 'ask_location', 'ask_issue', 'ask_solution_tradeoff').")
    info_update: Optional[InterviewInfo] = Field(None, description="Any new or updated information extracted from the latest turn.")

class AgentState(TypedDict):
    """The tracking state of the interview graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    info: InterviewInfo
    topics_covered: List[str]

# --- Prompt ---

SYSTEM_PROMPT = """
당신은 '부산 걷기 좋은 도시 만들기' 프로젝트의 전문 질적 연구원입니다.
시민들과 1:1 채팅을 통해 보행 환경에 대한 **구체적 경험(Fact)**과 **감정(Feeling)**을 수집합니다.

# 현재 수집된 정보:
{info}

# 이미 다룬 주제 (중복 질문 금지):
{topics_covered}

# 인터뷰 진행 가이드
1. **위치 확인**: 어디서 걷고 있는지 묻습니다. (이미 알면 묻지 않음)
2. **이슈 발굴**: 불편한 점, 위험한 점을 구체적으로 묻습니다.
3. **심층 탐구 (5 Whys)**: 문제의 원인과 구체적 상황을 파고듭니다.
4. **해결책 및 검증**: 사용자가 생각하는 등을 묻고, 그 대안의 트레이드오프(비용, 불편함 등)에 대해 질문합니다.

# 필수 지침 (Strict Rules)
- **중복 금지**: '이미 다룬 주제'에 있는 내용은 절대 다시 묻지 마세요.
- **단일 질문**: 한 번에 질문은 하나만 하세요.
- **중립성**: 답을 유도하지 마세요.
- **기계적 공감 금지**: "불편하셨겠네요" 대신 구체적 상황을 물으세요.

너의 응답은 반드시 주어진 JSON 포맷(response, current_topic, info_update)을 따라야 해.
"""

# --- Node Logic ---

class BusanDesignGraph:
    def __init__(self, api_key=None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.structured_llm = self.llm.with_structured_output(BotResponse)
        
        # Build Graph
        builder = StateGraph(AgentState)
        builder.add_node("interviewer", self.interviewer_node)
        builder.add_edge(START, "interviewer")
        builder.add_edge("interviewer", END)
        
        self.graph = builder.compile()

    def interviewer_node(self, state: AgentState):
        # 1. Prepare accumulated info for context
        current_info = state.get("info", InterviewInfo()).dict()
        covered = state.get("topics_covered", [])
        
        # 2. Format System Prompt
        sys_msg = SystemMessage(content=SYSTEM_PROMPT.format(
            info=json.dumps(current_info, indent=2, ensure_ascii=False),
            topics_covered=covered
        ))
        
        # 3. Invoke LLM
        # We pass full message history. 
        # Note: We must ensure messages are strictly correctly formatted.
        messages = [sys_msg] + state["messages"]
        result: BotResponse = self.structured_llm.invoke(messages)
        
        # 4. Update State
        # We need to merge new info into old info
        new_info = state.get("info", InterviewInfo())
        if result.info_update:
            update_dict = result.info_update.dict(exclude_unset=True, exclude_none=True)
            new_info = new_info.copy(update=update_dict)
            
        new_topic = result.current_topic
        if new_topic not in covered:
            covered.append(new_topic)
            
        return {
            "messages": [AIMessage(content=result.response)],
            "info": new_info,
            "topics_covered": covered
        }
