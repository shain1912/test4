import os
import json
from typing import TypedDict, Annotated, List, Optional
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field

# --- Schema Definitions ---

class PrimaryCategory(str, Enum):
    SAFETY = "안전 (Safety)"
    ACCESSIBILITY = "접근성 (Accessibility)"
    WAYFINDING = "길찾기 (Wayfinding)"
    COMFORT = "쾌적성/미관 (Comfort)"
    OTHER = "기타 (Other)"

class InterviewInfo(BaseModel):
    """Structured information extracted from the fixed 4-turn interview."""
    issue_text: Optional[str] = Field(None, description="The user's original complaint text")
    severity_score: Optional[int] = Field(None, description="Severity of the issue 0-4 (0=Not bad, 4=Very severe)")
    primary_category: Optional[PrimaryCategory] = Field(None, description="Category of the issue")
    location_bucket: Optional[str] = Field(None, description="Rough location bucket (e.g., 'Seomyeon Intersection')")
    evidence_span: Optional[str] = Field(None, description="Evidence text from user input supporting the analysis")

class BotResponse(BaseModel):
    """The structured response from the interviewer bot."""
    response: str = Field(description="The natural language response to the user.")
    suggested_replies: List[str] = Field(default_factory=list, description="List of suggested replies to show as buttons.")
    info_update: Optional[InterviewInfo] = Field(None, description="Any new or updated information extracted.")
    next_turn: int = Field(..., description="The next turn index to proceed to.")
    early_exit: bool = Field(False, description="True if user wants to quit or is annoyed.")

class AgentState(TypedDict):
    """The tracking state of the interview graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    info: InterviewInfo
    turn_index: int  # Tracks the strict flow: 0 -> 1 -> 2 -> 3 -> 4 (End)
    suggested_replies: List[str]  # For UI buttons

# --- Prompts ---

PROMPT_TEMPLATES = {
    0: """
    [Turn 1: Issue Collection]
    User has just started.
    Goal: Ask them about the most uncomfortable thing they experienced while walking in Busan.
    Style: Polite, welcoming, but direct.
    """,
    1: """
    [Turn 2: Severity Measurement]
    User has provided an issue: "{issue_text}"
    Goal: Ask them to rate the severity on a scale of 0 to 4.
    0 = 별로 심각하지 않음
    4 = 매우 심각하고 위험함
    Output: Provide suggested_replies ["0 (별로)", "1 (조금)", "2 (보통)", "3 (심각)", "4 (매우 심각)"].
    """,
    2: """
    [Turn 3: Categorization]
    User has rated severity: {severity_score}.
    Goal: Ask them to categorize this issue.
    Options: 안전, 접근성, 길찾기, 쾌적성, 기타.
    Output: Provide suggested_replies ["안전", "접근성", "길찾기", "쾌적성", "기타"].
    """,
    3: """
    [Turn 4: Location Identification]
    User Category: {primary_category}.
    Goal: Ask for the location.
    Style: Ask for a rough location (e.g., "Near Seomyeon Station") so they feel their privacy is safe.
    """,
    4: """
    [Ending]
    All data collected.
    Goal: Thank the user and tell them their opinion has been recorded as valuable data.
    """
}

COMMON_INSTRUCTIONS = """
You are a 'Public Design Researcher' AI.
Extract structured data into `info_update`.
Follow the `turn_index` progression strictly.
If the user says 'quit', 'stop', 'annoying', set `early_exit=True`.
"""

# --- Node Logic ---

class BusanDesignGraph:
    def __init__(self, api_key=None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3) # Lower temp for consistent flow
        self.structured_llm = self.llm.with_structured_output(BotResponse)
        
        # Build Graph
        builder = StateGraph(AgentState)
        builder.add_node("interviewer", self.interviewer_node)
        builder.add_edge(START, "interviewer")
        builder.add_edge("interviewer", END)
        
        self.graph = builder.compile()

    def interviewer_node(self, state: AgentState):
        current_turn = state.get("turn_index", 0)
        current_info = state.get("info", InterviewInfo())
        
        # Determine Prompt based on NEXT turn (predicting what to ask NEXT)
        # But actually, we are engaging based on PREVIOUS input.
        # Let's align: 
        # State turn_index 0: Initial state. Bot asks Q1.
        # State turn_index 1: User answered Q1. Bot analyzes A1, asks Q2.
        
        # Construct Prompt
        turn_prompt = PROMPT_TEMPLATES.get(current_turn, PROMPT_TEMPLATES[4])
        
        # Fill dynamic slots
        prompt_content = turn_prompt.format(
            issue_text=current_info.issue_text or "General Issue",
            severity_score=current_info.severity_score if current_info.severity_score is not None else "?",
            primary_category=current_info.primary_category or "Unknown"
        )
        
        full_system_prompt = f"{COMMON_INSTRUCTIONS}\n\nCurrent Turn: {current_turn}\n{prompt_content}"
        
        # Invoke LLM
        messages = [SystemMessage(content=full_system_prompt)] + state["messages"]
        result: BotResponse = self.structured_llm.invoke(messages)
        
        # Update State
        new_info = current_info.copy()
        if result.info_update:
            update_dict = result.info_update.dict(exclude_unset=True, exclude_none=True)
            new_info = new_info.copy(update=update_dict)
            
        # Determine next turn
        # If early exit, jump to end (e.g., 99)
        next_turn_val = result.next_turn
        if result.early_exit:
            next_turn_val = 99
            
        return {
            "messages": [AIMessage(content=result.response)],
            "info": new_info,
            "turn_index": next_turn_val,
            "suggested_replies": result.suggested_replies
        }
