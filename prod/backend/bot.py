import os
import warnings
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field

# Suppress annoying Pydantic v2 serialization warnings from langchain structured output
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")


from config_loader import (
    get_config_loader,
    TurnManager,
    DynamicSchemaBuilder,
    ConfigLoader
)

# --- Schema Definitions ---

class InterviewInfo(BaseModel):
    """Structured information for a single issue."""
    issue_text: Optional[str] = Field(None, description="The user's original complaint text")
    severity_score: Optional[int] = Field(None, description="Severity of the issue 0-4 (0=Not bad, 4=Very severe)")
    primary_category: Optional[str] = Field(None, description="Category of the issue")
    location_bucket: Optional[str] = Field(None, description="Rough location bucket (e.g., 'Seomyeon Intersection')")
    evidence_span: Optional[str] = Field(None, description="Evidence text from user input supporting the analysis")

class AnalysisResult(BaseModel):
    """Combined Extraction and Planning output from the Analyzer node"""
    info_update: Optional[InterviewInfo] = Field(None, description="Updated information extracted from the conversation.")
    new_issue_started: bool = Field(False, description="True if user mentioned a NEW/DIFFERENT issue.")
    interview_finished: bool = Field(False, description="True if user wants to end the entire interview.")
    early_exit: bool = Field(False, description="True if user wants to quit or is annoyed.")
    next_goal: str = Field(description="The ONE specific goal for the generator to achieve in the next response.")

class GenerationResult(BaseModel):
    """Generator output for final response"""
    suggested_replies: List[str] = Field(default_factory=list, description="List of suggested replies (optional).")

class BotResponse(BaseModel):
    """Legacy structured response from the interviewer bot (for turn_based mode)."""
    response: str = Field(description="The natural language response to the user.")
    suggested_replies: List[str] = Field(default_factory=list, description="List of suggested replies (optional).")
    info_update: Optional[InterviewInfo] = Field(None, description="Updated information extracted from the conversation.")
    current_issue_complete: bool = Field(False, description="True if current issue has all required information.")
    new_issue_started: bool = Field(False, description="True if user mentioned a NEW/DIFFERENT issue.")
    interview_finished: bool = Field(False, description="True if user wants to end the entire interview.")
    early_exit: bool = Field(False, description="True if user wants to quit or is annoyed.")
    next_turn: int = Field(0, description="The next turn index (for turn_based mode).")

class AgentState(TypedDict):
    """The tracking state of the interview graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    info: InterviewInfo
    collected_issues: List[dict]
    turn_index: int
    suggested_replies: List[str]
    is_complete: bool
    rag_context: str
    next_goal: str  # Added for multi-node planning

# --- Managers ---

class FieldBasedInterviewManager:
    """
    Manages semi-structured interviews.
    Supports RAG-based contextual knowledge.
    """
    def __init__(self, topic_config: Dict[str, Any], language: str = "ko", enable_rag: bool = True):
        self.topic_config = topic_config
        self.language = language
        self.required_fields = topic_config.get("required_fields", [])
        self.system_prompt_config = topic_config.get("system_prompt", {})
        self.enable_rag = enable_rag
        self.knowledge_base = None

        if enable_rag:
            try:
                from knowledge_base import get_knowledge_base
                self.knowledge_base = get_knowledge_base()
            except Exception as e:
                print(f"RAG initialization failed: {e}")
                self.enable_rag = False

    def get_missing_fields(self, info: InterviewInfo) -> List[Dict[str, Any]]:
        info_dict = info.model_dump()
        missing = []
        for field in self.required_fields:
            if info_dict.get(field["id"]) is None:
                missing.append(field)
        return missing

    def get_collected_fields(self, info: InterviewInfo) -> List[Dict[str, Any]]:
        info_dict = info.model_dump()
        collected = []
        for field in self.required_fields:
            if info_dict.get(field["id"]) is not None:
                collected.append({
                    **field,
                    "value": info_dict[field["id"]]
                })
        return collected

    def search_context(self, user_message: str) -> str:
        if not self.enable_rag or not self.knowledge_base:
            return ""
        try:
            context = self.knowledge_base.get_context_for_location(user_message)
            if context:
                return f"""
## 🔍 관련 배경 지식 (RAG)
**감지된 위치:** {context.location_name}
**관련 정보:**
{chr(10).join(['- ' + info for info in context.relevant_info])}
**제안되는 후속 질문 (자연스럽게 활용하세요):**
{chr(10).join(['- ' + probe for probe in context.suggested_probes])}
"""
        except Exception as e:
            print(f"RAG search error: {e}")
        return ""


# --- Configurable Interview Graph ---

class ConfigurableInterviewGraph:
    def __init__(self, api_key: str = None, topic_id: str = None, enable_rag: bool = True):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key

        self.loader = get_config_loader()
        self.topic_config = self.loader.load_topic_config(topic_id)
        self.interview_mode = self.topic_config.get("interview_mode", "turn_based")
        self.enable_rag = enable_rag and self.topic_config.get("enable_rag", True)

        llm_config = self.loader.llm_config
        self.llm = ChatOpenAI(
            model=llm_config.get("model", "gpt-5-nano"),
            temperature=llm_config.get("temperature", 0.3)
        )
        self.turn_manager = None
        self.field_manager = None

        builder = StateGraph(AgentState)

        if self.interview_mode == "field_based":
            self.field_manager = FieldBasedInterviewManager(self.topic_config, self.loader.language, self.enable_rag)
            
            # --- Two-Node Architecture for Performance ---
            builder.add_node("analyzer", self.analyzer_node)
            builder.add_node("generator", self.generator_node)

            builder.add_edge(START, "analyzer")
            builder.add_edge("analyzer", "generator")
            builder.add_edge("generator", END)
            
            # Sub-LLMs with specific structured outputs
            self.analyzer_llm = self.llm.with_structured_output(AnalysisResult)
            
            # Use raw LLM with a specific tag for the generator to enable token streaming
            self.generator_llm = self.llm.with_config({"tags": ["generator_node_llm"]})

        else:
            # Legacy Turn Based
            self.turn_manager = TurnManager(self.topic_config, self.loader.language)
            self.structured_llm = self.llm.with_structured_output(BotResponse)
            
            builder.add_node("interviewer", self._turn_based_node)
            builder.add_edge(START, "interviewer")
            builder.add_edge("interviewer", END)

        self.graph = builder.compile()

    def get_greeting(self) -> str:
        return self.loader.get_localized(self.topic_config.get("greeting", {}))

    def get_closing(self) -> str:
        return self.loader.get_localized(self.topic_config.get("closing", {}))

    # =========================================================================
    # MULTI-NODE ARCHITECTURE (FIELD_BASED)
    # =========================================================================

    def analyzer_node(self, state: AgentState):
        """Node 1: Extract info AND decide the NEXT GOAL simultaneously (saves 1 LLM roundtrip)."""
        if state.get("is_complete"):
            return {"next_goal": "Acknowledge the end of the interview.", "is_complete": True}
            
        current_info = state.get("info", InterviewInfo())
        collected_issues = state.get("collected_issues", [])
        missing = self.field_manager.get_missing_fields(current_info)
        
        sys_prompt = f"""You are the Interview Analyzer & Planner.
1. Extract valid information from the user's latest message based on the existing collected items and update info_update.
   - For 'primary_category', auto-classify it based on the issue context! NEVER ask the user to classify it themselves.
2. Look at the missing fields and the conversation context.
3. Decide EXACTLY ONE GOAL for the generator to ask next.
"""
        if missing:
            missing_info = []
            for f in missing:
                if f["id"] == "primary_category": continue # Skip category as it should be auto-inferred by the Extractor
                
                name = f.get("name", {}).get(self.loader.language, f["id"])
                desc = f.get("description", {}).get(self.loader.language, "")
                missing_info.append(f" - {f['id']} ({name}): {desc}")
                
            if missing_info:
                sys_prompt += f"\nCurrently Missing Fields:\n" + "\n".join(missing_info) + "\n\nYour task: Pick ONLY ONE missing field to ask about. Set it as the next_goal. Define the goal in natural language (e.g. 'Ask for the location')."
            else:
                sys_prompt += "\nAll fields for current issue are collected! Set next_goal to ask if the user has any other entirely different issues to report."
        else:
            sys_prompt += "\nAll fields for current issue are collected! Set next_goal to ask if the user has any other entirely different issues to report."

        messages = [SystemMessage(content=sys_prompt)] + state["messages"][-2:]
        result: AnalysisResult = self.analyzer_llm.invoke(messages)

        new_info = current_info.model_copy()
        if result.info_update:
            update_dict = result.info_update.model_dump(exclude_unset=True, exclude_none=True)
            new_info = new_info.model_copy(update=update_dict)

        # RAG Search if valid context found
        rag_context = state.get("rag_context", "")
        if self.enable_rag and state["messages"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                new_ctx = self.field_manager.search_context(last_msg.content)
                if new_ctx: rag_context = new_ctx

        new_collected_issues = list(collected_issues)
        
        # If complete (no missing) AND we have an issue, archive it
        missing_after = self.field_manager.get_missing_fields(new_info)
        if not missing_after and new_info.issue_text:
            new_collected_issues.append(new_info.model_dump())
            new_info = InterviewInfo()

        is_complete = state.get("is_complete", False) or result.interview_finished or result.early_exit

        return {
            "info": new_info,
            "next_goal": result.next_goal,
            "is_complete": is_complete,
            "rag_context": rag_context,
            "collected_issues": new_collected_issues
        }

    def generator_node(self, state: AgentState):
        """Node 3: Generate the final friendly response."""
        goal = state.get("next_goal", "Respond naturally")
        rag = state.get("rag_context", "")
        
        sys_prompt = f"""You are a gentle, empathetic AI interviewer for the Busan Walkability project.
        
YOUR STRICT GOAL FOR THIS TURN: 
[{goal}]

RULES:
1. MUST ONLY ACKNOWLEDGE THE USER'S PREVIOUS RESPONSE AND ACHIEVE THE STRICT GOAL ABOVE.
2. NEVER ASK MORE THAN ONE QUESTION IN A SINGLE RESPONSE.
3. BE EMPATHETIC AND CONVERSATIONAL (Mirroring).
4. PREVENT QUESTION BOMBING.
5. NEVER use raw database field names like 'location_bucket', 'primary_category', or 'severity_score' in your response.
6. Translate all questions into natural, conversational, and polite language (e.g. "어느 동네였나요?", "그때 얼마나 불편하셨나요? 0점에서 4점 사이로 표현해주신다면요?").
"""
        if rag:
            sys_prompt += f"\n[Available Background Info]\n{rag}\n(Use this naturally if it helps the goal without sounding robotic.)"

        messages = [SystemMessage(content=sys_prompt)] + state["messages"][-3:]
        # Invoke standard raw chat model to allow astream_events to intercept raw text chunks
        result = self.generator_llm.invoke(messages)

        return {
            "messages": [AIMessage(content=result.content)],
            "suggested_replies": [], # Left empty to minimize latency, can be populated async if needed
            "turn_index": state.get("turn_index", 0) + 1
        }

    # =========================================================================
    # LEGACY: TURN_BASED ARCHITECTURE
    # =========================================================================

    def _turn_based_node(self, state: AgentState):
        current_turn = state.get("turn_index", 0)
        current_info = state.get("info", InterviewInfo())
        full_system_prompt = self.turn_manager.build_system_prompt(
            current_turn, current_info, self.loader.language
        )

        messages = [SystemMessage(content=full_system_prompt)] + state["messages"]
        result: BotResponse = self.structured_llm.invoke(messages)

        new_info = current_info.model_copy()
        if result.info_update:
            update_dict = result.info_update.model_dump(exclude_unset=True, exclude_none=True)
            new_info = new_info.model_copy(update=update_dict)

        next_turn_val = result.next_turn if not result.early_exit else 99
        suggested = result.suggested_replies or self.turn_manager.get_suggested_replies(next_turn_val)

        return {
            "messages": [AIMessage(content=result.response)],
            "info": new_info,
            "turn_index": next_turn_val,
            "suggested_replies": suggested,
            "is_complete": result.early_exit or self.turn_manager.is_final_turn(next_turn_val)
        }

class BusanDesignGraph(ConfigurableInterviewGraph):
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key, topic_id="busan_walkability")

PrimaryCategory = Enum("PrimaryCategory", {
    "SAFETY": "안전 (Safety)",
    "ACCESSIBILITY": "접근성 (Accessibility)",
    "WAYFINDING": "길찾기 (Wayfinding)",
    "COMFORT": "쾌적성/미관 (Comfort)",
    "OTHER": "기타 (Other)"
})
