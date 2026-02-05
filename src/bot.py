"""
Configurable Interview Bot using LangGraph.
Supports two interview modes:
1. turn_based: Fixed turn sequence (legacy)
2. field_based: Semi-structured interview based on required fields

Features:
- Multiple issues collection per session
- RAG-based contextual knowledge for informed interviewing
"""

import os
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field

from src.config_loader import (
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


class BotResponse(BaseModel):
    """The structured response from the interviewer bot."""
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
    rag_context: str  # RAG에서 검색된 컨텍스트


# --- Field-Based Interview Manager with RAG ---

class FieldBasedInterviewManager:
    """
    Manages semi-structured interviews based on required fields.
    Supports RAG-based contextual knowledge.
    """

    def __init__(self, topic_config: Dict[str, Any], language: str = "ko", enable_rag: bool = True):
        self.topic_config = topic_config
        self.language = language
        self.required_fields = topic_config.get("required_fields", [])
        self.system_prompt_config = topic_config.get("system_prompt", {})
        self.enable_rag = enable_rag
        self.knowledge_base = None

        # Initialize RAG if enabled
        if enable_rag:
            try:
                from src.knowledge_base import get_knowledge_base
                self.knowledge_base = get_knowledge_base()
            except Exception as e:
                print(f"RAG initialization failed: {e}")
                self.enable_rag = False

    def get_missing_fields(self, info: InterviewInfo) -> List[Dict[str, Any]]:
        """Get list of fields that haven't been collected yet."""
        info_dict = info.model_dump()
        missing = []
        for field in self.required_fields:
            field_id = field["id"]
            if info_dict.get(field_id) is None:
                missing.append(field)
        return missing

    def get_collected_fields(self, info: InterviewInfo) -> List[Dict[str, Any]]:
        """Get list of fields that have been collected."""
        info_dict = info.model_dump()
        collected = []
        for field in self.required_fields:
            field_id = field["id"]
            if info_dict.get(field_id) is not None:
                collected.append({
                    **field,
                    "value": info_dict[field_id]
                })
        return collected

    def is_issue_complete(self, info: InterviewInfo) -> bool:
        """Check if all required fields for current issue have been collected."""
        return len(self.get_missing_fields(info)) == 0

    def search_context(self, user_message: str) -> str:
        """
        사용자 메시지에서 위치/시설을 감지하고 관련 컨텍스트 검색
        """
        if not self.enable_rag or not self.knowledge_base:
            return ""

        try:
            context = self.knowledge_base.get_context_for_location(user_message)
            if context:
                rag_info = f"""
## 🔍 관련 배경 지식 (RAG)
**감지된 위치:** {context.location_name}

**관련 정보:**
{chr(10).join(['- ' + info for info in context.relevant_info])}

**제안되는 후속 질문 (자연스럽게 활용하세요):**
{chr(10).join(['- ' + probe for probe in context.suggested_probes])}

⚠️ 주의: 이 정보를 직접 언급하지 말고, 자연스럽게 대화에 녹여서 질문하세요.
예: "혹시 그 근처에서 경사 때문에 힘드셨던 적 있으세요?" (O)
    "제가 알기로 그곳은 경사가 심한데요" (X)
"""
                return rag_info
        except Exception as e:
            print(f"RAG search error: {e}")

        return ""

    def build_system_prompt(self, info: InterviewInfo, collected_issues: List[dict], rag_context: str = "") -> str:
        """Build dynamic system prompt with RAG context."""
        lang = self.language

        # Role
        role = self.system_prompt_config.get("role", {}).get(lang, "You are an interviewer.")

        # Interview style guidelines
        style = self.system_prompt_config.get("interview_style", {}).get(lang, "")

        # Extraction rules
        extraction = self.system_prompt_config.get("extraction_rules", {}).get(lang, "")

        # Multi-issue instructions
        multi_issue_instructions = """
## 여러 이슈 수집
- 사용자가 "다른 문제도 있어요", "또 하나는", "그리고" 등으로 새로운 이슈를 언급하면 `new_issue_started=True`로 설정하세요.
- 현재 이슈의 정보가 모두 수집되면 `current_issue_complete=True`로 설정하세요.
- 사용자가 "끝", "없어요", "이게 다예요" 등으로 더 이상 이슈가 없다고 하면 `interview_finished=True`로 설정하세요.
- 현재 이슈가 완료되면 자연스럽게 "다른 불편한 점은 없으셨나요?"라고 물어보세요.
""" if lang == "ko" else """
## Multiple Issues Collection
- If user mentions "another issue", "also", "and another thing", set `new_issue_started=True`.
- When current issue has all required info, set `current_issue_complete=True`.
- If user says "that's all", "no more", "finished", set `interview_finished=True`.
- After completing an issue, naturally ask "Were there any other issues?"
"""

        # Current state
        collected = self.get_collected_fields(info)
        missing = self.get_missing_fields(info)

        # Previously collected issues
        prev_issues_str = ""
        if collected_issues:
            prev_issues_str = f"\n## 이전에 수집된 이슈 ({len(collected_issues)}건)\n" if lang == "ko" else f"\n## Previously Collected Issues ({len(collected_issues)})\n"
            for i, issue in enumerate(collected_issues, 1):
                prev_issues_str += f"{i}. {issue.get('issue_text', 'N/A')[:50]}...\n"

        collected_str = ""
        if collected:
            collected_str = "\n## 현재 이슈 - 수집된 정보\n" if lang == "ko" else "\n## Current Issue - Collected\n"
            for f in collected:
                name = f.get("name", {}).get(lang, f["id"])
                collected_str += f"- {name}: {f['value']}\n"

        missing_str = ""
        if missing:
            missing_str = "\n## 현재 이슈 - 아직 필요한 정보\n" if lang == "ko" else "\n## Current Issue - Still Needed\n"
            for f in missing:
                name = f.get("name", {}).get(lang, f["id"])
                desc = f.get("description", {}).get(lang, "")
                missing_str += f"- {name}: {desc}\n"

                if f.get("type") == "scale":
                    scale = f.get("scale", {})
                    labels = scale.get("labels", {}).get(lang, [])
                    if labels:
                        missing_str += f"  (척도: {', '.join([f'{i}={l}' for i, l in enumerate(labels)])})\n"

                if f.get("type") == "category":
                    options = f.get("options", [])
                    opt_labels = [o.get("label", {}).get(lang, o["id"]) for o in options]
                    missing_str += f"  (옵션: {', '.join(opt_labels)})\n"
        else:
            missing_str = "\n## 현재 이슈 완료! 다른 이슈가 있는지 물어보세요.\n" if lang == "ko" else "\n## Current issue complete! Ask if there are other issues.\n"

        # Combine all parts including RAG context
        prompt_parts = [
            role,
            style,
            extraction,
            multi_issue_instructions,
            prev_issues_str,
            collected_str,
            missing_str
        ]

        # Add RAG context if available
        if rag_context:
            prompt_parts.insert(3, rag_context)  # Insert after extraction rules

        return "\n\n".join(filter(None, prompt_parts))


# --- Configurable Interview Graph ---

class ConfigurableInterviewGraph:
    """
    A configurable interview graph that loads settings from YAML files.
    Supports turn_based and field_based modes with RAG integration.
    """

    def __init__(self, api_key: str = None, topic_id: str = None, enable_rag: bool = True):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self.api_key = api_key

        # Load configuration
        self.loader = get_config_loader()
        self.topic_config = self.loader.load_topic_config(topic_id)

        # Determine interview mode
        self.interview_mode = self.topic_config.get("interview_mode", "turn_based")

        # Check if RAG is enabled in config
        self.enable_rag = enable_rag and self.topic_config.get("enable_rag", True)

        # Initialize appropriate manager
        if self.interview_mode == "field_based":
            self.field_manager = FieldBasedInterviewManager(
                self.topic_config,
                self.loader.language,
                enable_rag=self.enable_rag
            )
            self.turn_manager = None
        else:
            self.turn_manager = TurnManager(self.topic_config, self.loader.language)
            self.field_manager = None

        # Get LLM config
        llm_config = self.loader.llm_config
        self.llm = ChatOpenAI(
            model=llm_config.get("model", "gpt-4o"),
            temperature=llm_config.get("temperature", 0.3)
        )
        self.structured_llm = self.llm.with_structured_output(BotResponse)

        # Build Graph
        builder = StateGraph(AgentState)
        builder.add_node("interviewer", self.interviewer_node)
        builder.add_edge(START, "interviewer")
        builder.add_edge("interviewer", END)

        self.graph = builder.compile()

    def get_greeting(self) -> str:
        """Get the initial greeting message from config."""
        greeting = self.topic_config.get("greeting", {})
        return self.loader.get_localized(greeting)

    def get_closing(self) -> str:
        """Get the closing message from config."""
        closing = self.topic_config.get("closing", {})
        return self.loader.get_localized(closing)

    def get_topic_name(self) -> str:
        """Get the topic name from config."""
        meta = self.topic_config.get("meta", {})
        return self.loader.get_localized(meta, "name")

    def get_topic_description(self) -> str:
        """Get the topic description from config."""
        meta = self.topic_config.get("meta", {})
        return self.loader.get_localized(meta, "description")

    def interviewer_node(self, state: AgentState):
        """Process user input and generate bot response."""
        current_info = state.get("info", InterviewInfo())
        collected_issues = state.get("collected_issues", [])

        if self.interview_mode == "field_based":
            return self._field_based_interview(state, current_info, collected_issues)
        else:
            return self._turn_based_interview(state, current_info)

    def _field_based_interview(self, state: AgentState, current_info: InterviewInfo, collected_issues: List[dict]):
        """Handle field-based interview with RAG support."""
        # Get the last user message for RAG search
        rag_context = state.get("rag_context", "")

        # Search for relevant context from user's latest message
        if self.enable_rag and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                new_context = self.field_manager.search_context(last_message.content)
                if new_context:
                    rag_context = new_context  # Update with new context

        # Build dynamic system prompt with RAG context
        system_prompt = self.field_manager.build_system_prompt(
            current_info, collected_issues, rag_context
        )

        # Invoke LLM
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        result: BotResponse = self.structured_llm.invoke(messages)

        # Update current info
        new_info = current_info.model_copy()
        if result.info_update:
            update_dict = result.info_update.model_dump(exclude_unset=True, exclude_none=True)
            new_info = new_info.model_copy(update=update_dict)

        # Handle issue completion and new issue detection
        new_collected_issues = list(collected_issues)

        if (result.current_issue_complete or result.new_issue_started) and new_info.issue_text:
            new_collected_issues.append(new_info.model_dump())
            new_info = InterviewInfo()

        is_complete = result.interview_finished or result.early_exit

        return {
            "messages": [AIMessage(content=result.response)],
            "info": new_info,
            "collected_issues": new_collected_issues,
            "turn_index": state.get("turn_index", 0) + 1,
            "suggested_replies": result.suggested_replies,
            "is_complete": is_complete,
            "rag_context": rag_context
        }

    def _turn_based_interview(self, state: AgentState, current_info: InterviewInfo):
        """Handle turn-based (fixed sequence) interview."""
        current_turn = state.get("turn_index", 0)

        full_system_prompt = self.turn_manager.build_system_prompt(
            current_turn, current_info, self.loader.language
        )

        messages = [SystemMessage(content=full_system_prompt)] + state["messages"]
        result: BotResponse = self.structured_llm.invoke(messages)

        new_info = current_info.model_copy()
        if result.info_update:
            update_dict = result.info_update.model_dump(exclude_unset=True, exclude_none=True)
            new_info = new_info.model_copy(update=update_dict)

        next_turn_val = result.next_turn
        if result.early_exit:
            next_turn_val = 99

        suggested = result.suggested_replies
        if not suggested:
            suggested = self.turn_manager.get_suggested_replies(next_turn_val)

        return {
            "messages": [AIMessage(content=result.response)],
            "info": new_info,
            "collected_issues": state.get("collected_issues", []),
            "turn_index": next_turn_val,
            "suggested_replies": suggested,
            "is_complete": result.early_exit or self.turn_manager.is_final_turn(next_turn_val),
            "rag_context": ""
        }


# --- Backwards Compatibility ---

class BusanDesignGraph(ConfigurableInterviewGraph):
    """Backwards compatibility alias."""

    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key, topic_id="busan_walkability")


PrimaryCategory = Enum("PrimaryCategory", {
    "SAFETY": "안전 (Safety)",
    "ACCESSIBILITY": "접근성 (Accessibility)",
    "WAYFINDING": "길찾기 (Wayfinding)",
    "COMFORT": "쾌적성/미관 (Comfort)",
    "OTHER": "기타 (Other)"
})
