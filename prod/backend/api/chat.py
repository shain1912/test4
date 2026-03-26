import os
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from bot import ConfigurableInterviewGraph, InterviewInfo
from core.db import supabase
from core.config import settings

router = APIRouter()

# In-memory session storage (For production, consider Redis or Supabase)
sessions: Dict[str, dict] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_complete: bool
    suggested_replies: List[str]
    collected_issues_count: int

def get_or_create_session(session_id: Optional[str]) -> str:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())[:8]
        # Initialize graph instance for this session
        graph = ConfigurableInterviewGraph(api_key=settings.OPENAI_API_KEY)
        greeting = graph.get_greeting()
        
        sessions[session_id] = {
            "messages": [AIMessage(content=greeting)],
            "info": InterviewInfo(),
            "collected_issues": [],
            "turn_index": 0,
            "is_complete": False,
            "bot_graph": graph,
            "greeting": greeting
        }
    return session_id

@router.post("/")
async def chat(request: ChatRequest):
    session_id = get_or_create_session(request.session_id)
    state = sessions[session_id]
    
    if state["is_complete"]:
        # If complete, we just return a fast JSON string chunk simulating a stream or normal.
        async def mock_stream():
            yield f'data: {json.dumps({"type": "chunk", "text": "인터뷰가 이미 종료되었습니다."})}\n\n'
            yield f'data: {json.dumps({"type": "complete", "is_complete": True, "collected_issues_count": len(state["collected_issues"])})}\n\n'
        return StreamingResponse(mock_stream(), media_type="text/event-stream")
        
    state["messages"].append(HumanMessage(content=request.message))
    
    current_state = {
        "messages": state["messages"],
        "info": state["info"],
        "collected_issues": state["collected_issues"],
        "turn_index": state.get("turn_index", 0),
        "is_complete": state.get("is_complete", False)
    }
    
    async def chat_stream_generator():
        try:
            graph = state["bot_graph"]
            
            # Stream events from LangGraph
            async for event in graph.graph.astream_events(current_state, version="v2"):
                # Intercept the raw generator model tokens
                if event["event"] == "on_chat_model_stream" and "generator_node_llm" in event.get("tags", []):
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        yield f'data: {json.dumps({"type": "chunk", "text": chunk})}\n\n'
                
                # Intercept the final state when the graph finishes
                elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                    final_state = event["data"]["output"]
                    state["messages"] = final_state["messages"]
                    state["info"] = final_state["info"]
                    state["collected_issues"] = final_state.get("collected_issues", [])
                    state["turn_index"] = final_state.get("turn_index", 0)
                    state["is_complete"] = final_state.get("is_complete", False)
                    
                    # If complete, save to Supabase
                    if state["is_complete"]:
                        all_issues = list(state["collected_issues"])
                        current_info = state["info"].model_dump()
                        if current_info.get("issue_text"):
                            all_issues.append(current_info)
                            
                        if supabase and all_issues:
                            for issue in all_issues:
                                data = {
                                    "session_id": session_id,
                                    "issue_text": issue.get('issue_text', ''),
                                    "severity_score": issue.get('severity_score', 0),
                                    "primary_category": issue.get('primary_category', ''),
                                    "location_bucket": issue.get('location_bucket', ''),
                                    "evidence_span": issue.get('evidence_span', ''),
                                    "raw_log": issue
                                }
                                supabase.table("interviews").insert(data).execute()
                                
                    yield f'data: {json.dumps({"type": "complete", "is_complete": state["is_complete"], "collected_issues_count": len(state["collected_issues"])})}\n\n'

        except Exception as e:
            yield f'data: {json.dumps({"type": "error", "error": str(e)})}\n\n'

    return StreamingResponse(chat_stream_generator(), media_type="text/event-stream")

@router.get("/start")
async def start_chat():
    """Initializes a new chat and returns the greeting"""
    session_id = get_or_create_session(None)
    state = sessions[session_id]
    return ChatResponse(
        response=state["greeting"],
        session_id=session_id,
        is_complete=False,
        suggested_replies=[],
        collected_issues_count=0
    )
