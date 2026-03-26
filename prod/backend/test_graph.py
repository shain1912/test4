import asyncio
from bot import ConfigurableInterviewGraph, InterviewInfo
from langchain_core.messages import AIMessage, HumanMessage
import os

os.environ["OPENAI_API_KEY"] = "sk-..." # Will load from env

async def test():
    try:
        graph = ConfigurableInterviewGraph(topic_id="busan_walkability_v2", enable_rag=False)
        state = {
            "messages": [AIMessage(content="안녕하세요!"), HumanMessage(content="길거리에 쓰레기가 많아서 불편해요.")],
            "info": InterviewInfo(),
            "collected_issues": [],
            "turn_index": 0,
            "is_complete": False,
        }
        print("Invoking graph...")
        result = await graph.graph.ainvoke(state)
        print("Result:", result)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
