import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.bot import BusanDesignGraph, InterviewInfo, BotResponse

@pytest.fixture
def mock_langchain():
    with patch('src.bot.ChatOpenAI') as MockChat:
        # Create a mock instance
        mock_instance = MockChat.return_value
        
        # Determine strict structure for BotResponse
        def side_effect(messages):
            # Inspect the last message to decide response
            last_msg = messages[-1]
            content = last_msg.content if isinstance(last_msg, HumanMessage) else ""
            
            if "부산" in content:
                # Mock response for location
                return BotResponse(
                    response="어디신가요?",
                    current_topic="ask_location",
                    info_update=InterviewInfo(location="Busan")
                )
            else:
                return BotResponse(
                    response="불편한 점은?",
                    current_topic="ask_issue",
                    info_update=InterviewInfo(issue="Noise")
                )

        # Mock the structured_llm.invoke method
        mock_instance.with_structured_output.return_value.invoke.side_effect = side_effect
        yield MockChat

def test_graph_initialization(mock_langchain):
    bot = BusanDesignGraph(api_key="fake-key")
    assert bot.graph is not None

def test_graph_flow(mock_langchain):
    bot = BusanDesignGraph(api_key="fake-key")
    
    # 1. User Input
    initial_state = {
        "messages": [HumanMessage(content="부산에 살아요")],
        "info": InterviewInfo(),
        "topics_covered": []
    }
    
    # 2. Invoke Graph
    result = bot.graph.invoke(initial_state)
    
    # 3. Verify assertions
    assert len(result["messages"]) == 2  # Human + AI
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].content == "어디신가요?"  # From mock
    
    # 4. Verify State Update
    assert result["info"].location == "Busan"
    assert "ask_location" in result["topics_covered"]
