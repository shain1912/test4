import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.bot import BusanDesignGraph, ConfigurableInterviewGraph, InterviewInfo, BotResponse


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
                # Mock response for issue collection
                return BotResponse(
                    response="심각도는 어느 정도인가요?",
                    suggested_replies=["0 (별로)", "1 (조금)", "2 (보통)", "3 (심각)", "4 (매우 심각)"],
                    info_update=InterviewInfo(issue_text="부산 보행 문제"),
                    next_turn=1,
                    early_exit=False
                )
            else:
                return BotResponse(
                    response="불편한 점은 무엇인가요?",
                    suggested_replies=[],
                    info_update=None,
                    next_turn=0,
                    early_exit=False
                )

        # Mock the structured_llm.invoke method
        mock_instance.with_structured_output.return_value.invoke.side_effect = side_effect
        yield MockChat


def test_graph_initialization(mock_langchain):
    bot = BusanDesignGraph(api_key="fake-key")
    assert bot.graph is not None


def test_configurable_graph_initialization(mock_langchain):
    bot = ConfigurableInterviewGraph(api_key="fake-key")
    assert bot.graph is not None
    assert bot.topic_config is not None
    assert bot.turn_manager is not None


def test_graph_flow(mock_langchain):
    bot = BusanDesignGraph(api_key="fake-key")

    # 1. User Input
    initial_state = {
        "messages": [HumanMessage(content="부산에서 걸을 때 불편해요")],
        "info": InterviewInfo(),
        "turn_index": 0,
        "suggested_replies": []
    }

    # 2. Invoke Graph
    result = bot.graph.invoke(initial_state)

    # 3. Verify assertions
    assert len(result["messages"]) == 2  # Human + AI
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].content == "심각도는 어느 정도인가요?"  # From mock

    # 4. Verify State Update
    assert result["info"].issue_text == "부산 보행 문제"
    assert result["turn_index"] == 1


def test_get_greeting(mock_langchain):
    bot = BusanDesignGraph(api_key="fake-key")
    greeting = bot.get_greeting()
    # Greeting should come from config
    assert greeting is not None
    assert len(greeting) > 0


def test_get_topic_name(mock_langchain):
    bot = BusanDesignGraph(api_key="fake-key")
    topic_name = bot.get_topic_name()
    assert "부산" in topic_name or "Busan" in topic_name


def test_early_exit(mock_langchain):
    """Test that early_exit flag properly terminates the interview."""
    with patch('src.bot.ChatOpenAI') as MockChat:
        mock_instance = MockChat.return_value

        # Mock early exit response
        mock_instance.with_structured_output.return_value.invoke.return_value = BotResponse(
            response="알겠습니다. 인터뷰를 종료하겠습니다.",
            suggested_replies=[],
            info_update=None,
            next_turn=4,
            early_exit=True
        )

        bot = BusanDesignGraph(api_key="fake-key")

        initial_state = {
            "messages": [HumanMessage(content="그만")],
            "info": InterviewInfo(),
            "turn_index": 1,
            "suggested_replies": []
        }

        result = bot.graph.invoke(initial_state)

        # Should jump to end state (99)
        assert result["turn_index"] == 99
