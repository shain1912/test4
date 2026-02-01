import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from bot import BusanDesignBot

@pytest.fixture
def mock_openai():
    with patch('bot.OpenAI') as mock:
        yield mock

def test_chat_interaction(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Mocking the chat completion response
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "안녕하세요! 부산디자인봇입니다."
    mock_completion.choices = [mock_choice]
    
    mock_client.chat.completions.create.return_value = mock_completion
    
    bot = BusanDesignBot(api_key="test_key")
    response = bot.chat("안녕하세요")
    
    assert response == "안녕하세요! 부산디자인봇입니다."
    
    # Check if OpenAI client was called with correct messages
    call_args = mock_client.chat.completions.create.call_args
    assert call_args is not None
    messages = call_args.kwargs['messages']
    
    # First message should be system prompt
    assert messages[0]['role'] == 'system'
    assert "전문 질적 연구원" in messages[0]['content']
    
    # Second message should be user input
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == "안녕하세요"

def test_chat_history_memory(mock_openai):
    # Verify that conversation history is maintained
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # First response
    mock_completion1 = MagicMock()
    mock_choice1 = MagicMock()
    mock_choice1.message.content = "반갑습니다. 어디에 계신가요?"
    mock_completion1.choices = [mock_choice1]
    
    # Second response
    mock_completion2 = MagicMock()
    mock_choice2 = MagicMock()
    mock_choice2.message.content = "영도구 봉래동이군요."
    mock_completion2.choices = [mock_choice2]
    
    mock_client.chat.completions.create.side_effect = [mock_completion1, mock_completion2]
    
    bot = BusanDesignBot(api_key="test_key")
    
    # First turn
    resp1 = bot.chat("안녕하세요")
    assert resp1 == "반갑습니다. 어디에 계신가요?"
    
    # Second turn
    resp2 = bot.chat("영도구 봉래동입니다.")
    assert resp2 == "영도구 봉래동이군요."
    
    # Check history in second call
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs['messages']
    
    assert len(messages) == 5 # System + User1 + Assistant1 + User2
    assert messages[2]['role'] == 'assistant'
    assert messages[2]['content'] == "반갑습니다. 어디에 계신가요?"
