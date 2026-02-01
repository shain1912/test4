import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from bot import BusanDesignBot

@pytest.fixture
def mock_openai():
    with patch('bot.OpenAI') as mock:
        yield mock

def test_save_log(mock_openai, tmp_path):
    bot = BusanDesignBot(api_key="test")
    bot.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}
    ]
    
    log_file = tmp_path / "chat_log.json"
    bot.save_log(str(log_file))
    
    # Check if file exists
    assert log_file.exists()
    
    # Check content
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 3
        assert data[1]['content'] == "hi"
