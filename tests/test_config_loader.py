"""Tests for the configuration loader module."""

import pytest
from pathlib import Path
from src.config_loader import (
    ConfigLoader,
    TurnManager,
    DynamicSchemaBuilder,
    get_config_loader,
    reset_config_loader,
    get_project_root,
    get_configs_dir
)


@pytest.fixture
def loader():
    """Create a fresh ConfigLoader for testing."""
    reset_config_loader()
    return get_config_loader()


def test_get_project_root():
    """Test that project root is correctly identified."""
    root = get_project_root()
    assert root.exists()
    assert (root / "src").exists()


def test_get_configs_dir():
    """Test that configs directory is correctly identified."""
    configs = get_configs_dir()
    assert configs.exists()
    assert (configs / "config.yaml").exists()


def test_config_loader_initialization(loader):
    """Test ConfigLoader basic initialization."""
    assert loader is not None
    assert loader.config_dir.exists()


def test_main_config_loading(loader):
    """Test loading main configuration."""
    config = loader.main_config
    assert config is not None
    assert "active_topic" in config
    assert "language" in config
    assert "llm" in config


def test_active_topic(loader):
    """Test getting active topic."""
    topic = loader.active_topic
    assert topic == "busan_walkability"


def test_language(loader):
    """Test getting language setting."""
    lang = loader.language
    assert lang in ["ko", "en"]


def test_llm_config(loader):
    """Test getting LLM configuration."""
    llm = loader.llm_config
    assert "model" in llm
    assert "temperature" in llm


def test_load_topic_config(loader):
    """Test loading topic configuration."""
    topic = loader.load_topic_config("busan_walkability")
    assert topic is not None
    assert "meta" in topic
    assert "categories" in topic
    assert "turns" in topic
    assert "system_prompt" in topic
    assert "greeting" in topic


def test_load_topic_config_default(loader):
    """Test loading default (active) topic configuration."""
    topic = loader.load_topic_config()
    assert topic is not None
    # Should load the active topic (busan_walkability)
    assert topic["meta"]["id"] == "busan_walkability"


def test_load_topic_config_not_found(loader):
    """Test loading non-existent topic raises error."""
    with pytest.raises(FileNotFoundError):
        loader.load_topic_config("nonexistent_topic")


def test_load_ui_strings(loader):
    """Test loading UI strings."""
    ui = loader.load_ui_strings("ko")
    assert ui is not None
    assert "page" in ui
    assert "tabs" in ui
    assert "interview" in ui
    assert "dashboard" in ui
    assert "cli" in ui


def test_load_ui_strings_fallback(loader):
    """Test UI strings fallback to Korean."""
    # Load non-existent language should fallback
    ui = loader.load_ui_strings("fr")  # French doesn't exist
    assert ui is not None
    # Should have loaded Korean as fallback
    assert "page" in ui


def test_get_localized(loader):
    """Test getting localized strings."""
    topic = loader.load_topic_config()
    name = loader.get_localized(topic["meta"], "name")
    assert name is not None
    # Default language is Korean
    assert "부산" in name


def test_config_caching(loader):
    """Test that configs are cached."""
    # Load topic config twice
    topic1 = loader.load_topic_config()
    topic2 = loader.load_topic_config()
    # Should be the same cached object
    assert topic1 is topic2


def test_config_reload(loader):
    """Test reloading configurations."""
    topic1 = loader.load_topic_config()
    loader.reload()
    topic2 = loader.load_topic_config()
    # After reload, should be different objects with same content
    assert topic1 is not topic2
    assert topic1["meta"]["id"] == topic2["meta"]["id"]


class TestTurnManager:
    """Tests for TurnManager class."""

    @pytest.fixture
    def turn_manager(self, loader):
        """Create a TurnManager instance."""
        topic_config = loader.load_topic_config()
        return TurnManager(topic_config, "ko")

    def test_turn_manager_initialization(self, turn_manager):
        """Test TurnManager initialization."""
        assert turn_manager is not None
        assert len(turn_manager.turns) > 0

    def test_get_turn(self, turn_manager):
        """Test getting turn by ID."""
        turn = turn_manager.get_turn(0)
        assert turn is not None
        assert turn["name"] == "greeting"

    def test_get_turn_not_found(self, turn_manager):
        """Test getting non-existent turn."""
        turn = turn_manager.get_turn(999)
        assert turn is None

    def test_get_prompt_template(self, turn_manager):
        """Test getting prompt template."""
        prompt = turn_manager.get_prompt_template(0)
        assert prompt is not None
        assert len(prompt) > 0

    def test_get_suggested_replies(self, turn_manager):
        """Test getting suggested replies."""
        replies = turn_manager.get_suggested_replies(1)  # severity turn
        assert replies is not None
        assert len(replies) == 5  # 0-4 severity levels

    def test_get_suggested_replies_empty(self, turn_manager):
        """Test getting suggested replies for turn with none."""
        replies = turn_manager.get_suggested_replies(0)  # greeting has none
        assert replies == []

    def test_get_next_turn(self, turn_manager):
        """Test getting next turn ID."""
        next_turn = turn_manager.get_next_turn(0)
        assert next_turn == 1

    def test_is_final_turn(self, turn_manager):
        """Test checking if turn is final."""
        assert not turn_manager.is_final_turn(0)
        assert turn_manager.is_final_turn(4)

    def test_build_system_prompt(self, turn_manager):
        """Test building system prompt."""
        # Use a dict instead of InterviewInfo to avoid importing bot.py
        info = {"issue_text": "보행 문제", "severity_score": None, "primary_category": None}
        prompt = turn_manager.build_system_prompt(1, info, "ko")
        assert prompt is not None
        assert "공공디자인" in prompt or "연구원" in prompt
        assert "보행 문제" in prompt


class TestDynamicSchemaBuilder:
    """Tests for DynamicSchemaBuilder class."""

    def test_build_category_enum(self):
        """Test building category enum from config."""
        categories = [
            {"id": "safety", "label": {"ko": "안전", "en": "Safety"}},
            {"id": "accessibility", "label": {"ko": "접근성", "en": "Accessibility"}},
        ]
        enum_class = DynamicSchemaBuilder.build_category_enum(categories, "ko")
        assert enum_class is not None
        assert hasattr(enum_class, "SAFETY")
        assert hasattr(enum_class, "ACCESSIBILITY")

    def test_get_category_values(self):
        """Test getting category values."""
        categories = [
            {"id": "safety", "label": {"ko": "안전", "en": "Safety"}},
            {"id": "accessibility", "label": {"ko": "접근성", "en": "Accessibility"}},
        ]
        values = DynamicSchemaBuilder.get_category_values(categories, "ko")
        assert "안전" in values
        assert "접근성" in values


def test_singleton_behavior():
    """Test that get_config_loader returns singleton."""
    reset_config_loader()
    loader1 = get_config_loader()
    loader2 = get_config_loader()
    assert loader1 is loader2


def test_reset_config_loader():
    """Test that reset_config_loader clears the singleton."""
    loader1 = get_config_loader()
    reset_config_loader()
    loader2 = get_config_loader()
    assert loader1 is not loader2
