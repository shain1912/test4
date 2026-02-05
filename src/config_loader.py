"""
Configuration loader for the interview chatbot.
Provides centralized access to YAML configuration files.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache
import yaml
from enum import Enum
from pydantic import BaseModel, Field, create_model


# --- Path Resolution ---

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return get_project_root() / "configs"


# --- YAML Loading ---

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# --- Config Loader Class ---

class ConfigLoader:
    """
    Central configuration loader that manages all YAML configs.
    Implements caching and provides easy access to topic and UI configs.
    """

    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or get_configs_dir()
        self._main_config: Optional[Dict[str, Any]] = None
        self._topic_configs: Dict[str, Dict[str, Any]] = {}
        self._ui_configs: Dict[str, Dict[str, Any]] = {}

    @property
    def main_config(self) -> Dict[str, Any]:
        """Load and cache the main configuration."""
        if self._main_config is None:
            self._main_config = load_yaml(self.config_dir / "config.yaml")
        return self._main_config

    @property
    def active_topic(self) -> str:
        """Get the currently active topic ID."""
        return self.main_config.get("active_topic", "busan_walkability")

    @property
    def language(self) -> str:
        """Get the current language setting."""
        return self.main_config.get("language", "ko")

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.main_config.get("llm", {"model": "gpt-4o", "temperature": 0.3})

    def load_topic_config(self, topic_id: str = None) -> Dict[str, Any]:
        """Load a topic configuration by ID."""
        topic_id = topic_id or self.active_topic

        if topic_id not in self._topic_configs:
            topic_file = self.config_dir / "topics" / f"{topic_id}.yaml"
            if not topic_file.exists():
                raise FileNotFoundError(f"Topic config not found: {topic_file}")
            self._topic_configs[topic_id] = load_yaml(topic_file)

        return self._topic_configs[topic_id]

    def load_ui_strings(self, language: str = None) -> Dict[str, Any]:
        """Load UI strings for a specific language."""
        language = language or self.language

        if language not in self._ui_configs:
            ui_file = self.config_dir / "ui" / f"{language}.yaml"
            if not ui_file.exists():
                # Fallback to Korean
                ui_file = self.config_dir / "ui" / "ko.yaml"
            self._ui_configs[language] = load_yaml(ui_file)

        return self._ui_configs[language]

    def get_localized(self, obj: Dict[str, Any], key: str = None) -> str:
        """
        Get a localized string from an object with language keys.
        If key is provided, first access that key then get the language.
        """
        if key:
            obj = obj.get(key, {})

        if isinstance(obj, dict):
            return obj.get(self.language, obj.get("ko", obj.get("en", str(obj))))
        return str(obj)

    def reload(self):
        """Clear all cached configs and reload."""
        self._main_config = None
        self._topic_configs.clear()
        self._ui_configs.clear()


# --- Turn Manager Class ---

class TurnManager:
    """
    Manages interview turn progression based on topic configuration.
    """

    def __init__(self, topic_config: Dict[str, Any], language: str = "ko"):
        self.topic_config = topic_config
        self.language = language
        self.turns = {t["id"]: t for t in topic_config.get("turns", [])}
        self.max_turn = max(self.turns.keys()) if self.turns else 0

    def get_turn(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """Get turn configuration by ID."""
        return self.turns.get(turn_id)

    def get_prompt_template(self, turn_id: int) -> str:
        """Get the prompt template for a specific turn."""
        turn = self.get_turn(turn_id)
        if not turn:
            # Return closing template if turn not found
            final_turn = self.get_final_turn()
            if final_turn:
                return final_turn.get("prompt_template", {}).get(self.language, "")
            return ""

        return turn.get("prompt_template", {}).get(self.language, "")

    def get_suggested_replies(self, turn_id: int) -> List[str]:
        """Get suggested replies for a turn (localized labels)."""
        turn = self.get_turn(turn_id)
        if not turn:
            return []

        replies = turn.get("suggested_replies", [])
        return [r.get("label", {}).get(self.language, r.get("value", "")) for r in replies]

    def get_next_turn(self, turn_id: int) -> int:
        """Get the next turn ID."""
        turn = self.get_turn(turn_id)
        if not turn:
            return 99  # End state
        return turn.get("next_turn", turn_id + 1)

    def is_final_turn(self, turn_id: int) -> bool:
        """Check if this is the final turn."""
        turn = self.get_turn(turn_id)
        return turn.get("is_final", False) if turn else True

    def get_final_turn(self) -> Optional[Dict[str, Any]]:
        """Get the final (closing) turn configuration."""
        for turn in self.turns.values():
            if turn.get("is_final", False):
                return turn
        return None

    def get_extracts(self, turn_id: int) -> List[str]:
        """Get the fields to extract for a turn."""
        turn = self.get_turn(turn_id)
        if not turn:
            return []
        return turn.get("extracts", [])

    def build_system_prompt(self, turn_id: int, info: Any, language: str = None) -> str:
        """Build the complete system prompt for a turn."""
        language = language or self.language
        system_config = self.topic_config.get("system_prompt", {})

        # Role
        role = system_config.get("role", {}).get(language, "You are an AI interviewer.")

        # Instructions
        instructions = system_config.get("instructions", {}).get(language, "")

        # Turn-specific prompt
        turn_prompt = self.get_prompt_template(turn_id)

        # Fill in dynamic values from info
        if hasattr(info, 'dict'):
            info_dict = info.dict()
        elif isinstance(info, dict):
            info_dict = info
        else:
            info_dict = {}

        # Safe format with defaults
        format_dict = {
            "issue_text": info_dict.get("issue_text") or "General Issue",
            "severity_score": info_dict.get("severity_score") if info_dict.get("severity_score") is not None else "?",
            "primary_category": info_dict.get("primary_category") or "Unknown",
            "location_bucket": info_dict.get("location_bucket") or "Unknown"
        }

        try:
            turn_prompt = turn_prompt.format(**format_dict)
        except KeyError:
            pass  # Keep original if formatting fails

        return f"{role}\n{instructions}\n\nCurrent Turn: {turn_id}\n{turn_prompt}"


# --- Dynamic Schema Builder ---

class DynamicSchemaBuilder:
    """
    Builds Pydantic models dynamically based on topic configuration.
    """

    @staticmethod
    def build_category_enum(categories: List[Dict[str, Any]], language: str = "ko") -> type:
        """Create a dynamic Enum for categories."""
        enum_dict = {}
        for cat in categories:
            cat_id = cat["id"].upper()
            label = cat.get("label", {}).get(language, cat["id"])
            full_label = f"{label} ({cat['id'].capitalize()})"
            enum_dict[cat_id] = full_label

        return Enum("PrimaryCategory", enum_dict)

    @staticmethod
    def get_category_values(categories: List[Dict[str, Any]], language: str = "ko") -> List[str]:
        """Get list of category values for validation."""
        return [cat.get("label", {}).get(language, cat["id"]) for cat in categories]


# --- Global Singleton ---

_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global ConfigLoader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def reset_config_loader():
    """Reset the global ConfigLoader (useful for testing)."""
    global _config_loader
    _config_loader = None
