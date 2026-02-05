# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Configurable AI Interview Platform - An AI-powered interview system that uses LangGraph-driven conversations to collect qualitative feedback. Interview topics, prompts, categories, and UI strings are defined in YAML configuration files, enabling easy customization without code changes.

**Current Topic**: Busan Walkable City (부산 걷기 좋은 도시 만들기) - Collecting citizen feedback about Busan's walking environment.

## Commands

### Setup
```bash
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
# Create .env with OPENAI_API_KEY=sk-...
```

### Run
```bash
streamlit run src/app.py          # Main app (Chat + Dashboard)
python src/main.py                # CLI version
streamlit run src/dashboard.py    # Dashboard only
```

### Test
```bash
pytest                            # All tests
pytest tests/test_bot.py -v       # Single test file
pytest tests/test_config_loader.py -v  # Config loader tests
```

### Seed Test Data
```bash
python src/seed_rich.py           # Generate ~230 sample interviews
```

## Architecture

```
configs/
├── config.yaml              # Main settings (active_topic, language, llm)
├── topics/
│   └── busan_walkability.yaml   # Topic-specific: categories, turns, prompts
└── ui/
    ├── ko.yaml              # Korean UI strings
    └── en.yaml              # English UI strings

src/
├── config_loader.py         # ConfigLoader, TurnManager, schema builders
├── bot.py                   # ConfigurableInterviewGraph (LangGraph)
├── app.py                   # Streamlit UI
├── main.py                  # CLI interface
├── db.py                    # SQLite operations
└── analysis.py              # Semantic analysis (embeddings, t-SNE)

Streamlit UI (app.py)
    ├── Chat Tab → ConfigurableInterviewGraph (bot.py) → SQLite (db.py)
    └── Dashboard Tab → SemanticAnalyzer (analysis.py) → 3D Plotly
                              ↓
                        OpenAI API (GPT-4o, embeddings)
```

### Core Components

- **config_loader.py**: Configuration infrastructure
  - `ConfigLoader`: Loads and caches YAML configs
  - `TurnManager`: Manages turn progression and prompt building
  - `DynamicSchemaBuilder`: Creates Pydantic models from config

- **bot.py**:
  - `ConfigurableInterviewGraph`: Config-driven LangGraph interview flow
  - `BusanDesignGraph`: Backwards-compatible alias

- **db.py**: SQLite operations for `interviews` table
- **analysis.py**: Semantic pipeline (embeddings → K-Means → t-SNE → topic labeling)
- **app.py**: Streamlit UI with session state management

### Interview Flow (LangGraph)
Turn 0 (greeting) → Turn 1 (severity 0-4) → Turn 2 (category) → Turn 3 (location) → Turn 4 (closing)

Turn progression is defined in `configs/topics/<topic>.yaml`.

### Data Models
- `InterviewInfo` (Pydantic): Structured interview data
- `BotResponse` (Pydantic): LLM response with suggested replies
- `AgentState` (TypedDict): LangGraph state

## Configuration System

### Adding a New Topic

1. Create `configs/topics/new_topic.yaml`:
```yaml
meta:
  id: "new_topic"
  name: { ko: "새 주제", en: "New Topic" }
  description: { ko: "설명", en: "Description" }

categories:
  - id: "cat1"
    label: { ko: "카테고리1", en: "Category 1" }
  # ...

turns:
  - id: 0
    name: "greeting"
    goal: { ko: "인사", en: "Greet user" }
    prompt_template: { ko: "...", en: "..." }
    next_turn: 1
  # ... more turns

system_prompt:
  role: { ko: "AI 역할", en: "AI role" }
  instructions: { ko: "지시사항", en: "Instructions" }

greeting: { ko: "초기 인사", en: "Initial greeting" }
```

2. Update `configs/config.yaml`:
```yaml
active_topic: "new_topic"
```

3. Run the app - no code changes needed!

### Changing Language

Edit `configs/config.yaml`:
```yaml
language: "en"  # or "ko"
```

## Key Implementation Details

- All LLM calls use `with_structured_output()` with Pydantic models for validation
- t-SNE perplexity is dynamically calculated based on dataset size (5-30 range)
- Configuration is loaded once and cached; use `loader.reload()` to refresh
- `init_db()` is intentionally commented out in app.py to prevent data loss on reload
- `BusanDesignGraph` is a backwards-compatible alias for `ConfigurableInterviewGraph`
