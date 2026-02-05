"""
Survey Configuration Generator
자연어 설명을 받아 YAML 설정 파일을 생성합니다.
"""

import os
import yaml
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class CategoryConfig(BaseModel):
    """카테고리 설정"""
    id: str
    label_ko: str
    label_en: str
    keywords: List[str] = Field(default_factory=list)


class FieldConfig(BaseModel):
    """필수 필드 설정"""
    id: str
    name_ko: str
    name_en: str
    description_ko: str
    description_en: str
    field_type: str = "text"  # text, scale, category
    scale_min: Optional[int] = None
    scale_max: Optional[int] = None
    scale_labels_ko: Optional[List[str]] = None
    scale_labels_en: Optional[List[str]] = None


class SurveyConfig(BaseModel):
    """전체 설문 설정"""
    topic_id: str = Field(description="영문 ID (예: customer_feedback)")
    name_ko: str = Field(description="한국어 설문 이름")
    name_en: str = Field(description="영어 설문 이름")
    description_ko: str = Field(description="한국어 설명")
    description_en: str = Field(description="영어 설명")
    greeting_ko: str = Field(description="한국어 인사말")
    greeting_en: str = Field(description="영어 인사말")
    categories: List[CategoryConfig] = Field(description="카테고리 목록")
    required_fields: List[FieldConfig] = Field(description="수집할 필드 목록")
    role_ko: str = Field(description="AI 역할 설명 (한국어)")
    role_en: str = Field(description="AI 역할 설명 (영어)")


GENERATOR_SYSTEM_PROMPT = """당신은 설문 조사 설정 파일을 생성하는 전문가입니다.
사용자가 자연어로 설명하는 설문 조사를 분석하여 구조화된 설정을 생성합니다.

## 규칙
1. topic_id는 영문 소문자와 언더스코어만 사용 (예: customer_feedback, employee_survey)
2. 모든 텍스트는 한국어(ko)와 영어(en) 두 버전을 제공
3. required_fields에는 반드시 수집해야 할 정보를 정의
4. categories는 응답을 분류할 카테고리들
5. 인사말(greeting)은 친근하고 전문적으로 작성

## 필드 타입
- text: 자유 텍스트 입력
- scale: 숫자 척도 (예: 1-5점, 0-10점)
- category: 선택형 카테고리

## 예시 필드들
- main_feedback: 주요 피드백/의견 (text)
- satisfaction_score: 만족도 점수 (scale, 1-5)
- category: 분류 카테고리 (category)
- location: 위치/장소 (text)
- severity: 심각도 (scale, 0-4)

사용자의 설명을 분석하여 적절한 설문 구조를 생성하세요.
"""


class SurveyConfigGenerator:
    """설문 설정 생성기"""

    def __init__(self, api_key: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.structured_llm = self.llm.with_structured_output(SurveyConfig)

    def generate_config(self, user_description: str) -> SurveyConfig:
        """
        자연어 설명을 받아 설문 설정을 생성합니다.

        Args:
            user_description: 사용자가 설명한 설문 내용

        Returns:
            SurveyConfig: 생성된 설문 설정
        """
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"다음 설문 조사를 위한 설정을 생성해주세요:\n\n{user_description}")
        ]

        result = self.structured_llm.invoke(messages)
        return result

    def config_to_yaml(self, config: SurveyConfig) -> str:
        """
        SurveyConfig를 YAML 문자열로 변환합니다.
        """
        # Build the YAML structure
        yaml_dict = {
            "meta": {
                "id": config.topic_id,
                "name": {
                    "ko": config.name_ko,
                    "en": config.name_en
                },
                "description": {
                    "ko": config.description_ko,
                    "en": config.description_en
                }
            },
            "interview_mode": "field_based",
            "categories": [],
            "required_fields": [],
            "system_prompt": {
                "role": {
                    "ko": config.role_ko,
                    "en": config.role_en
                },
                "interview_style": {
                    "ko": """## 인터뷰 스타일
- 자연스러운 대화처럼 진행하세요
- 응답자의 말을 경청하고, 그들이 언급한 내용을 따라가세요
- 모호하거나 흥미로운 답변에는 후속 질문(probing)을 하세요
- 필요한 정보가 자연스럽게 나오면 별도로 물어보지 않아도 됩니다""",
                    "en": """## Interview Style
- Conduct the interview like a natural conversation
- Listen actively and follow up on what the respondent mentions
- Use probing questions for vague or interesting responses
- If needed information comes up naturally, you don't need to ask separately"""
                },
                "extraction_rules": {
                    "ko": """## 정보 추출 규칙
- 응답에서 자연스럽게 언급된 정보를 추출하세요
- 모든 필수 정보가 수집되면 인터뷰를 마무리하세요""",
                    "en": """## Information Extraction Rules
- Extract information naturally mentioned in responses
- Wrap up the interview once all required information is collected"""
                }
            },
            "greeting": {
                "ko": config.greeting_ko,
                "en": config.greeting_en
            },
            "closing": {
                "ko": "소중한 의견을 나눠주셔서 정말 감사합니다. 말씀해주신 내용은 큰 도움이 될 것입니다.",
                "en": "Thank you so much for sharing your valuable feedback. Your input will be very helpful."
            }
        }

        # Add categories
        for cat in config.categories:
            yaml_dict["categories"].append({
                "id": cat.id,
                "label": {
                    "ko": cat.label_ko,
                    "en": cat.label_en
                },
                "keywords": cat.keywords
            })

        # Add required fields
        for field in config.required_fields:
            field_dict = {
                "id": field.id,
                "name": {
                    "ko": field.name_ko,
                    "en": field.name_en
                },
                "description": {
                    "ko": field.description_ko,
                    "en": field.description_en
                }
            }

            if field.field_type == "scale":
                field_dict["type"] = "scale"
                field_dict["scale"] = {
                    "min": field.scale_min or 0,
                    "max": field.scale_max or 5,
                    "labels": {
                        "ko": field.scale_labels_ko or [],
                        "en": field.scale_labels_en or []
                    }
                }
            elif field.field_type == "category":
                field_dict["type"] = "category"
                # Categories will reference the main categories list

            yaml_dict["required_fields"].append(field_dict)

        # Convert to YAML string
        return yaml.dump(yaml_dict, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def save_config(self, config: SurveyConfig, configs_dir: str = None) -> str:
        """
        설정을 YAML 파일로 저장합니다.

        Args:
            config: 저장할 설문 설정
            configs_dir: 설정 파일 디렉토리 (기본값: configs/topics/)

        Returns:
            저장된 파일 경로
        """
        if configs_dir is None:
            from src.config_loader import get_configs_dir
            configs_dir = get_configs_dir() / "topics"

        yaml_content = self.config_to_yaml(config)
        file_path = os.path.join(configs_dir, f"{config.topic_id}.yaml")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        return file_path

    def update_main_config(self, topic_id: str, configs_dir: str = None):
        """
        메인 설정 파일의 active_topic을 업데이트합니다.
        """
        if configs_dir is None:
            from src.config_loader import get_configs_dir
            configs_dir = get_configs_dir()

        config_path = os.path.join(configs_dir, "config.yaml")

        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)

        main_config['active_topic'] = topic_id

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(main_config, f, allow_unicode=True, default_flow_style=False)
