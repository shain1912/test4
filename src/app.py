"""
Streamlit application for the AI Interview Platform.
Supports collecting multiple issues and natural language survey configuration.
"""

import streamlit as st
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bot import ConfigurableInterviewGraph, InterviewInfo
from src.db import insert_interview, insert_multiple_issues, get_all_interviews, generate_session_id
from src.analysis import SemanticAnalyzer
from src.config_loader import get_config_loader, get_configs_dir, reset_config_loader
from src.survey_generator import SurveyConfigGenerator
from src.knowledge_base import KnowledgeBase, LocationInfo, get_knowledge_base, reset_knowledge_base
import plotly.express as px
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load configuration
loader = get_config_loader()
ui = loader.load_ui_strings()
topic_config = loader.load_topic_config()
topic_meta = topic_config.get("meta", {})
interview_mode = topic_config.get("interview_mode", "turn_based")

# Page configuration
page_title = f"{loader.get_localized(topic_meta, 'name')} - {ui['page']['title']}"
st.set_page_config(page_title=page_title, page_icon=ui['page']['icon'], layout="wide")

st.title(f"{ui['page']['icon']} {loader.get_localized(topic_meta, 'name')} Platform")

tab1, tab2, tab3, tab4 = st.tabs([
    ui['tabs']['interview'],
    ui['tabs']['dashboard'],
    "⚙️ 설문 설정",
    "📚 지식 관리 (RAG)"
])

# --- TAB 1: Chat ---
with tab1:
    st.header(ui['interview']['header'])
    topic_description = loader.get_localized(topic_meta, 'description')
    st.markdown(topic_description)

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "interview_info" not in st.session_state:
        st.session_state.interview_info = InterviewInfo()
    if "collected_issues" not in st.session_state:
        st.session_state.collected_issues = []
    if "is_complete" not in st.session_state:
        st.session_state.is_complete = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()

    # Initialize Graph
    if "bot_graph" not in st.session_state:
        if api_key:
            st.session_state.bot_graph = ConfigurableInterviewGraph(api_key=api_key)
            greeting = st.session_state.bot_graph.get_greeting()
            st.session_state.messages.append(AIMessage(content=greeting))
        else:
            st.error(ui['interview']['api_key_error'])

    # Sidebar: Show collected issues (field_based mode)
    if interview_mode == "field_based":
        with st.sidebar:
            st.subheader("📋 수집된 이슈")

            if st.session_state.collected_issues:
                for i, issue in enumerate(st.session_state.collected_issues, 1):
                    with st.expander(f"✅ 이슈 {i}: {issue.get('issue_text', '')[:30]}...", expanded=False):
                        st.write(f"**내용:** {issue.get('issue_text', '-')}")
                        st.write(f"**심각도:** {issue.get('severity_score', '-')}")
                        st.write(f"**카테고리:** {issue.get('primary_category', '-')}")
                        st.write(f"**위치:** {issue.get('location_bucket', '-')}")

            st.divider()
            st.subheader("🔄 현재 이슈")
            info = st.session_state.interview_info
            info_dict = info.model_dump()

            required_fields = topic_config.get("required_fields", [])
            for field in required_fields:
                field_id = field["id"]
                field_name = field.get("name", {}).get(loader.language, field_id)
                value = info_dict.get(field_id)

                if value is not None:
                    st.success(f"✅ {field_name}")
                else:
                    st.warning(f"⏳ {field_name}")

            st.metric("총 수집된 이슈", f"{len(st.session_state.collected_issues)}건")

            if st.session_state.is_complete:
                st.balloons()
                st.info("🎉 인터뷰가 완료되었습니다!")

    # Chat Interface
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    def process_input(user_text):
        st.session_state.messages.append(HumanMessage(content=user_text))

        with st.spinner(ui['interview']['ai_spinner']):
            current_state = {
                "messages": st.session_state.messages,
                "info": st.session_state.interview_info,
                "collected_issues": st.session_state.collected_issues,
                "turn_index": st.session_state.get("turn_index", 0),
                "is_complete": st.session_state.get("is_complete", False)
            }

            try:
                result = st.session_state.bot_graph.graph.invoke(current_state)
                st.session_state.messages = result["messages"]
                st.session_state.interview_info = result["info"]
                st.session_state.collected_issues = result.get("collected_issues", [])
                st.session_state.turn_index = result.get("turn_index", 0)
                st.session_state.suggested_replies = result.get("suggested_replies", [])
                st.session_state.is_complete = result.get("is_complete", False)
                st.rerun()
            except Exception as e:
                st.error(ui['errors']['general'].format(error=e))

    if st.session_state.is_complete:
        total = len(st.session_state.collected_issues)
        st.success(f"✅ 인터뷰가 완료되었습니다! 총 {total}개의 이슈가 수집되었습니다.")

    if not st.session_state.is_complete:
        if "suggested_replies" in st.session_state and st.session_state.suggested_replies:
            st.markdown(ui['interview']['suggested_replies_label'])
            cols = st.columns(len(st.session_state.suggested_replies))
            for idx, reply in enumerate(st.session_state.suggested_replies):
                if cols[idx].button(reply, key=f"btn_{len(st.session_state.messages)}_{idx}"):
                    process_input(reply)

    if not st.session_state.is_complete:
        if prompt := st.chat_input(ui['interview']['input_placeholder']):
            process_input(prompt)

    st.markdown("---")

    button_type = "primary" if st.session_state.is_complete else "secondary"
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button(ui['interview']['finish_button'], type=button_type):
            with st.spinner(ui['interview']['saving_spinner']):
                all_issues = list(st.session_state.collected_issues)
                current_info = st.session_state.interview_info.model_dump()
                if current_info.get("issue_text"):
                    all_issues.append(current_info)

                if all_issues:
                    session_id = insert_multiple_issues(all_issues, st.session_state.session_id)
                    st.success(f"✅ {len(all_issues)}개의 이슈가 저장되었습니다!")
                else:
                    st.warning("저장할 이슈가 없습니다.")

                st.session_state.messages = []
                st.session_state.interview_info = InterviewInfo()
                st.session_state.collected_issues = []
                st.session_state.is_complete = False
                st.session_state.session_id = generate_session_id()
                for key in ["turn_index", "suggested_replies", "bot_graph"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    with col2:
        total = len(st.session_state.collected_issues)
        if st.session_state.interview_info.issue_text:
            total += 1
        st.metric("이슈 수", f"{total}건")

# --- TAB 2: Dashboard ---
with tab2:
    st.header(ui['dashboard']['header'])

    try:
        data = get_all_interviews()
        df = pd.DataFrame(data)
    except Exception as e:
        st.error(ui['dashboard']['data_load_error'].format(error=e))
        st.stop()

    if df.empty:
        st.info(ui['dashboard']['no_data'])
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(ui['dashboard']['total_interviews'], len(df))

        if 'session_id' in df.columns:
            unique_sessions = df['session_id'].nunique()
            col2.metric("총 세션 수", unique_sessions)
        else:
            col2.metric(ui['dashboard']['recent_collection'], df['timestamp'].iloc[0] if 'timestamp' in df.columns else "-")

        avg_severity = "-"
        if 'severity_score' in df.columns:
            val = pd.to_numeric(df['severity_score'], errors='coerce').mean()
            if not pd.isna(val):
                avg_severity = f"{val:.1f}/4.0"
        col3.metric(ui['dashboard']['avg_severity'], avg_severity)

        if 'session_id' in df.columns:
            issues_per_session = len(df) / max(df['session_id'].nunique(), 1)
            col4.metric("세션당 평균 이슈", f"{issues_per_session:.1f}건")

        st.divider()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader(ui['dashboard']['location_distribution'])
            if 'location_bucket' in df.columns:
                location_counts = df['location_bucket'].value_counts()
                if not location_counts.empty:
                    st.bar_chart(location_counts)
        with col_c2:
            st.subheader(ui['dashboard']['category_distribution'])
            if 'primary_category' in df.columns:
                category_counts = df['primary_category'].value_counts()
                if not category_counts.empty:
                    st.bar_chart(category_counts)

        st.divider()
        st.header(ui['dashboard']['semantic_analysis_header'])

        if st.button(ui['dashboard']['run_analysis_button']):
            with st.spinner(ui['dashboard']['analysis_spinner']):
                analyzer = SemanticAnalyzer(api_key=api_key)
                result_df = analyzer.process_and_analyze(df, text_column='issue_text', n_dimensions=3)
                st.session_state['analysis_result'] = result_df
                st.success(ui['dashboard']['analysis_complete'])

        if 'analysis_result' in st.session_state:
            result_df = st.session_state['analysis_result']

            if 'x' in result_df.columns:
                st.markdown(ui['dashboard']['semantic_space_title'])
                fig_3d = px.scatter_3d(
                    result_df, x='x', y='y', z='z', color='topic_label',
                    hover_data=['issue_text', 'location_bucket', 'severity_score'],
                    title=ui['dashboard']['chart_title'], template="plotly_dark", height=600
                )
                fig_3d.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
                fig_3d.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig_3d, use_container_width=True)

                st.divider()
                st.header(ui['dashboard']['topic_list_header'])

                unique_labels = sorted(result_df['topic_label'].unique())
                cols = st.columns(2)

                for idx, label in enumerate(unique_labels):
                    with cols[idx % 2]:
                        cluster_data = result_df[result_df['topic_label'] == label]
                        count = len(cluster_data)
                        avg_sev = cluster_data['severity_score'].mean() if 'severity_score' in cluster_data.columns else 0

                        with st.container(border=True):
                            st.subheader(f"{label}")
                            m1, m2 = st.columns(2)
                            m1.metric(ui['dashboard']['opinion_count'], f"{count}{ui['dashboard']['opinion_count_unit']}")
                            m2.metric(ui['dashboard']['avg_severity_label'], f"{avg_sev:.1f}")

                            sample_issues = cluster_data['issue_text'].dropna().sample(min(2, count)).tolist()
                            for issue in sample_issues:
                                st.caption(f"- {issue}")

                st.bar_chart(result_df['topic_label'].value_counts())
            else:
                st.warning(ui['dashboard']['analysis_insufficient'])

        st.divider()
        with st.expander(ui['dashboard']['full_data_log']):
            st.dataframe(df)

# --- TAB 3: Survey Configuration ---
with tab3:
    st.header("⚙️ 설문 설정 생성기")
    st.markdown("""
    **자연어로 설문을 설명하면 AI가 설정 파일을 생성합니다.**

    예시:
    - "카페 고객 만족도 조사를 하고 싶어. 음료 맛, 서비스, 가격에 대한 의견을 수집하고 1-5점 만족도도 받고 싶어."
    - "회사 직원들의 근무 환경에 대한 피드백을 수집하고 싶어. 불편한 점, 개선 요청사항, 부서 정보를 받고 싶어."
    """)

    # Show current active topic
    st.info(f"📌 현재 활성 설문: **{loader.get_localized(topic_meta, 'name')}** (`{loader.active_topic}`)")

    # List existing topics
    topics_dir = get_configs_dir() / "topics"
    existing_topics = [f.stem for f in topics_dir.glob("*.yaml")]

    with st.expander("📁 기존 설문 목록"):
        for topic in existing_topics:
            is_active = "✅" if topic == loader.active_topic else ""
            st.write(f"{is_active} `{topic}`")

    st.divider()

    # Survey description input
    st.subheader("1️⃣ 설문 설명 입력")
    survey_description = st.text_area(
        "어떤 설문을 만들고 싶은지 자유롭게 설명해주세요:",
        height=150,
        placeholder="예: 대학교 학생들의 캠퍼스 생활 만족도를 조사하고 싶어. 시설, 수업, 교통 편의성에 대한 의견을 수집하고, 학년과 전공도 알고 싶어."
    )

    # Generate button
    if st.button("🤖 AI로 설문 설정 생성", type="primary", disabled=not survey_description):
        if not api_key:
            st.error("API 키가 설정되지 않았습니다.")
        else:
            with st.spinner("AI가 설문 설정을 생성하고 있습니다..."):
                try:
                    generator = SurveyConfigGenerator(api_key=api_key)
                    config = generator.generate_config(survey_description)
                    st.session_state['generated_config'] = config
                    st.session_state['generated_yaml'] = generator.config_to_yaml(config)
                    st.success("✅ 설문 설정이 생성되었습니다!")
                except Exception as e:
                    st.error(f"생성 중 오류 발생: {e}")

    # Show generated config
    if 'generated_config' in st.session_state:
        st.divider()
        st.subheader("2️⃣ 생성된 설문 설정 확인")

        config = st.session_state['generated_config']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**설문 ID:** `{config.topic_id}`")
            st.markdown(f"**설문 이름:** {config.name_ko}")
        with col2:
            st.markdown(f"**카테고리 수:** {len(config.categories)}개")
            st.markdown(f"**수집 필드 수:** {len(config.required_fields)}개")

        # Preview tabs
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📋 요약", "📝 YAML 미리보기", "✏️ 수정"])

        with preview_tab1:
            st.markdown("#### 인사말")
            st.info(config.greeting_ko)

            st.markdown("#### 카테고리")
            for cat in config.categories:
                st.write(f"- **{cat.label_ko}** (`{cat.id}`)")

            st.markdown("#### 수집 필드")
            for field in config.required_fields:
                field_type_label = {"text": "텍스트", "scale": "척도", "category": "선택"}.get(field.field_type, field.field_type)
                st.write(f"- **{field.name_ko}** ({field_type_label}): {field.description_ko}")

        with preview_tab2:
            st.code(st.session_state['generated_yaml'], language='yaml')

        with preview_tab3:
            st.markdown("**설문 ID 수정:**")
            new_topic_id = st.text_input("Topic ID", value=config.topic_id, key="edit_topic_id")

            st.markdown("**설문 이름 수정:**")
            new_name_ko = st.text_input("이름 (한국어)", value=config.name_ko, key="edit_name_ko")

            st.markdown("**인사말 수정:**")
            new_greeting = st.text_area("인사말", value=config.greeting_ko, key="edit_greeting")

            if st.button("수정 사항 적용"):
                config.topic_id = new_topic_id
                config.name_ko = new_name_ko
                config.greeting_ko = new_greeting
                st.session_state['generated_config'] = config
                generator = SurveyConfigGenerator(api_key=api_key)
                st.session_state['generated_yaml'] = generator.config_to_yaml(config)
                st.success("수정 사항이 적용되었습니다.")
                st.rerun()

        st.divider()
        st.subheader("3️⃣ 저장 및 적용")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 설정 파일 저장", type="primary"):
                try:
                    generator = SurveyConfigGenerator(api_key=api_key)
                    file_path = generator.save_config(config)
                    st.success(f"✅ 저장 완료: `{file_path}`")
                except Exception as e:
                    st.error(f"저장 중 오류: {e}")

        with col2:
            if st.button("🚀 저장하고 바로 적용"):
                try:
                    generator = SurveyConfigGenerator(api_key=api_key)
                    file_path = generator.save_config(config)
                    generator.update_main_config(config.topic_id)

                    # Reset config loader to reload
                    reset_config_loader()

                    # Clear interview state
                    for key in list(st.session_state.keys()):
                        if key not in ['generated_config', 'generated_yaml']:
                            del st.session_state[key]

                    st.success(f"✅ 저장 및 적용 완료! 새 설문: **{config.name_ko}**")
                    st.info("페이지를 새로고침하면 새 설문이 적용됩니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"오류: {e}")

    # Quick switch existing topic
    st.divider()
    st.subheader("🔄 기존 설문으로 전환")

    selected_topic = st.selectbox(
        "활성화할 설문 선택:",
        options=existing_topics,
        index=existing_topics.index(loader.active_topic) if loader.active_topic in existing_topics else 0
    )

    if st.button("이 설문으로 전환"):
        try:
            generator = SurveyConfigGenerator(api_key=api_key)
            generator.update_main_config(selected_topic)
            reset_config_loader()

            for key in list(st.session_state.keys()):
                del st.session_state[key]

            st.success(f"✅ `{selected_topic}` 설문으로 전환되었습니다!")
            st.rerun()
        except Exception as e:
            st.error(f"오류: {e}")

# --- TAB 4: Knowledge Management (RAG) ---
with tab4:
    st.header("📚 지식 베이스 관리")
    st.markdown("""
    **RAG (Retrieval Augmented Generation) 지식 관리**

    인터뷰어 AI가 위치/시설에 대한 배경 지식을 가지고 더 스마트한 질문을 할 수 있도록 정보를 등록합니다.
    예: "부산대학교는 경사가 심하다" → AI가 경사 관련 후속 질문을 유도
    """)

    # Initialize knowledge base
    if 'knowledge_base' not in st.session_state:
        try:
            st.session_state.knowledge_base = get_knowledge_base(api_key)
        except Exception as e:
            st.error(f"지식 베이스 초기화 실패: {e}")
            st.stop()

    kb = st.session_state.knowledge_base

    # Sub-tabs for knowledge management
    kb_tab1, kb_tab2, kb_tab3, kb_tab4 = st.tabs([
        "📋 등록된 지식",
        "➕ 새 항목 추가",
        "📁 파일 업로드",
        "🔍 검색 테스트"
    ])

    # --- KB Tab 1: View existing knowledge ---
    with kb_tab1:
        st.subheader("등록된 위치/시설 정보")

        if not kb.knowledge_data:
            st.info("등록된 지식이 없습니다. 샘플 데이터를 로드하거나 새 항목을 추가하세요.")
            if st.button("🔄 샘플 데이터 로드 (부산 지역)"):
                from src.knowledge_base import create_sample_knowledge_base
                st.session_state.knowledge_base = create_sample_knowledge_base(api_key)
                st.session_state.knowledge_base.save()
                st.success("샘플 데이터가 로드되었습니다!")
                st.rerun()
        else:
            # Display knowledge entries
            for name, info in kb.knowledge_data.items():
                with st.expander(f"📍 {name} ({info.type} - {info.region})", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**특성:**")
                        for char in info.characteristics:
                            st.write(f"- {char}")

                        if info.demographics:
                            st.markdown(f"**주 이용자:** {info.demographics}")

                    with col2:
                        if info.known_issues:
                            st.markdown("**알려진 문제:**")
                            for issue in info.known_issues:
                                st.write(f"- ⚠️ {issue}")

                        if info.additional_info:
                            st.markdown(f"**추가 정보:** {info.additional_info}")

                    # Delete button
                    if st.button(f"🗑️ 삭제", key=f"delete_{name}"):
                        del kb.knowledge_data[name]
                        kb.save()
                        st.success(f"'{name}' 항목이 삭제되었습니다.")
                        st.rerun()

            st.divider()
            st.metric("총 등록 항목", f"{len(kb.knowledge_data)}개")

            # Save button
            if st.button("💾 변경사항 저장"):
                kb.save()
                st.success("저장되었습니다!")

    # --- KB Tab 2: Add new entry ---
    with kb_tab2:
        st.subheader("새 위치/시설 정보 추가")

        with st.form("add_knowledge_form"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("위치/시설 이름 *", placeholder="예: 서면역")
                location_type = st.selectbox(
                    "유형 *",
                    ["지하철역", "대학교", "관광지", "상권", "공원", "병원", "교통시설", "주거지역", "기타"]
                )
                region = st.text_input("지역구 *", placeholder="예: 부산진구")

            with col2:
                characteristics = st.text_area(
                    "주요 특성 (쉼표로 구분) *",
                    placeholder="예: 유동인구 많음, 상업지구, 교통 허브"
                )
                known_issues = st.text_area(
                    "알려진 문제점 (쉼표로 구분)",
                    placeholder="예: 혼잡한 보행로, 불법 주정차"
                )

            demographics = st.text_input("주 이용자층", placeholder="예: 직장인, 쇼핑객")
            additional_info = st.text_area("추가 정보", placeholder="예: 1호선 2호선 환승역")

            submitted = st.form_submit_button("➕ 추가", type="primary")

            if submitted:
                if not name or not location_type or not region or not characteristics:
                    st.error("필수 항목(*)을 모두 입력해주세요.")
                else:
                    try:
                        new_info = LocationInfo(
                            name=name,
                            type=location_type,
                            region=region,
                            characteristics=[c.strip() for c in characteristics.split(",") if c.strip()],
                            known_issues=[i.strip() for i in known_issues.split(",") if i.strip()] if known_issues else [],
                            demographics=demographics if demographics else None,
                            additional_info=additional_info if additional_info else None
                        )
                        kb.add_location(new_info)
                        kb.save()
                        st.success(f"✅ '{name}' 항목이 추가되었습니다!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"추가 실패: {e}")

    # --- KB Tab 3: File upload ---
    with kb_tab3:
        st.subheader("파일에서 지식 로드")

        st.markdown("""
        **지원 형식:**
        - **JSON**: 아래 형식의 배열
        - **CSV**: 컬럼명이 동일한 CSV 파일

        ```json
        [
          {
            "name": "부산역",
            "type": "교통시설",
            "region": "동구",
            "characteristics": ["KTX 정차역", "대중교통 허브"],
            "known_issues": ["복잡한 동선", "노숙자 문제"],
            "demographics": "여행객, 출장자",
            "additional_info": "부산의 관문"
          }
        ]
        ```
        """)

        uploaded_file = st.file_uploader(
            "JSON 또는 CSV 파일 업로드",
            type=["json", "csv"]
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    st.write(f"**로드된 항목 수:** {len(data)}개")

                    # Preview
                    with st.expander("미리보기"):
                        for item in data[:3]:
                            st.json(item)
                        if len(data) > 3:
                            st.write(f"... 외 {len(data) - 3}개")

                    if st.button("📥 이 데이터 가져오기", key="import_json"):
                        for item in data:
                            info = LocationInfo(**item)
                            kb.add_location(info)
                        kb.save()
                        st.success(f"✅ {len(data)}개 항목을 가져왔습니다!")
                        st.rerun()

                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    st.write(f"**로드된 행 수:** {len(df)}개")
                    st.dataframe(df.head())

                    if st.button("📥 이 데이터 가져오기", key="import_csv"):
                        # Save temporarily and use kb's csv loader
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                            df.to_csv(f.name, index=False)
                            kb.load_from_csv(f.name)
                        kb.save()
                        st.success(f"✅ {len(df)}개 항목을 가져왔습니다!")
                        st.rerun()

            except Exception as e:
                st.error(f"파일 로드 실패: {e}")

        st.divider()

        # Export current data
        st.subheader("현재 데이터 내보내기")

        if kb.knowledge_data:
            export_data = [info.model_dump() for info in kb.knowledge_data.values()]
            export_json = json.dumps(export_data, ensure_ascii=False, indent=2)

            st.download_button(
                "📤 JSON으로 내보내기",
                data=export_json,
                file_name="knowledge_base_export.json",
                mime="application/json"
            )

    # --- KB Tab 4: Search test ---
    with kb_tab4:
        st.subheader("RAG 검색 테스트")
        st.markdown("""
        사용자가 특정 위치를 언급했을 때 AI가 어떤 컨텍스트를 받게 되는지 테스트합니다.
        """)

        test_query = st.text_input(
            "테스트 쿼리",
            placeholder="예: 부산대학교 근처에서 걸을 때 불편했어요"
        )

        if st.button("🔍 검색", disabled=not test_query):
            if not kb.vector_store:
                st.warning("벡터 스토어가 초기화되지 않았습니다. 먼저 지식을 추가하세요.")
            else:
                with st.spinner("검색 중..."):
                    try:
                        # Direct similarity search
                        docs = kb.search(test_query, k=3)

                        st.markdown("### 📄 유사도 검색 결과")
                        if docs:
                            for i, doc in enumerate(docs, 1):
                                with st.expander(f"결과 {i}: {doc.metadata.get('name', 'Unknown')}"):
                                    st.text(doc.page_content)
                        else:
                            st.info("검색 결과가 없습니다.")

                        # Context generation
                        st.markdown("### 🤖 AI에게 전달될 컨텍스트")
                        context = kb.get_context_for_location(test_query)

                        if context:
                            st.success(f"**감지된 위치:** {context.location_name}")

                            st.markdown("**관련 정보:**")
                            for info in context.relevant_info:
                                st.text(info)

                            st.markdown("**제안되는 후속 질문:**")
                            for probe in context.suggested_probes:
                                st.write(f"- 💡 {probe}")
                        else:
                            st.info("관련 컨텍스트를 찾지 못했습니다.")

                    except Exception as e:
                        st.error(f"검색 오류: {e}")

        st.divider()

        # RAG status
        st.subheader("📊 RAG 상태")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("등록된 위치", f"{len(kb.knowledge_data)}개")
        with col2:
            vector_status = "✅ 활성" if kb.vector_store else "❌ 비활성"
            st.metric("벡터 스토어", vector_status)
        with col3:
            rag_enabled = topic_config.get("enable_rag", True)
            rag_status = "✅ 활성" if rag_enabled else "❌ 비활성"
            st.metric("RAG 모드", rag_status)

        # Reset knowledge base
        st.divider()
        if st.button("🔄 지식 베이스 초기화", type="secondary"):
            reset_knowledge_base()
            if 'knowledge_base' in st.session_state:
                del st.session_state['knowledge_base']
            st.success("지식 베이스가 초기화되었습니다.")
            st.rerun()
