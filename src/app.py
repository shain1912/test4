import streamlit as st
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bot import BusanDesignGraph, InterviewInfo
from src.db import init_db, insert_interview, get_all_interviews
from src.analysis import SemanticAnalyzer
import plotly.express as px

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize DB
init_db()

st.set_page_config(page_title="부산 걷기 좋은 도시 - AI 인터뷰어", page_icon="🏙️", layout="wide")

st.title("🏙️ 부산 걷기 좋은 도시 만들기 Platform")

tab1, tab2 = st.tabs(["💬 인터뷰 (Chat)", "📊 분석 대시보드 (Dashboard)"])

# --- TAB 1: Chat ---
with tab1:
    st.header("시민 인터뷰 (AI Interview)")
    st.markdown("부산의 보행 환경에 대한 여러분의 소중한 경험을 들려주세요.")

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "topics_covered" not in st.session_state:
        st.session_state.topics_covered = []
    if "interview_info" not in st.session_state:
        st.session_state.interview_info = InterviewInfo()
    
    # Initialize Graph
    if "bot_graph" not in st.session_state:
        if api_key:
            st.session_state.bot_graph = BusanDesignGraph(api_key)
            # Add initial greeting
            greeting = "안녕하세요! 부산의 걷기 좋은 도시 만들기에 참여해주셔서 감사합니다. 지금 계신 곳은 어디인가요?"
            st.session_state.messages.append(AIMessage(content=greeting))
        else:
            st.error("API Key가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    # Display Chat History
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    # Chat Input
    if prompt := st.chat_input("답변을 입력해주세요..."):
        if "bot_graph" in st.session_state:
            # 1. Append User Message
            st.session_state.messages.append(HumanMessage(content=prompt))
            
            # 2. Invoke Graph (Show Spinner)
            with st.spinner("AI가 답변을 생성 중입니다..."):
                current_state = {
                    "messages": st.session_state.messages,
                    "info": st.session_state.interview_info,
                    "topics_covered": st.session_state.topics_covered
                }
                
                try:
                    result = st.session_state.bot_graph.graph.invoke(current_state)
                    
                    # 3. Update State
                    st.session_state.messages = result["messages"]
                    st.session_state.interview_info = result["info"]
                    st.session_state.topics_covered = result["topics_covered"]
                    
                    # 4. Rerun to refresh the chat history UI
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"오류 발생: {e}")

    st.markdown("---")
    if st.button("인터뷰 종료 및 저장 (Finish & Save)"):
        with st.spinner("데이터를 저장하고 있습니다..."):
            # Save to DB
            info_dict = st.session_state.interview_info.dict()
            insert_interview(info_dict)
            st.success("소중한 의견이 저장되었습니다! 대시보드에서 결과를 확인해보세요.")
            
            # Reset
            st.session_state.messages = []
            st.session_state.topics_covered = []
            st.session_state.interview_info = InterviewInfo()
            st.rerun()

# --- TAB 2: Dashboard ---
with tab2:
    st.header("실시간 데이터 분석 (Real-time Analysis)")
    
    df = pd.DataFrame(get_all_interviews())
    
    if df.empty:
        st.info("아직 수집된 데이터가 없습니다. 인터뷰 탭에서 의견을 남겨주세요!")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("총 인터뷰 수", len(df))
        col2.metric("최근 수집", df['timestamp'].iloc[0] if 'timestamp' in df.columns else "-")
        col3.metric("주요 이슈 유형", df['issue'].nunique() if 'issue' in df.columns else 0)

        st.divider()

        # Basic Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📍 지역별 분포")
            if 'location' in df.columns:
                st.bar_chart(df['location'].value_counts())
        with c2:
            st.subheader("🚧 해결책 제안 유형")
            if 'solution_type' in df.columns:
                st.bar_chart(df['solution_type'].value_counts())

        # Semantic Analysis Section
        st.divider()
        st.header("🧠 AI 의미 분석 (Semantic Cluster)")
        
        if st.button("심층 분석 실행 (Run Semantic Analysis)"):
            with st.spinner("AI가 데이터를 분석하여 3D 지도를 그리고 있습니다..."):
                analyzer = SemanticAnalyzer(api_key=api_key)
                # Compute 3D t-SNE
                result_df = analyzer.process_and_analyze(df, text_column='issue', n_dimensions=3)
                st.session_state['analysis_result'] = result_df
                st.success("분석 완료!")

        if 'analysis_result' in st.session_state:
            result_df = st.session_state['analysis_result']

            if 'z' in result_df.columns:
                tab_viz1, tab_viz2, tab_viz3 = st.tabs(["3D 의미 지도", "주제 계층 구조", "이슈 흐름도"])
                
                with tab_viz1:
                    st.markdown("#### 🌐 3D Semantic Space")
                    st.caption("마우스로 회전/확대/축소하여 군집을 확인하세요.")
                    fig_3d = px.scatter_3d(
                        result_df, 
                        x='x', y='y', z='z',
                        color='topic_label',
                        hover_data=['issue', 'location', 'solution_detail'],
                        title="시민 의견 3D 군집 지도",
                        template="plotly_dark",
                        height=600
                    )
                    fig_3d.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
                    st.plotly_chart(fig_3d, use_container_width=True)

                with tab_viz2:
                    st.markdown("#### ☀️ Topic Hierarchy (Sunburst)")
                    st.caption("주제(Topic) -> 해결책 유형(Solution) -> 구체적 장소(Location)의 비율을 보여줍니다.")
                    # Handle missing values for cleaner chart
                    clean_df = result_df.fillna("Unknown")
                    fig_sun = px.sunburst(
                        clean_df, 
                        path=['topic_label', 'solution_type', 'location'],
                        values='id', # Just count
                        title="주제별 계층 분포",
                        height=600
                    )
                    st.plotly_chart(fig_sun, use_container_width=True)

                with tab_viz3:
                    st.markdown("#### 🌊 Parallel Categories (Flow)")
                    st.caption("지역(Location)에서 발생한 문제가 어떤 가치(Value)와 연결되는지 보여줍니다.")
                    fig_flow = px.parallel_categories(
                        result_df,
                        dimensions=['location', 'topic_label', 'primary_value'],
                        title="지역 - 이슈 토픽 - 가치 연결 흐름",
                        height=500
                    )
                    st.plotly_chart(fig_flow, use_container_width=True)

            else:
                st.warning("3D 좌표 생성에 실패했습니다. 데이터가 너무 적을 수 있습니다.")
        
        # Raw Data
        with st.expander("전체 데이터 로그 보기"):
            st.dataframe(df)
