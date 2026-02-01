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
# init_db() # Disabled to prevent wiping data on reload

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
            st.session_state.bot_graph = BusanDesignGraph(api_key=api_key)
            # Add initial greeting
            greeting = "안녕하세요! 부산의 걷기 좋은 도시 만들기에 참여해주셔서 감사합니다. 지금 계신 곳은 어디인가요?"
            st.session_state.messages.append(AIMessage(content=greeting))
        else:
            st.error("API Key가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    # --- Chat Interface ---
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Helper to process user input (text or button)
    def process_input(user_text):
        st.session_state.messages.append(HumanMessage(content=user_text))
        
        with st.spinner("AI가 응답 생성 중..."):
            current_state = {
                "messages": st.session_state.messages,
                "info": st.session_state.interview_info,
                "turn_index": st.session_state.get("turn_index", 0)
            }
            
            try:
                result = st.session_state.bot_graph.graph.invoke(current_state)
                
                # Update Session State
                st.session_state.messages = result["messages"]
                st.session_state.interview_info = result["info"]
                st.session_state.turn_index = result["turn_index"]
                st.session_state.suggested_replies = result.get("suggested_replies", [])
                
                st.rerun()
            except Exception as e:
                st.error(f"오류: {e}")

    # 1. Show Buttons if available
    if "suggested_replies" in st.session_state and st.session_state.suggested_replies:
        st.markdown("##### 답변 선택하기:")
        cols = st.columns(len(st.session_state.suggested_replies))
        for idx, reply in enumerate(st.session_state.suggested_replies):
            if cols[idx].button(reply, key=f"btn_{len(st.session_state.messages)}_{idx}"):
                process_input(reply)

    # 2. Chat Input (Always available for fallback or open-ended)
    if prompt := st.chat_input("답변을 입력해주세요..."):
        process_input(prompt)

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
    
    # 1. Load Data
    try:
        from src.db import get_all_interviews
        data = get_all_interviews()
        df = pd.DataFrame(data)
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.stop()
    
    # 2. Check Empty
    if df.empty:
        st.info("아직 수집된 데이터가 없습니다. 인터뷰 탭에서 의견을 남겨주세요!")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("총 인터뷰 수", len(df))
        col2.metric("최근 수집", df['timestamp'].iloc[0] if 'timestamp' in df.columns else "-")
        
        avg_severity = "-"
        if 'severity_score' in df.columns:
            val = pd.to_numeric(df['severity_score'], errors='coerce').mean()
            if not pd.isna(val):
                avg_severity = f"{val:.1f}/4.0"
        col3.metric("평균 심각도", avg_severity)

        st.divider()

        # Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("🏙️ 지역별 분포")
            if 'location_bucket' in df.columns:
                st.bar_chart(df['location_bucket'].value_counts())
        with col_c2:
            st.subheader("🚨 카테고리별 분포")
            if 'primary_category' in df.columns:
                st.bar_chart(df['primary_category'].value_counts())

        # Semantic Analysis Section
        st.divider()
        st.header("🧠 AI 의미 분석 (Semantic Cluster)")
        
        if st.button("심층 분석 실행 (Run Semantic Analysis)"):
            with st.spinner("AI가 데이터를 분석하여 3D 지도를 그리고 있습니다..."):
                analyzer = SemanticAnalyzer(api_key=api_key)
                
                # Compute 3D t-SNE using 'issue_text' (Requesting 3 dimensions explicitly)
                result_df = analyzer.process_and_analyze(df, text_column='issue_text', n_dimensions=3)
                
                # Store result in session state
                st.session_state['analysis_result'] = result_df
                st.success("분석 완료!")

        if 'analysis_result' in st.session_state:
            result_df = st.session_state['analysis_result']

            if 'x' in result_df.columns:
                # 1. 3D Chart
                st.markdown("#### 🌐 3D Semantic Space")
                fig_3d = px.scatter_3d(
                    result_df, 
                    x='x', y='y', z='z',
                    color='topic_label',
                    hover_data=['issue_text', 'location_bucket', 'severity_score'],
                    title="시민 의견 3D 군집 지도",
                    template="plotly_dark",
                    height=600
                )
                fig_3d.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
                fig_3d.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig_3d, use_container_width=True)

                # 2. Cluster Detail Cards
                st.divider()
                st.header("📑 상세 토픽 리스트 (Topic List)")
                st.info("💡 위 3D 지도에 표시된 색상별 토픽의 상세 내용입니다.")
                
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
                            m1.metric("의견 수", f"{count}건")
                            m2.metric("평균 심각도", f"{avg_sev:.1f}")
                            
                            st.markdown("**주요 키워드 & 예시:**")
                            sample_issues = cluster_data['issue_text'].sample(min(2, count)).tolist()
                            for issue in sample_issues:
                                st.caption(f"- {issue}")

                st.divider()
                st.subheader("📊 주제별 데이터 분포")
                st.bar_chart(result_df['topic_label'].value_counts())
            else:
                st.warning("분석 결과가 충분하지 않습니다.")

        st.divider()
        with st.expander("전체 데이터 로그 보기"):
            st.dataframe(df)
