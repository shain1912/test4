import streamlit as st
import pandas as pd
import sqlite3
import os
import sys

# Ensure proper path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import get_all_interviews, DB_PATH
from src.analysis import SemanticAnalyzer
import plotly.express as px

st.set_page_config(page_title="부산 걷기 좋은 도시 - 인터뷰 보드", layout="wide")

st.title("🚶 부산 걷기 좋은 도시 만들기 - 시민 인터뷰 현황")
st.markdown("시민들과의 1:1 채팅 인터뷰를 통해 수집된 보행 환경 데이터입니다.")

# 1. Load Data
try:
    data = get_all_interviews()
    df = pd.DataFrame(data)
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# 2. Check Empty
if df.empty:
    st.warning("아직 수집된 데이터가 없습니다.")
    st.stop()

# 3. Metrics
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

# 4. Charts
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.subheader("🏙️ 지역별 분포")
    if 'location_bucket' in df.columns:
        loc_counts = df['location_bucket'].value_counts()
        st.bar_chart(loc_counts)
    else:
        st.info("지역 데이터 없음")

with col_c2:
    st.subheader("💡 카테고리별 분포")
    if 'primary_category' in df.columns:
        cat_counts = df['primary_category'].value_counts()
        st.bar_chart(cat_counts)
    else:
        st.info("카테고리 데이터 없음")

st.divider()

# 5. Data Table
st.subheader("📋 상세 인터뷰 로그")

# Display key columns safely
available_cols = df.columns.tolist()
target_cols = ['id', 'timestamp', 'location_bucket', 'primary_category', 'issue_text', 'severity_score']
display_cols = [c for c in target_cols if c in available_cols]

st.dataframe(df[display_cols], use_container_width=True)

# Expander for full details
with st.expander("상세 데이터 보기 (JSON)"):
    selected_id = st.number_input("ID 입력", min_value=1, max_value=len(df), step=1)
    if 'id' in df.columns:
        row = df[df['id'] == selected_id]
        if not row.empty:
            st.json(row.to_dict(orient='records')[0])

st.divider()

# 6. Semantic Analysis (Optional)
st.header("🧠 AI 의미 분석 (Semantic Analysis)")
st.markdown("데이터의 의미를 분석하여 자동으로 토픽을 분류하고 시각화합니다.")

if st.button("분석 실행 (Analyze)"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("API Key not found in environment.")
    else:
        with st.spinner("AI 분석 중..."):
            try:
                analyzer = SemanticAnalyzer(api_key=api_key)
                # Use 'issue_text' column for clustering
                target_text_col = 'issue_text' if 'issue_text' in df.columns else 'issue'
                
                result_df = analyzer.process_and_analyze(df, text_column=target_text_col, n_dimensions=3)
                
                if 'x' in result_df.columns:
                    st.success("분석 완료!")
                    
                    fig = px.scatter(
                        result_df, 
                        x='x', 
                        y='y', 
                        color='topic_label',
                        hover_data=[target_text_col, 'location_bucket', 'severity_score'],
                        title="이슈 의미 연결망 (Semantic Network)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                if 'x' in result_df.columns:
                    st.success("분석 완료!")
                    
                    # 1. 3D/2D Chart
                    fig = px.scatter_3d(
                        result_df, 
                        x='x', y='y', z='z',
                        color='topic_label',
                        hover_data=[target_text_col, 'location_bucket', 'severity_score'],
                        title="이슈 의미 연결망 (3D Semantic Network)",
                        template="plotly_white",
                        opacity=0.8
                    )
                    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
                    fig.update_layout(
                        height=600, 
                        showlegend=False, # Hide small legend as requested
                        scene=dict(
                            xaxis=dict(showticklabels=False, title=''),
                            yaxis=dict(showticklabels=False, title=''),
                            zaxis=dict(showticklabels=False, title=''),
                        ),
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Cluster Detail Cards (The "Text Legend")
                    st.divider()
                    st.header("📑 상세 토픽 리스트 (Topic List)")
                    st.info("💡 위 3D 지도에 표시된 색상별 토픽의 상세 내용입니다.")
                    
                    unique_labels = sorted(result_df['topic_label'].unique())
                    
                    # Create a color map or just list them clearly
                    cols = st.columns(2) # 2 columns layout
                    
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
                                # Get simple samples
                                sample_issues = cluster_data[target_text_col].sample(min(2, count)).tolist()
                                for issue in sample_issues:
                                    st.caption(f"- {issue}")

                    # 3. Summary Chart
                    st.divider()
                    st.subheader("📊 토픽 비중")
                    st.bar_chart(result_df['topic_label'].value_counts())
                else:
                    st.warning("분석할 텍스트 데이터가 충분하지 않습니다.")
            except Exception as e:
                st.error(f"분석 중 오류: {e}")
