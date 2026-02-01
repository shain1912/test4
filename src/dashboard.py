import streamlit as st
import pandas as pd
import sqlite3
import os
import json

# Adjust path to find db
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interviews.db')

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM interviews", conn)
    conn.close()
    return df

st.set_page_config(page_title="부산 걷기 좋은 도시 - 인터뷰 보드", layout="wide")

st.title("🚶 부산 걷기 좋은 도시 만들기 - 시민 인터뷰 현황")
st.markdown("시민들과의 1:1 채팅 인터뷰를 통해 수집된 보행 환경 데이터입니다.")

try:
    df = load_data()
    
    if df.empty:
        st.warning("아직 수집된 데이터가 없습니다.")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("총 인터뷰 수", len(df))
        col2.metric("최근 수집", df['timestamp'].iloc[-1] if not df.empty else "-")
        col3.metric("주요 이슈 유형", df['issue'].nunique())

        st.divider()

        # Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("🏙️ 지역별 분포")
            if 'location' in df.columns:
                loc_counts = df['location'].value_counts()
                st.bar_chart(loc_counts)
        
        with col_c2:
            st.subheader("💡 제안된 해결책 유형")
            if 'solution_type' in df.columns:
                sol_counts = df['solution_type'].value_counts()
                st.bar_chart(sol_counts)

        st.divider()
        
        # Data Table
        st.subheader("📋 상세 인터뷰 로그")
        
        # Display key columns
        display_cols = ['id', 'timestamp', 'location', 'urban_element', 'issue', 'solution_type', 'primary_value']
        st.dataframe(df[display_cols], use_container_width=True)
        
# ... existing code ...
        # Expander for full details
        with st.expander("상세 데이터 보기 (선택)"):
            selected_id = st.number_input("ID 입력", min_value=1, max_value=len(df), step=1)
            row = df[df['id'] == selected_id]
            if not row.empty:
                st.json(row.to_dict(orient='records')[0])

        st.divider()
        st.header("🧠 AI 의미 분석 (Semantic Analysis)")
        st.markdown("데이터의 의미를 분석하여 자동으로 토픽을 분류하고 시각화합니다.")

        if st.button("분석 실행 (Analyze)"):
            from src.analysis import SemanticAnalyzer
            import plotly.express as px

            with st.spinner("임베딩 생성 및 군집화 분석 중... (시간이 조금 걸릴 수 있습니다)"):
                analyzer = SemanticAnalyzer()
                # Use 'issue' column for clustering
                result_df = analyzer.process_and_analyze(df, text_column='issue')
                
                if 'x' in result_df.columns:
                    st.success("분석 완료!")
                    
                    # Store in session state to avoid re-running on interaction? (Simple for now)
                    
                    fig = px.scatter(
                        result_df, 
                        x='x', 
                        y='y', 
                        color='topic_label',
                        hover_data=['issue', 'location', 'solution_type'],
                        title="이슈 데이터 의미 연결망 (Semantic Network)",
                        template="plotly_white"
                    )
                    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
                    fig.update_layout(showlegend=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("분류된 토픽 요약")
                    topic_counts = result_df['topic_label'].value_counts()
                    st.bar_chart(topic_counts)
                else:
                    st.warning("분석할 텍스트 데이터가 충분하지 않습니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")

