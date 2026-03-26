from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import Dict, Any

from core.db import supabase
from analysis import SemanticAnalyzer
from core.config import settings

router = APIRouter()

@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
        
    response = supabase.table("interviews").select("*").execute()
    data = response.data
    
    if not data:
        return {
            "total_interviews": 0,
            "total_sessions": 0,
            "avg_severity": 0,
            "issues_per_session": 0,
            "location_distribution": {},
            "category_distribution": {}
        }
        
    df = pd.DataFrame(data)
    
    total = len(df)
    unique_sessions = df["session_id"].nunique() if "session_id" in df.columns else 1
    
    avg_severity = 0
    if "severity_score" in df.columns:
        avg_severity = pd.to_numeric(df["severity_score"], errors="coerce").mean()
        
    issues_per_session = total / max(unique_sessions, 1)
    
    location_dist = df["location_bucket"].value_counts().to_dict() if "location_bucket" in df.columns else {}
    category_dist = df["primary_category"].value_counts().to_dict() if "primary_category" in df.columns else {}
    
    return {
        "total_interviews": total,
        "total_sessions": unique_sessions,
        "avg_severity": round(float(avg_severity), 1) if not pd.isna(avg_severity) else 0,
        "issues_per_session": round(float(issues_per_session), 1),
        "location_distribution": location_dist,
        "category_distribution": category_dist
    }

@router.get("/tsne")
async def get_tsne_data() -> Dict[str, Any]:
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
        
    response = supabase.table("interviews").select("*").execute()
    data = response.data
    
    if not data:
        return {}
        
    df = pd.DataFrame(data)
    
    try:
        analyzer = SemanticAnalyzer(api_key=settings.OPENAI_API_KEY)
        result_df = analyzer.process_and_analyze(df, text_column='issue_text', n_dimensions=3)
        
        if 'x' not in result_df.columns:
            return {"error": "Not enough data for t-SNE analysis"}
            
        topics = result_df['topic_label'].unique().tolist()
        topic_stats = []
        
        for topic in topics:
            cluster_data = result_df[result_df['topic_label'] == topic]
            topic_stats.append({
                "label": topic,
                "count": int(len(cluster_data)),
                "avg_severity": float(cluster_data["severity_score"].mean()) if "severity_score" in cluster_data.columns else 0,
                "sample_issues": cluster_data["issue_text"].dropna().sample(min(2, len(cluster_data))).tolist()
            })
            
        return {
            "scatter_data": {
                "x": result_df['x'].tolist(),
                "y": result_df['y'].tolist(),
                "z": result_df['z'].tolist(),
                "topic_labels": result_df['topic_label'].tolist(),
                "hover_texts": result_df['issue_text'].tolist(),
                "locations": result_df['location_bucket'].tolist(),
                "severities": result_df['severity_score'].tolist()
            },
            "topics": topic_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
