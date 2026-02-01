import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class SemanticAnalyzer:
    def __init__(self, api_key: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Use a lightweight embedding model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts."""
        if not texts:
            return np.array([])
        vectors = self.embeddings.embed_documents(texts)
        return np.array(vectors)

    def perform_clustering(self, vectors: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """Performs K-Means clustering."""
        if len(vectors) < n_clusters:
            n_clusters = max(1, len(vectors))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        return kmeans.fit_predict(vectors)

    def reduce_dimensions(self, vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduces dimensions to 2D or 3D using t-SNE."""
        if len(vectors) < n_components:
             # Fallback for very small data
            return np.zeros((len(vectors), n_components))
        
        # Perplexity must be less than n_samples
        perplexity = min(30, len(vectors) - 1)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='random')
        return tsne.fit_transform(vectors)

    def generate_topic_labels(self, df: pd.DataFrame, text_col: str, cluster_col: str) -> Dict[int, str]:
        """Generates a short topic label for each cluster using GPT-4o."""
        labels = {}
        unique_clusters = sorted(df[cluster_col].unique())
        
        for cluster_id in unique_clusters:
            # Get samples from this cluster
            samples = df[df[cluster_col] == cluster_id][text_col].head(10).tolist()
            text_dump = "\n".join([f"- {s}" for s in samples])
            
            prompt = f"""
            Analyze the following list of urban design issues/feedback and generate a concise, 2-3 word category tag that represents the common theme.
            
            Feedback Samples:
            {text_dump}
            
            Format: Just the tag. No extra text.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            labels[cluster_id] = response.content.strip()
            
        return labels

    def process_and_analyze(self, df: pd.DataFrame, text_column: str = 'issue', n_dimensions: int = 2) -> pd.DataFrame:
        """Main pipeline: Embed -> Cluster -> t-SNE -> Auto-Label."""
        if df.empty or text_column not in df.columns:
            return df

        # 1. Clean Data
        df = df[df[text_column].notna() & (df[text_column] != "")].copy()
        texts = df[text_column].tolist()
        
        if not texts:
            return df
        
        # 2. Embeddings
        vectors = self.generate_embeddings(texts)
        
        # 3. Clustering 
        # Heuristic: sqrt(N/2) usually works fine for small N
        n_clusters = max(2, int(np.sqrt(len(texts)/2)))
        n_clusters = min(n_clusters, 8) # Cap at 8 clusters
            
        clusters = self.perform_clustering(vectors, n_clusters=n_clusters)
        df['cluster'] = clusters
        
        # 4. t-SNE (2D or 3D)
        coords = self.reduce_dimensions(vectors, n_components=n_dimensions)
        df['x'] = coords[:, 0]
        df['y'] = coords[:, 1]
        if n_dimensions == 3:
            df['z'] = coords[:, 2]
        
        # 5. Labeling
        topic_map = self.generate_topic_labels(df, text_column, 'cluster')
        df['topic_label'] = df['cluster'].map(topic_map)
        
        return df
