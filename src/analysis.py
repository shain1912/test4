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

    def perform_clustering(self, vectors: np.ndarray, min_clusters: int = 3, max_clusters: int = 8) -> np.ndarray:
        """Performs K-Means clustering with automatic K selection using Silhouette Score."""
        from sklearn.metrics import silhouette_score
        
        best_k = min_clusters
        best_score = -1
        best_labels = None
        
        # If data is too small, just use min_clusters
        if len(vectors) < min_clusters + 2:
            kmeans = KMeans(n_clusters=min(len(vectors), min_clusters), random_state=42, n_init='auto')
            return kmeans.fit_predict(vectors)

        # Iterate to find optimal K
        for k in range(min_clusters, min(max_clusters + 1, len(vectors))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        
        print(f"DEBUG: Optimal Clusters (K) = {best_k} (Silhouette Score: {best_score:.3f})")
        return best_labels

    def reduce_dimensions(self, vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduces dimensions to 2D or 3D using t-SNE with robust parameters."""
        if len(vectors) < n_components:
            return np.zeros((len(vectors), n_components))
        
        # Perplexity heuristic: closely related to number of nearest neighbors
        # For small datasets (<100), perplexity between 5 and 30 is good.
        # We make it dynamic based on data size.
        n_samples = len(vectors)
        perplexity = min(30, max(5, n_samples // 4))
        
        # Increase iterations for better stability
        tsne = TSNE(
            n_components=n_components, 
            perplexity=perplexity, 
            random_state=42, 
            init='pca', 
            learning_rate='auto'
        )
        return tsne.fit_transform(vectors)

    def generate_topic_labels(self, df: pd.DataFrame, text_col: str, cluster_col: str) -> Dict[int, str]:
        """Generates a short topic label for each cluster using GPT-4o."""
        labels = {}
        unique_clusters = sorted(df[cluster_col].unique())
        
        for cluster_id in unique_clusters:
            # Get samples from this cluster (increase sample size for better context)
            samples = df[df[cluster_col] == cluster_id][text_col].head(15).tolist()
            text_dump = "\n".join([f"- {s}" for s in samples])
            
            prompt = f"""
            You are an expert Urban Analyst. 
            Below are complaints/feedback from citizens about a specific urban issue in Busan.
            Identify the **single core theme** that binds them together.
            
            Key attributes to look for:
            - Location (e.g., Seomyeon, Haeundae)
            - Incident Type (e.g., Broken lights, Trash, Accessibility)
            
            Output strictly a Short Tag (max 3 words). 
            Example: "Seomyeon Night Safety", "Station Accessibility", "Cafe Street Trash"
            
            Feedback Samples:
            {text_dump}
            
            Tag:
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            labels[cluster_id] = response.content.strip().replace('"', '')
            
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
        
        # 3. Clustering (Auto-tuned)
        clusters = self.perform_clustering(vectors, min_clusters=3, max_clusters=8)
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
