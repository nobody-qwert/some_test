#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Relevance Ranking System Module
Ranks news articles based on their relevance to a user's target text using embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from utils.config import Config

class RelevanceRanker:
    """Ranks articles by relevance using embedding similarity"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config

    def _calculate_cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not emb1 or not emb2:
            return 0.0
        
        # Convert lists to numpy arrays
        vec1 = np.array(emb1).reshape(1, -1)
        vec2 = np.array(emb2).reshape(1, -1)
        
        try:
            sim_score = cosine_similarity(vec1, vec2)[0][0]
            return float(sim_score) # Ensure it's a standard float
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def rank_articles(self, 
                      articles: List[Dict[str, Any]], 
                      target_embedding: List[float]
                     ) -> List[Dict[str, Any]]:
        """
        Rank articles based on the cosine similarity of their embeddings to the target embedding.
        Articles are expected to have an 'embedding' key.
        """
        if not target_embedding:
            if self.config.debug_mode:
                print("Target embedding is missing, cannot rank articles.")
            return articles # Return original list or an empty list, depending on desired behavior

        ranked_articles = []
        for article in articles:
            article_embedding = article.get('embedding')
            if not article_embedding:
                if self.config.debug_mode:
                    print(f"Article '{article.get('title', 'N/A')}' missing embedding, assigning score 0.")
                score = 0.0
            else:
                score = self._calculate_cosine_similarity(target_embedding, article_embedding)
            
            # Add relevance score to the article dictionary
            # Create a copy to avoid modifying the original list of dicts in place if it's reused
            ranked_article = article.copy()
            ranked_article['relevance_score'] = score
            ranked_articles.append(ranked_article)
            
        # Sort articles by relevance score in descending order
        ranked_articles.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        
        return ranked_articles

# Example Usage (for testing)
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    # Dummy Config for testing
    class TestConfig(Config):
        def __init__(self):
            super().__init__()
            self.debug_mode = True

    test_config = TestConfig()
    ranker = RelevanceRanker(test_config)

    # Sample processed articles with embeddings (replace with actual embeddings)
    # Dimensions should match your embedding model (e.g., 384 for all-MiniLM-L6-v2)
    dim = 384 
    sample_articles_processed = [
        {'title': 'Article A about AI', 'embedding': np.random.rand(dim).tolist(), 'source': 'TechCrunch', 'date': '2024-01-01', 'url': 'http://example.com/a'},
        {'title': 'Article B about Space', 'embedding': np.random.rand(dim).tolist(), 'source': 'NASA News', 'date': '2024-01-02', 'url': 'http://example.com/b'},
        {'title': 'Article C also on AI', 'embedding': np.random.rand(dim).tolist(), 'source': 'Wired', 'date': '2024-01-03', 'url': 'http://example.com/c'},
        {'title': 'Article D with no embedding', 'source': 'Old News', 'date': '2023-12-01', 'url': 'http://example.com/d'}, # No embedding
        {'title': 'Article E very relevant to AI', 'embedding': np.random.rand(dim).tolist(), 'source': 'AI Journal', 'date': '2024-01-05', 'url': 'http://example.com/e'},
    ]
    
    # Simulate a target embedding (e.g., for "risks of artificial intelligence")
    sample_target_embedding = np.random.rand(dim).tolist()
    
    # Make Article E's embedding closer to the target for a more predictable top result
    if sample_articles_processed[4].get('embedding'): # Check if embedding exists
        # Create a slightly perturbed version of the target embedding
        noise = np.random.normal(0, 0.1, dim) # Small noise
        perturbed_target = np.array(sample_target_embedding) + noise
        sample_articles_processed[4]['embedding'] = perturbed_target.tolist()


    print("\nRanking sample articles...\n")
    ranked_list = ranker.rank_articles(sample_articles_processed, sample_target_embedding)

    if ranked_list:
        print("Ranked Articles (Top 5):")
        for i, article in enumerate(ranked_list[:5]):
            print(f"{i+1}. {article['title']} (Score: {article.get('relevance_score', 0.0):.4f})")
            print(f"   Source: {article['source']}, Date: {article['date']}, URL: {article['url']}")
    else:
        print("No articles were ranked.")

    print("\nTesting with missing target embedding:")
    ranked_no_target_emb = ranker.rank_articles(sample_articles_processed, None)
    if ranked_no_target_emb == sample_articles_processed:
         print("Returned original list as expected when target embedding is missing.")
