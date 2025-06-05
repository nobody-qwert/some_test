#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
News-Based Risk Assessment Application
Main entry point for the application
"""

import os
import argparse
from dotenv import load_dotenv

# Import modules
from modules.news_collector import NewsCollector
from modules.text_processor import TextProcessor
from modules.relevance_ranker import RelevanceRanker
from modules.assessment_generator import AssessmentGenerator
from utils.config import Config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='News-Based Risk Assessment Tool')
    parser.add_argument('--topic', type=str, required=True, help='Topic to search for news')
    parser.add_argument('--target', type=str, required=True, help='Target text for relevance analysis')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back for news (default: 7)')
    parser.add_argument('--articles', type=int, default=10, help='Number of articles to analyze (default: 10)')
    parser.add_argument('--detail', choices=['low', 'medium', 'high'], default='medium', 
                        help='Assessment detail level (default: medium)')
    return parser.parse_args()

def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config()
    
    # Initialize modules
    news_collector = NewsCollector(config)
    text_processor = TextProcessor(config)
    relevance_ranker = RelevanceRanker(config)
    assessment_generator = AssessmentGenerator(config)
    
    # Step 1: Collect news articles
    print(f"Collecting news articles for topic: {args.topic}")
    articles = news_collector.fetch_articles(args.topic, days=args.days)
    
    if not articles:
        print("No articles found for the specified topic.")
        return
    
    print(f"Found {len(articles)} articles")
    
    # Step 2: Process articles and generate embeddings
    print("Processing articles and generating embeddings...")
    processed_articles = text_processor.process_articles(articles)
    
    # Step 3: Rank articles by relevance to target text
    print("Ranking articles by relevance...")
    target_embedding = text_processor.generate_embedding(args.target)
    ranked_articles = relevance_ranker.rank_articles(processed_articles, target_embedding)
    
    # Get top N articles
    top_articles = ranked_articles[:args.articles]
    
    # Step 4: Generate risk assessment
    print("Generating risk assessment...")
    assessment = assessment_generator.generate_assessment(top_articles, args.target, detail_level=args.detail)
    
    # Step 5: Display results
    print("\n" + "="*80)
    print("RISK ASSESSMENT REPORT")
    print("="*80)
    print(f"Topic: {args.topic}")
    print(f"Target: {args.target}")
    print(f"Date Range: Last {args.days} days")
    print(f"Articles Analyzed: {len(top_articles)}")
    print("="*80 + "\n")
    
    print(assessment)
    
    print("\n" + "="*80)
    print("TOP RELEVANT ARTICLES")
    print("="*80)
    
    for i, article in enumerate(top_articles, 1):
        print(f"{i}. {article['title']} ({article['source']})")
        print(f"   Date: {article['date']}")
        print(f"   Relevance Score: {article['relevance_score']:.2f}")
        print(f"   URL: {article['url']}")
        print()

if __name__ == "__main__":
    main()
