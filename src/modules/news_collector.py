#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
News Data Collection Module
Fetches news articles from various sources like NewsAPI, Google News, etc.
"""

import os
import requests
import datetime
import feedparser # For Google News RSS
from typing import List, Dict, Any, Optional

from utils.config import Config
from utils.helpers import (
    generate_cache_key, 
    save_to_cache, 
    load_from_cache, 
    format_date,
    extract_domain
)

class NewsCollector:
    """Collects news articles from configured sources"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'RiskAssessmentApp/1.0'})

    def fetch_articles(self, topic: str, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch articles from all configured sources"""
        all_articles = []
        
        # Primary: NewsAPI
        if self.config.news_api_key:
            articles_newsapi = self._fetch_from_newsapi(topic, days)
            if articles_newsapi:
                all_articles.extend(articles_newsapi)
        
        # Alternative 1: Google News RSS
        if self.config.use_google_news:
            articles_google = self._fetch_from_google_news(topic, days)
            if articles_google:
                all_articles.extend(articles_google)
        
        # TODO: Implement other sources (Twitter, Web Scraping) as per plan
        
        # Deduplicate articles based on URL
        unique_articles = {article['url']: article for article in all_articles}.values()
        
        return list(unique_articles)

    def _fetch_from_newsapi(self, topic: str, days: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch articles from NewsAPI"""
        cache_key = generate_cache_key(f"newsapi_{topic}_{days}", source="newsapi")
        cached_data = load_from_cache(cache_key, self.config.cache_dir, self.config.cache_duration)
        
        if cached_data:
            if self.config.debug_mode:
                print(f"NewsAPI: Loaded {len(cached_data)} articles from cache for topic '{topic}'")
            return cached_data
            
        if self.config.debug_mode:
            print(f"NewsAPI: Fetching articles for topic '{topic}'")

        from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        params = {
            'q': topic,
            'apiKey': self.config.news_api_key,
            'from': from_date,
            'sortBy': 'relevancy', # or 'publishedAt'
            'language': 'en',
            'pageSize': 100 # Max allowed by NewsAPI free tier
        }
        
        try:
            response = self.session.get(f"{self.config.news_api_endpoint}everything", params=params, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            
            articles = []
            for item in data.get('articles', []):
                articles.append({
                    'title': item.get('title'),
                    'summary': item.get('description'),
                    'content': item.get('content'), # Often truncated in NewsAPI
                    'url': item.get('url'),
                    'date': format_date(item.get('publishedAt')),
                    'source': item.get('source', {}).get('name', extract_domain(item.get('url'))),
                    'raw_data': item # Store original data if needed
                })
            
            save_to_cache(articles, cache_key, self.config.cache_dir)
            if self.config.debug_mode:
                print(f"NewsAPI: Fetched and cached {len(articles)} articles for topic '{topic}'")
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from NewsAPI: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred with NewsAPI: {e}")
            return None

    def _fetch_from_google_news(self, topic: str, days: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch articles from Google News RSS"""
        cache_key = generate_cache_key(f"googlenews_{topic}_{days}", source="googlenews")
        cached_data = load_from_cache(cache_key, self.config.cache_dir, self.config.cache_duration)

        if cached_data:
            if self.config.debug_mode:
                print(f"GoogleNews: Loaded {len(cached_data)} articles from cache for topic '{topic}'")
            return cached_data

        if self.config.debug_mode:
            print(f"GoogleNews: Fetching articles for topic '{topic}'")
            
        # Construct Google News RSS URL
        # Example: https://news.google.com/rss/search?q=python+programming&hl=en-US&gl=US&ceid=US:en
        # We can also add date filtering, e.g., q=topic+when:7d
        query = f"{topic} when:{days}d"
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(url)
            articles = []
            
            if feed.bozo: # Check for malformed feed
                if isinstance(feed.bozo_exception, Exception):
                    raise feed.bozo_exception
                print(f"Warning: Google News RSS feed for '{topic}' may be malformed.")

            for entry in feed.entries:
                articles.append({
                    'title': entry.get('title'),
                    'summary': entry.get('summary'), # Often same as title or short snippet
                    'content': entry.get('summary'), # RSS usually doesn't provide full content
                    'url': entry.get('link'),
                    'date': format_date(entry.get('published')), # 'published_parsed' is also available
                    'source': entry.get('source', {}).get('title', extract_domain(entry.get('link'))),
                    'raw_data': entry
                })
            
            save_to_cache(articles, cache_key, self.config.cache_dir)
            if self.config.debug_mode:
                print(f"GoogleNews: Fetched and cached {len(articles)} articles for topic '{topic}'")
            return articles

        except Exception as e:
            print(f"Error fetching from Google News: {e}")
            return None

# Example Usage (for testing)
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv() # Load .env file from project root
    
    # Create a dummy config for testing
    class TestConfig(Config):
        def __init__(self):
            super().__init__()
            # Override for testing if needed
            # self.news_api_key = "YOUR_NEWS_API_KEY" # Replace with a valid key for testing NewsAPI
            self.use_google_news = True
            self.debug_mode = True
            self.cache_dir = '../../src/data/cache' # Adjust path if running from modules dir
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

    test_config = TestConfig()
    
    # Validate config (optional, but good practice)
    errors = test_config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"- {error}")
        exit(1)

    collector = NewsCollector(test_config)
    
    test_topic = "artificial intelligence"
    test_days = 3
    
    print(f"\nFetching articles for topic: '{test_topic}' (last {test_days} days)\n")
    
    fetched_articles = collector.fetch_articles(test_topic, days=test_days)
    
    if fetched_articles:
        print(f"\nSuccessfully fetched {len(fetched_articles)} unique articles:")
        for i, article in enumerate(fetched_articles[:5], 1): # Print first 5
            print(f"{i}. {article['title']} ({article['source']}) - {article['date']}")
            print(f"   URL: {article['url']}")
            if article['summary']:
                 print(f"   Summary: {article['summary'][:100]}...")
            print()
    else:
        print("No articles found.")
