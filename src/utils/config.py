#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration utility for the Risk Assessment Application
"""

import os
from dotenv import load_dotenv

class Config:
    """Configuration class for the application"""
    
    def __init__(self):
        """Initialize configuration with environment variables"""
        # Ensure environment variables are loaded
        load_dotenv()
        
        # News API configuration
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.news_api_endpoint = os.getenv('NEWS_API_ENDPOINT', 'https://newsapi.org/v2/')
        
        # Alternative news sources
        self.use_google_news = os.getenv('USE_GOOGLE_NEWS', 'false').lower() == 'true'
        self.use_twitter = os.getenv('USE_TWITTER', 'false').lower() == 'true'
        self.use_web_scraping = os.getenv('USE_WEB_SCRAPING', 'false').lower() == 'true'
        
        # Twitter API configuration (if enabled)
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        # Embedding configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.use_openai_embeddings = os.getenv('USE_OPENAI_EMBEDDINGS', 'false').lower() == 'true'
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # LLM configuration
        self.llm_provider = os.getenv('LLM_PROVIDER', 'ollama')  # 'ollama', 'openai', 'lmstudio', or 'huggingface'
        self.llm_model = os.getenv('LLM_MODEL', 'llama2')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.lmstudio_base_url = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234')
        
        # Application settings
        self.cache_dir = os.getenv('CACHE_DIR', 'src/data/cache')
        self.cache_duration = int(os.getenv('CACHE_DURATION', '3600'))  # Default: 1 hour
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
    def validate(self):
        """Validate the configuration"""
        errors = []
        
        # Check if at least one news source is configured
        if not self.news_api_key and not self.use_google_news and not self.use_twitter and not self.use_web_scraping:
            errors.append("No news source configured. Please set NEWS_API_KEY or enable an alternative source.")
        
        # Check Twitter configuration if enabled
        if self.use_twitter and not (self.twitter_api_key and self.twitter_api_secret and self.twitter_bearer_token):
            errors.append("Twitter API is enabled but credentials are missing.")
        
        # Check OpenAI configuration if enabled
        if self.use_openai_embeddings and not self.openai_api_key:
            errors.append("OpenAI embeddings are enabled but API key is missing.")
        
        if self.llm_provider == 'openai' and not self.openai_api_key:
            errors.append("OpenAI LLM is selected but API key is missing.")
            
        if self.llm_provider == 'lmstudio' and not self.lmstudio_base_url:
            errors.append("LMStudio LLM is selected but base URL is missing.")
        
        return errors
