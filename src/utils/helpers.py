#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper utilities for the Risk Assessment Application
"""

import os
import json
import hashlib
import datetime
from typing import Dict, List, Any, Optional

def create_cache_dir(cache_dir: str) -> None:
    """Create cache directory if it doesn't exist"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

def generate_cache_key(query: str, source: str = "default") -> str:
    """Generate a unique cache key based on query and source"""
    hash_input = f"{query}_{source}_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def save_to_cache(data: Any, cache_key: str, cache_dir: str) -> None:
    """Save data to cache file"""
    create_cache_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'data': data
        }, f, ensure_ascii=False, indent=2)

def load_from_cache(cache_key: str, cache_dir: str, max_age_seconds: int = 3600) -> Optional[Any]:
    """Load data from cache if it exists and is not expired"""
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        timestamp = datetime.datetime.fromisoformat(cache_data['timestamp'])
        age = (datetime.datetime.now() - timestamp).total_seconds()
        
        if age > max_age_seconds:
            return None
        
        return cache_data['data']
    except (json.JSONDecodeError, KeyError, ValueError):
        # If cache file is corrupted, ignore it
        return None

def format_date(date_str: str) -> str:
    """Format date string to a consistent format"""
    try:
        # Try to parse various date formats
        for fmt in ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
            try:
                dt = datetime.datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If none of the formats match, return the original string
        return date_str
    except Exception:
        return date_str

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace('www.', '')
    except Exception:
        return url
