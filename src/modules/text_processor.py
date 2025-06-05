#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Processing and Embedding Module
Handles text cleaning, splitting, and generating embeddings for news articles.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
# LlamaIndex might be used for more advanced indexing if needed later
# from llama_index.core.node_parser import SentenceSplitter

from utils.config import Config
from utils.helpers import truncate_text

class TextProcessor:
    """Processes text and generates embeddings"""
    
    def __init__(self, config: Config):
        """Initialize with configuration and embedding model"""
        self.config = config
        
        if self.config.use_openai_embeddings:
            # Placeholder for OpenAIEmbeddings if chosen
            # from langchain_openai import OpenAIEmbeddings
            # self.embedding_model = OpenAIEmbeddings(api_key=self.config.openai_api_key)
            # For now, we'll raise a NotImplementedError if OpenAI is selected without full implementation
            raise NotImplementedError("OpenAI embeddings are not fully implemented in this version.")
        else:
            # Use local sentence-transformers model
            try:
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                print(f"Error loading SentenceTransformer model '{self.config.embedding_model}': {e}")
                print("Please ensure the model is installed or use a default one like 'all-MiniLM-L6-v2'.")
                # Fallback or re-raise depending on desired robustness
                raise
        
        # LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Adjust as needed
            chunk_overlap=64, # Adjust as needed
            length_function=len,
            is_separator_regex=False,
        )

    def preprocess_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        # Add more cleaning steps if needed (e.g., remove HTML, special chars)
        text = text.strip()
        return text

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single piece of text"""
        if not text:
            return None
        try:
            clean_text = self.preprocess_text(text)
            # SentenceTransformer expects a list of sentences, but can take a single string
            embedding = self.embedding_model.encode(clean_text)
            return embedding.tolist() # Convert numpy array to list
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of articles: clean, chunk, and generate embeddings"""
        processed_articles = []
        for article in articles:
            # Combine relevant text fields for embedding
            # Prioritize content, then summary, then title
            text_to_embed = article.get('content') or article.get('summary') or article.get('title')
            
            if not text_to_embed:
                if self.config.debug_mode:
                    print(f"Skipping article with no text content: {article.get('title', 'N/A')}")
                continue

            # Preprocess and clean the text
            clean_text = self.preprocess_text(text_to_embed)
            
            # Generate embedding for the (potentially truncated) main text
            # For simplicity, we embed the summary or a truncated version of content.
            # For more complex scenarios, chunking and averaging embeddings might be better.
            # Here, we'll use a truncated version of the clean_text for the main embedding.
            main_text_for_embedding = truncate_text(clean_text, max_length=1000) # Ensure not too long for ST model
            embedding = self.generate_embedding(main_text_for_embedding)
            
            if embedding:
                processed_article = article.copy() # Avoid modifying original dict
                processed_article['embedding'] = embedding
                processed_article['processed_text'] = clean_text # Store the full cleaned text
                
                # Optional: Chunking for more granular analysis if needed later
                # chunks = self.text_splitter.split_text(clean_text)
                # processed_article['chunks'] = chunks
                # processed_article['chunk_embeddings'] = [self.generate_embedding(chunk) for chunk in chunks if chunk]
                
                processed_articles.append(processed_article)
            elif self.config.debug_mode:
                print(f"Failed to generate embedding for article: {article.get('title', 'N/A')}")
                
        return processed_articles

# Example Usage (for testing)
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    class TestConfig(Config):
        def __init__(self):
            super().__init__()
            self.embedding_model = 'all-MiniLM-L6-v2' # Ensure this model is available
            self.debug_mode = True
            # self.use_openai_embeddings = True # Test this path if OpenAI is set up
            # self.openai_api_key = "sk-..."

    test_config = TestConfig()
    
    # Check if OpenAI is selected and handle potential NotImplementedError
    try:
        processor = TextProcessor(test_config)
    except NotImplementedError as e:
        print(e)
        print("Skipping OpenAI-dependent tests.")
        exit()
    except Exception as e:
        print(f"Failed to initialize TextProcessor: {e}")
        exit()

    sample_articles = [
        {
            'title': 'AI Revolutionizes Healthcare', 
            'summary': 'Artificial intelligence is making significant strides in healthcare diagnostics and treatment.',
            'content': 'Detailed content about AI in healthcare, covering various applications like image analysis, drug discovery, and personalized medicine. It discusses the benefits and challenges associated with integrating AI into clinical workflows.',
            'url': 'http://example.com/ai-healthcare',
            'date': '2024-01-15',
            'source': 'Tech News'
        },
        {
            'title': 'Market Trends Q1 2024',
            'summary': 'The first quarter of 2024 saw mixed results across different market sectors.',
            # No 'content' field for this one
            'url': 'http://example.com/market-trends',
            'date': '2024-04-01',
            'source': 'Finance Today'
        },
        {
            'title': None, # Test missing title
            'summary': None, # Test missing summary
            'content': None, # Test missing content
            'url': 'http://example.com/empty',
            'date': '2024-01-01',
            'source': 'Empty News'
        }
    ]

    print("\nProcessing sample articles...\n")
    processed = processor.process_articles(sample_articles)

    if processed:
        print(f"Successfully processed {len(processed)} articles:")
        for i, article in enumerate(processed):
            print(f"\nArticle {i+1}: {article['title']}")
            print(f"  Embedding dimensions: {len(article['embedding']) if article.get('embedding') else 'N/A'}")
            # print(f"  Processed text sample: {article.get('processed_text', '')[:100]}...")
            # if 'chunks' in article:
            #     print(f"  Number of chunks: {len(article['chunks'])}")
            #     if article['chunks']:
            #         print(f"    First chunk embedding: {len(article['chunk_embeddings'][0]) if article['chunk_embeddings'][0] else 'N/A'}")
        
        print("\nGenerating embedding for a sample target text:")
        target_text = "risks associated with artificial intelligence in medical diagnosis"
        target_embedding = processor.generate_embedding(target_text)
        if target_embedding:
            print(f"  Target text: '{target_text}'")
            print(f"  Embedding dimensions: {len(target_embedding)}")
        else:
            print(f"  Failed to generate embedding for target text.")
            
    else:
        print("No articles were processed.")
