#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Assessment Generation Module
Generates risk assessments using an LLM based on relevant news articles.
"""

from typing import List, Dict, Any, Optional
from utils.config import Config
from utils.helpers import truncate_text

# LangChain components for LLM interaction
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM providers - import based on config
# Ollama is the default for local execution
try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None # type: ignore

# OpenAI (if configured)
try:
    from langchain_openai import OpenAI
except ImportError:
    OpenAI = None # type: ignore

# HuggingFace (if configured, e.g., using HuggingFaceHub or local pipelines)
# For simplicity, we'll focus on Ollama and OpenAI first.
# from langchain_community.llms import HuggingFaceHub


class AssessmentGenerator:
    """Generates risk assessments using a configured LLM"""
    
    def __init__(self, config: Config):
        """Initialize with configuration and LLM"""
        self.config = config
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        provider = self.config.llm_provider.lower()
        model_name = self.config.llm_model
        
        if provider == 'ollama':
            if Ollama is None:
                raise ImportError("Ollama provider selected, but 'langchain_community.llms.Ollama' could not be imported. Please install 'langchain-community'.")
            try:
                return Ollama(
                    model=model_name, 
                    base_url=self.config.ollama_base_url
                    # Add other parameters like temperature, top_k, etc. if needed
                )
            except Exception as e:
                print(f"Error initializing Ollama LLM ({model_name} at {self.config.ollama_base_url}): {e}")
                print("Ensure Ollama server is running and the model is available.")
                raise
        
        elif provider == 'openai':
            if OpenAI is None:
                raise ImportError("OpenAI provider selected, but 'langchain_openai.OpenAI' could not be imported. Please install 'langchain-openai'.")
            if not self.config.openai_api_key:
                raise ValueError("OpenAI provider selected, but OPENAI_API_KEY is not set.")
            try:
                return OpenAI(
                    model_name=model_name, # e.g., "gpt-3.5-turbo-instruct" for older models or specify newer ones
                    openai_api_key=self.config.openai_api_key
                    # temperature=0.7, etc.
                )
            except Exception as e:
                print(f"Error initializing OpenAI LLM ({model_name}): {e}")
                raise
                
        elif provider == 'lmstudio':
            if OpenAI is None:
                raise ImportError("LMStudio provider selected, but 'langchain_openai.OpenAI' could not be imported. Please install 'langchain-openai'.")
            try:
                # LMStudio provides an OpenAI-compatible API, so we use the OpenAI client
                # but with a custom base_url pointing to the LMStudio endpoint
                return OpenAI(
                    model_name=model_name,
                    base_url=f"{self.config.lmstudio_base_url}/v1",
                    api_key="not-needed"  # LMStudio doesn't require an API key, but the client expects one
                )
            except Exception as e:
                print(f"Error initializing LMStudio LLM ({model_name} at {self.config.lmstudio_base_url}): {e}")
                print("Ensure LMStudio server is running and the model is loaded.")
                raise
        
        # elif provider == 'huggingface':
        #     # Example for HuggingFaceHub - requires HF_TOKEN
        #     # return HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature":0.5, "max_length":64})
        #     raise NotImplementedError("HuggingFace LLM provider is not fully implemented in this version.")
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Choose from 'ollama', 'openai', 'lmstudio'.")

    def _format_articles_for_prompt(self, articles: List[Dict[str, Any]], max_articles: int = 5) -> str:
        """Format a list of articles into a string for the LLM prompt"""
        formatted_articles = []
        for i, article in enumerate(articles[:max_articles]):
            title = article.get('title', 'N/A')
            summary = truncate_text(article.get('summary') or article.get('processed_text', ''), 300)
            source = article.get('source', 'N/A')
            date = article.get('date', 'N/A')
            relevance = article.get('relevance_score', 0.0)
            
            formatted_articles.append(
                f"Article {i+1}:\n"
                f"  Title: {title}\n"
                f"  Source: {source}\n"
                f"  Date: {date}\n"
                f"  Relevance Score: {relevance:.2f}\n"
                f"  Summary: {summary}\n"
            )
        return "\n".join(formatted_articles)

    def generate_assessment(self, 
                            articles: List[Dict[str, Any]], 
                            target_text: str, 
                            detail_level: str = 'medium'
                           ) -> str:
        """
        Generate a risk assessment based on the provided articles and target text.
        Detail level can be 'low', 'medium', 'high'.
        """
        if not self.llm:
            return "LLM not initialized. Cannot generate assessment."
        if not articles:
            return "No relevant articles provided to generate an assessment."

        # Prepare context from articles
        article_context = self._format_articles_for_prompt(articles)

        # Define prompt template based on detail level
        # This is a basic template, can be significantly improved with more sophisticated prompting
        prompt_str = (
            "You are a risk assessment analyst. Based on the following news articles "
            "and the user's area of interest, provide a concise risk assessment.\n\n"
            "Area of Interest: {target_text}\n\n"
            "Relevant News Articles (summaries):\n"
            "{article_context}\n\n"
            "Risk Assessment Guidelines:\n"
            "- Identify potential risks related to the area of interest, supported by the news.\n"
            "- Briefly explain the nature of each risk.\n"
        )

        if detail_level == 'low':
            prompt_str += (
                "- Provide an overall risk level (Low, Medium, High).\n"
                "- Keep the assessment very brief (1-2 paragraphs).\n\n"
                "Assessment:"
            )
        elif detail_level == 'high':
            prompt_str += (
                "- Provide an overall risk level (Low, Medium, High) with justification.\n"
                "- For each identified risk, suggest potential mitigation strategies or areas for further investigation.\n"
                "- Discuss any uncertainties or conflicting information from the articles.\n"
                "- Conclude with a summary of key takeaways.\n"
                "- The assessment should be detailed and well-structured (multiple paragraphs).\n\n"
                "Detailed Assessment:"
            )
        else: # Medium detail
            prompt_str += (
                "- Provide an overall risk level (Low, Medium, High) with brief justification.\n"
                "- Summarize key findings and their implications.\n"
                "- Keep the assessment concise but informative (2-3 paragraphs).\n\n"
                "Assessment:"
            )
            
        prompt = PromptTemplate.from_template(prompt_str)
        
        # Create a simple chain: prompt -> llm -> output_parser
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            if self.config.debug_mode:
                print(f"\n--- LLM Prompt for Assessment ---")
                print(prompt.format(target_text=target_text, article_context=article_context))
                print("--- End of LLM Prompt ---\n")

            response = chain.invoke({
                "target_text": target_text,
                "article_context": article_context
            })
            return response.strip()
        except Exception as e:
            print(f"Error during LLM assessment generation: {e}")
            return f"Failed to generate assessment due to an error: {e}"

# Example Usage (for testing)
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv() # Load .env from project root

    # Dummy Config for testing
    class TestConfig(Config):
        def __init__(self):
            super().__init__()
            # Ensure Ollama is running with 'llama2' model or change to a model you have
            self.llm_provider = 'ollama' 
            self.llm_model = 'llama2' # or 'mistral', 'phi', etc.
            self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            self.lmstudio_base_url = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234')
            self.debug_mode = True
            
            # To test OpenAI, uncomment these and set OPENAI_API_KEY in .env
            # self.llm_provider = 'openai'
            # self.llm_model = 'gpt-3.5-turbo-instruct' # Or another suitable model
            # self.openai_api_key = os.getenv('OPENAI_API_KEY')
            
            # To test LMStudio, uncomment these and ensure LMStudio is running
            # self.llm_provider = 'lmstudio'
            # self.llm_model = 'model_name' # The model name you loaded in LMStudio
            # self.lmstudio_base_url = 'http://localhost:1234' # Default LMStudio URL


    test_config = TestConfig()
    
    try:
        generator = AssessmentGenerator(test_config)
    except (ImportError, ValueError, RuntimeError, Exception) as e:
        print(f"Could not initialize AssessmentGenerator: {e}")
        print("Skipping assessment generation test.")
        exit()

    sample_ranked_articles = [
        {'title': 'AI Stock Prices Surge', 'summary': 'Stocks for AI companies have seen a significant increase following new breakthroughs.', 'source': 'Finance News', 'date': '2024-03-10', 'relevance_score': 0.85, 'processed_text': 'Stocks for AI companies have seen a significant increase following new breakthroughs in generative models.'},
        {'title': 'Concerns over AI Job Displacement', 'summary': 'Experts express concerns about potential job displacement due to advancements in AI automation.', 'source': 'Labor Today', 'date': '2024-03-08', 'relevance_score': 0.78, 'processed_text': 'Experts express concerns about potential job displacement due to advancements in AI automation across various sectors.'},
        {'title': 'New AI Regulations Proposed', 'summary': 'Governments are considering new regulations to manage the development and deployment of AI technologies.', 'source': 'Policy Review', 'date': '2024-03-05', 'relevance_score': 0.70, 'processed_text': 'Governments worldwide are considering new regulations to manage the ethical and societal impacts of AI technologies.'}
    ]
    
    target = "investment risks in the artificial intelligence sector"

    print(f"\nGenerating 'medium' detail assessment for: '{target}'\n")
    assessment_medium = generator.generate_assessment(sample_ranked_articles, target, detail_level='medium')
    print("--- Medium Assessment ---")
    print(assessment_medium)
    
    print(f"\nGenerating 'high' detail assessment for: '{target}'\n")
    assessment_high = generator.generate_assessment(sample_ranked_articles, target, detail_level='high')
    print("\n--- High Assessment ---")
    print(assessment_high)

    print(f"\nGenerating 'low' detail assessment for: '{target}'\n")
    assessment_low = generator.generate_assessment(sample_ranked_articles, target, detail_level='low')
    print("\n--- Low Assessment ---")
    print(assessment_low)

    print("\nTesting with no articles:")
    assessment_no_articles = generator.generate_assessment([], target)
    print(assessment_no_articles)
