"""
Intelligence: The "brain" that processes information and makes decisions using LLMs.
This component handles context understanding, instruction following, and response generation.

More info: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import requests
import json


def basic_intelligence(prompt: str, model: str = "qwen2.5-coder:14b") -> str:
    """
    Send a prompt to Ollama and get a response.
    
    Args:
        prompt: The input text to send to the LLM
        model: The Ollama model to use (default: llama3.2)
    
    Returns:
        The generated text response
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["response"]


if __name__ == "__main__":
    while True:
        your_prompt = input("What do you want to know?\n")
        result = basic_intelligence(prompt=your_prompt)
        print("Basic Intelligence Output:")
        print(result)
