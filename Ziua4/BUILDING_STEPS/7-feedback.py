"""
Feedback: Provides strategic points where human judgement is required.
This component implements approval workflows and human-in-the-loop processes for high-risk decisions or complex judgments.
"""

import requests


def get_human_approval(content: str) -> bool:
    """
    Request human approval for generated content.
    
    Args:
        content: The generated content to review
    
    Returns:
        True if approved, False otherwise
    """
    print(f"Generated content:\n{content}\n")
    response = input("Approve this? (y/n): ")
    return response.lower().startswith("y")


def intelligence_with_human_feedback(prompt: str, model: str = "gemma3:270m") -> None:
    """
    Generate content with Ollama and request human approval before finalizing.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The Ollama model to use (default: llama3.2)
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
    draft_response = result["response"]

    if get_human_approval(draft_response):
        print("✅ Final answer approved")
    else:
        print("❌ Answer not approved")


if __name__ == "__main__":
    intelligence_with_human_feedback("Write a short poem about technology")
