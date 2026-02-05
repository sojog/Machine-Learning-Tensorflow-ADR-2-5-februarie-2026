"""
Memory: Stores and retrieves relevant information across interactions.
This component maintains conversation history and context to enable coherent multi-turn interactions.

More info: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
"""

import requests
import json


MODEL = "gemma3:270m"
PROMPT = "Tell me a joke about programming"


def chat_with_ollama(messages: list, model: str = MODEL ) -> str:
    """
    Send a conversation to Ollama and get a response.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The Ollama model to use (default: llama3.2)
    
    Returns:
        The assistant's response text
    """
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["message"]["content"]


def ask_joke_without_memory():
    messages = [
        {"role": "user", "content": PROMPT},
    ]
    return chat_with_ollama(messages)


def ask_followup_without_memory():
    messages = [
        {"role": "user", "content": "What was my previous question?"},
    ]
    return chat_with_ollama(messages)


def ask_followup_with_memory(joke_response: str):
    messages = [
        {"role": "user", "content": "Tell me a joke about programming"},
        {"role": "assistant", "content": joke_response},
        {"role": "user", "content": "What was my previous question?"},
    ]
    return chat_with_ollama(messages)


if __name__ == "__main__":
    # First: Ask for a joke
    joke_response = ask_joke_without_memory()
    print(joke_response, "\n")

    # Second: Ask follow-up without memory (AI will be confused)
    confused_response = ask_followup_without_memory()
    print(confused_response, "\n")

    # Third: Ask follow-up with memory (AI will remember)
    memory_response = ask_followup_with_memory(joke_response)
    print(memory_response)
