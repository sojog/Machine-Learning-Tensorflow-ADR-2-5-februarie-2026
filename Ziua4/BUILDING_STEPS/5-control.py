"""
Control: Provides deterministic decision-making and process flow control.
This component handles if/then logic, routing based on conditions, and process orchestration for predictable behavior.
"""

import requests
import json
from pydantic import BaseModel, ValidationError
from typing import Literal


class IntentClassification(BaseModel):
    intent: Literal["question", "request", "complaint"]
    confidence: float
    reasoning: str


def classify_intent(user_input: str, model: str = "gemma3:270m") -> IntentClassification:
    """
    Classify user input into one of three categories using Ollama.
    
    Args:
        user_input: The text to classify
        model: The Ollama model to use (default: llama3.2)
    
    Returns:
        IntentClassification object with intent, confidence, and reasoning
    """
    url = "http://localhost:11434/api/chat"
    
    system_prompt = """Classify user input into one of three categories: question, request, or complaint.
Return ONLY a JSON object with these fields:
- intent: must be exactly one of: "question", "request", or "complaint"
- confidence: a number between 0 and 1
- reasoning: a brief explanation of your classification

Example: {"intent": "question", "confidence": 0.95, "reasoning": "User is asking for information"}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json"
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    content = result["message"]["content"]
    data = json.loads(content)
    
    return IntentClassification(**data)


def route_based_on_intent(user_input: str, model: str = "gemma3:270m") -> tuple[str, IntentClassification]:
    """
    Route user input to appropriate handler based on intent classification.
    
    Args:
        user_input: The user's message
        model: The Ollama model to use
    
    Returns:
        Tuple of (response, classification)
    """
    classification = classify_intent(user_input, model)
    intent = classification.intent

    # Deterministic routing based on classification
    if intent == "question":
        result = answer_question(user_input, model)
    elif intent == "request":
        result = process_request(user_input)
    elif intent == "complaint":
        result = handle_complaint(user_input)
    else:
        result = "I'm not sure how to help with that."

    return result, classification


def answer_question(question: str, model: str = "gemma3:270m") -> str:
    """Answer a question using Ollama."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": f"Answer this question: {question}",
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    return result["response"]


def process_request(request: str) -> str:
    """Handle a user request with deterministic logic."""
    return f"Processing your request: {request}"


def handle_complaint(complaint: str) -> str:
    """Handle a complaint with deterministic logic."""
    return f"I understand your concern about: {complaint}. Let me escalate this."


if __name__ == "__main__":
    # Test different types of inputs
    test_inputs = [
        "What is machine learning?",
        "Please schedule a meeting for tomorrow",
        "I'm unhappy with the service quality",
    ]

    for user_input in test_inputs:
        print(f"\nInput: {user_input}")
        result, classification = route_based_on_intent(user_input)
        print(
            f"Intent: {classification.intent} (confidence: {classification.confidence})"
        )
        print(f"Reasoning: {classification.reasoning}")
        print(f"Response: {result}")
