"""
Tools: Enables agents to execute specific actions in external systems.
This component provides the capability to make API calls, database updates, file operations, and other practical actions.

More info: https://github.com/ollama/ollama/blob/main/docs/api.md#tools
"""

import json
import requests


def get_weather(latitude: float, longitude: float) -> float:
    """Get current temperature for provided coordinates."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def call_function(name: str, args: dict):
    """Execute a function by name with given arguments."""
    if name == "get_weather":
        return get_weather(**args)
    raise ValueError(f"Unknown function: {name}")


def intelligence_with_tools(prompt: str, model: str = "qwen2.5-coder:14b") -> str:
    """
    Use Ollama with tool calling to answer questions that require external data.
    
    Args:
        prompt: The user's question
        model: The Ollama model to use (default: llama3.2)
    
    Returns:
        The final response after executing tools
    """
    url = "http://localhost:11434/api/chat"

    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for provided coordinates in celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "Latitude coordinate"
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude coordinate"
                        },
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": prompt}]

    # Step 1: Call model with tools
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    
    assistant_message = result["message"]
    messages.append(assistant_message)

    # Step 2: Check if model wants to call tools
    if assistant_message.get("tool_calls"):
        for tool_call in assistant_message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]
            
            # Step 3: Execute the function
            function_result = call_function(function_name, function_args)
            
            # Step 4: Add tool result to messages
            messages.append({
                "role": "tool",
                "content": str(function_result)
            })

        # Step 5: Get final response with tool results
        payload["messages"] = messages
        final_response = requests.post(url, json=payload)
        final_response.raise_for_status()
        final_result = final_response.json()
        
        return final_result["message"]["content"]
    
    # If no tool calls, return the initial response
    return assistant_message.get("content", "")


if __name__ == "__main__":
    result = intelligence_with_tools(prompt="What's the weather like in Paris today?")
    print("Tool Calling Output:")
    print(result)
