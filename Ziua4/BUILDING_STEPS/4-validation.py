"""
Validation: Ensures LLM outputs match predefined data schemas.
This component provides schema validation and structured data parsing to guarantee consistent data formats for downstream code.

More info: https://github.com/ollama/ollama/blob/main/docs/api.md#request-json-mode
"""

import requests
import json
from pydantic import BaseModel, ValidationError


class TaskResult(BaseModel):
    """
    Task information schema.
    More info: https://docs.pydantic.dev
    """

    task: str
    completed: bool
    priority: int


def structured_intelligence(prompt: str, model: str = "gemma3:270m", max_retries: int = 3) -> TaskResult:
    """
    Get structured output from Ollama with validation and retry logic.
    
    Args:
        prompt: The user input to extract information from
        model: The Ollama model to use (default: llama3.2)
        max_retries: Maximum number of retry attempts if validation fails
    
    Returns:
        Validated TaskResult object
    """
    url = "http://localhost:11434/api/chat"
    
    system_prompt = """Extract task information from the user input.
Return ONLY a JSON object with these fields:
- task: string describing the task
- completed: boolean (true/false)
- priority: integer (1=low, 2=medium, 3=high)

Example: {"task": "finish report", "completed": false, "priority": 3}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_retries):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json"  # Request JSON format output
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        try:
            # Parse the response content as JSON
            content = result["message"]["content"]
            data = json.loads(content)
            
            # Validate against Pydantic model
            validated_result = TaskResult(**data)
            return validated_result
            
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"❌ Validation failed (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                # Add feedback to help the model correct itself
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"That output was invalid. Error: {str(e)}. Please provide valid JSON matching the schema."
                })
            else:
                raise ValueError(f"Failed to get valid structured output after {max_retries} attempts")
    
    raise ValueError("Unexpected error in structured_intelligence")


if __name__ == "__main__":
    result = structured_intelligence(
        "I need to complete the project presentation by Friday, it's high priority"
    )
    print("Structured Output:")
    print(result.model_dump_json(indent=2))
    print(f"Extracted task: {result.task}")
