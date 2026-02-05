"""
Recovery: Manages failures and exceptions gracefully in agent workflows.
This component implements retry logic, fallback processes, and error handling to ensure system resilience.
"""

import requests
import json
import time
from typing import Optional
from pydantic import BaseModel, ValidationError


class UserInfo(BaseModel):
    name: str
    email: str
    age: Optional[int] = None  # Optional field


def get_user_info_with_retry(prompt: str, model: str = "llama3.2", max_retries: int = 3) -> UserInfo:
    """
    Extract user information with retry logic and exponential backoff.
    
    Args:
        prompt: The text to extract information from
        model: The Ollama model to use
        max_retries: Maximum number of retry attempts
    
    Returns:
        UserInfo object
    """
    url = "http://localhost:11434/api/chat"
    
    system_prompt = """Extract user information from the text.
Return ONLY a JSON object with these fields:
- name: string with the person's name
- email: string with the email address
- age: integer with age (or null if not mentioned)

Example: {"name": "John Doe", "email": "john@example.com", "age": 30}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0}
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            content = result["message"]["content"]
            data = json.loads(content)
            
            return UserInfo(**data)
            
        except (requests.RequestException, json.JSONDecodeError, ValidationError) as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                # Exponential backoff: wait 1s, 2s, 4s...
                wait_time = 2 ** attempt
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
    
    raise Exception("Unexpected error in get_user_info_with_retry")


def resilient_intelligence(prompt: str) -> str:
    """
    Extract user information with graceful error handling and fallbacks.
    
    Args:
        prompt: The text to extract information from
    
    Returns:
        Formatted user information or fallback message
    """
    try:
        # Try to get structured user information
        user_info = get_user_info_with_retry(prompt)
        user_data = user_info.model_dump()
        
        # Try to access age field and check if it's valid
        age = user_data["age"]
        if age is None:
            raise ValueError("Age is None")
        age_info = f"User is {age} years old"
        return age_info

    except (KeyError, TypeError, ValueError) as e:
        print(f"❌ Age not available ({str(e)}), using fallback info...")

        # Fallback to available information
        try:
            return f"User {user_data['name']} has email {user_data['email']}"
        except:
            return "Unable to extract complete user information"

    except Exception as e:
        print(f"❌ Complete failure: {str(e)}")
        # Ultimate fallback
        return "Service temporarily unavailable. Please try again later."


if __name__ == "__main__":
    result = resilient_intelligence(
        "My name is John Smith and my email is john@example.com"
    )
    print("Recovery Output:")
    print(result)
