import os
import time
import json
from openai import OpenAI

class LLMClient:
    def __init__(self, model="google/gemini-pro-1.5", temperature=0.0):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        self.temperature = temperature
        
        if not self.api_key:
            # Fallback to OpenAI if OpenRouter not set
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = None # Default OpenAI
            self.model = "gpt-4o" # Fallback model
            print("Warning: OPENROUTER_API_KEY not found. Using OpenAI API directly.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(self, prompt, system_prompt="You are a helpful AI assistant.", max_tokens=1000):
        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error calling LLM (attempt {attempt+1}/{retries}): {e}")
                time.sleep(2 ** attempt)
        return "Error: Failed to generate response."

    def generate_vision(self, prompt, image_paths, system_prompt="You are a helpful AI assistant."):
        # Simplified vision support - implies sending image URLs or base64
        # For this prototype, we'll assume image_paths are handled elsewhere or this specific client 
        # is just for the text recursion part. 
        # TODO: Implement proper vision support if images are available.
        pass
