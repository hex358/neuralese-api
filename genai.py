from google import genai
from google.genai import types
import os

client = genai.Client(api_key="AIzaSyDuYrpq7TCEHWGrCUJ47hTJQd2pkZQAzXk")

with open("prompt", "r+") as f:
    prompt = f.read()
    prompt += "\n\nПользователь спрашивает: пожалуйста, создай нейросеть которая распознает числа"

# Stream the response as it’s generated (prints chunks immediately)
for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt,
        # Optional: control the 'reasoning' budget; -1 lets the model decide dynamically.
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1)
        )
    ):
    if chunk.text:
        print(chunk.text or "", end="", flush=True)
    