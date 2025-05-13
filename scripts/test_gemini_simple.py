#!/usr/bin/env python3
from google import genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"API key found: {'Yes' if api_key else 'No'}")

# Initialize the client
client = genai.Client(api_key=api_key)

# Use the simplest possible API call that's known to work
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello, please generate a short Python function that adds two numbers"
)

print(f"\nAPI Response:\n{response.text}")
