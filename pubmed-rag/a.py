import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print("--- Models supporting 'embedContent' ---")
for m in client.models.list():
    if 'embedContent' in m.supported_actions:
        print(f"Model Name: {m.name}")
        print(f"Display Name: {m.display_name}")
        print(f"Description: {m.description}\n")