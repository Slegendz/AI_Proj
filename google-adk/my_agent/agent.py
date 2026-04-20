import os
from google.adk.agents import Agent
from dotenv import load_dotenv
from .prompt import SYSTEM_PROMPT

load_dotenv()

assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY not found!"

root_agent = Agent(
    name = 'greetAgent',
    model = "gemini-2.5-flash",
    instruction=SYSTEM_PROMPT,
    description="A high-energy digital concierge that greets every interaction with contagious curiosity and proactive warmth.",
)