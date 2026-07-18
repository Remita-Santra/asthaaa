# nodes.py
import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Force load_dotenv to look in the exact directory of this specific file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path) 

from google import genai
from google.genai import types
from state import ASHAAgentState

# Fallback check: If environment lookup still fails, read it or pass it directly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # ⚠️ TEMPORARY SAFEGUARD: If your .env file is completely broken or unreadable,
    # you can paste your key directly here to get unblocked right away:
    api_key = "AIzaSy..." 

# Initialize the client explicitly passing your key variable
client = genai.Client(api_key=api_key)
