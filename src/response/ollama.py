"""
This file is for testing purposes only. Used for interacting with local ollama to assess performance
and explore possibilities.
This will not come under the project context, unless you use ollama as the response generation path.
"""

import requests
from ..utils import load_config

config = load_config()
# this will print "Ollama is running" if ollama server is up
ollama_url = config.get("llm").get("ollama").get("url")
response = requests.get(ollama_url, timeout=10)
print(response._content)


# REFERENCE:
# use /api/generate for single-turn conversations.
# use /api/chat for multi-turn converstations, but you will manually need to maintain the
# conversation history and pass it as an object every single time you call the API.
