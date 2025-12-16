import requests
from typing import Dict, Any

from .prompt_templates import SYSTEM_PROMPT, USER_TEMPLATE
from .json_utils import extract_json, safe_loads

def call_ollama(instruction: str,
                model: str,
                url: str = "http://localhost:11434",
                temperature: float = 0.1,
                timeout: int = 120) -> Dict[str, Any]:
    """
    Call a local open-source LLM via Ollama's /api/chat.
    We set format="json" so the model returns strict JSON.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": USER_TEMPLATE.format(instruction=instruction).strip()}
        ],
        "stream": False,
        "options": {"temperature": temperature},
        "format": "json"
    }
    resp = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    json_text = extract_json(content)
    return safe_loads(json_text)
