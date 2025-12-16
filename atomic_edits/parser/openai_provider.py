from typing import Dict, Any
from openai import OpenAI

from .prompt_templates import SYSTEM_PROMPT, USER_TEMPLATE
from .json_utils import extract_json, safe_loads

def call_openai(instruction: str,
                model: str = "gpt-4",
                temperature: float = 0.1) -> Dict[str, Any]:
    """
    Call GPT-4 (or family) using the OpenAI Python client.
    Requires OPENAI_API_KEY in environment.
    """
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": USER_TEMPLATE.format(instruction=instruction).strip()},
        ]
    )
    content = resp.choices[0].message.content or ""
    json_text = extract_json(content)
    return safe_loads(json_text)
