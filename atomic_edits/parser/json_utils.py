import json, re
from typing import Any, Dict

def extract_json(text: str) -> str:
    """
    Extract the largest JSON-looking block from text.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    # Try array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return "{ \"sub_instructions\": " + text[start:end+1].strip() + " }"
    return text.strip()

def safe_loads(json_text: str) -> Dict[str, Any]:
    """
    Be tolerant of minor JSON glitches (trailing commas, stray backticks).
    """
    cleaned = json_text.replace("```", "").strip()
    cleaned = re.sub(r",\\s*([}\\]])", r"\\1", cleaned)
    return json.loads(cleaned)
