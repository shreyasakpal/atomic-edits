from typing import List
from .schema import SubInstruction, ParseResult
from .ollama_provider import call_ollama
from .openai_provider import call_openai
import re

# map synonyms to canonical actions
_ACTION_MAP = {
    "recolor": "change_color",
    "colorize": "change_color",
    "tint": "change_color",
    "brighten": "increase",   # with attribute 'brightness'
    "darken": "decrease",     # with attribute 'brightness'
    "remove_text": "remove",
    "delete_text": "remove",
    "erase_text": "remove",
    "delete": "remove",
    "erase": "remove",
}

# when these appear, the target is a localized region (if not global)
_REGION_KEYWORDS = {"shirt","hair","logo","text","sticker","decal","eyes","background",
                    "sky","mug","cup","laptop","bag","sign","car","face","moustache",
                    "beard","pants","shoe","sleeve","desk","table","wall"}

_PROP_PATTERNS = [
    (re.compile(r"^(?:color|hue)\s+of\s+(.+)$", re.I), "color"),
    (re.compile(r"^(?:brightness|light(?:ing|ness)?)\s+of\s+(.+)$", re.I), "brightness"),
    (re.compile(r"^(?:sharpness|detail)\s+of\s+(.+)$", re.I), "sharpness"),
    (re.compile(r"^(?:contrast)\s+of\s+(.+)$", re.I), "contrast"),
    (re.compile(r"^(?:saturation)\s+of\s+(.+)$", re.I), "saturation"),
]

def _normalize_step(si: SubInstruction) -> SubInstruction:
    act = (si.action or "").strip().lower()
    obj = (si.object or "").strip()
    attr = (si.attribute or None)
    val = (si.value or None)

    # 1) canonicalize action
    if act in _ACTION_MAP:
        act = _ACTION_MAP[act]
        # 'brighten'/'darken' imply brightness if missing
        if act in ("increase","decrease") and not attr:
            attr = "brightness"

    # 2) repair patterns like "sharpness of the background"
    if obj:
        for rx, prop in _PROP_PATTERNS:
            m = rx.match(obj)
            if m:
                obj = m.group(1).strip()
                if not attr:
                    attr = prop
                break

    # 3) ensure object is a bare noun phrase (no leading 'the ')
    obj = re.sub(r"^\bthe\b\s+", "", obj, flags=re.I).strip()

    # 4) infer requires_region if missing/false but object is region-like (and not 'image'/'photo')
    rr = bool(si.requires_region)
    if not rr:
        low = obj.lower()
        if any(k in low for k in _REGION_KEYWORDS) and low not in {"image", "photo", "picture"}:
            rr = True

    # 5) write back
    si.action = act
    si.object = obj
    si.attribute = attr
    si.value = val
    si.requires_region = rr
    return si

def _to_parse_result(original_text: str, data: dict) -> ParseResult:
    sub_instructions: List[SubInstruction] = []
    for i, step in enumerate(data.get("sub_instructions", []), start=1):
        si = SubInstruction(
            action=str(step.get("action", "")).strip(),
            object=str(step.get("object", "")).strip(),
            attribute=(step.get("attribute") or None),
            value=(step.get("value") or None),
            qualifiers=[str(q) for q in step.get("qualifiers", [])],
            requires_region=bool(step.get("requires_region", False)),
            order=int(step.get("order", i)),
        )
        sub_instructions.append(_normalize_step(si))
    return ParseResult(
        original_text=original_text,
        sub_instructions=sub_instructions,
        ambiguities=[str(a) for a in data.get("ambiguities", [])],
        notes=str(data.get("notes", "")),
    )

def parse_with_ollama(instruction: str, ollama_model: str, url: str = "http://localhost:11434") -> ParseResult:
    data = call_ollama(instruction=instruction, model=ollama_model, url=url)
    return _to_parse_result(instruction, data)

def parse_with_openai(instruction: str, openai_model: str = "gpt-4") -> ParseResult:
    data = call_openai(instruction=instruction, model=openai_model)
    return _to_parse_result(instruction, data)
