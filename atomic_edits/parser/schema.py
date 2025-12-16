from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

CANONICAL_ACTIONS = [
    # prefer 'change_color' (parser will normalize), keep 'recolor' for backward-compat only
    "change_color", "add", "remove", "replace", "recolor",
    "increase", "decrease", "move", "resize", "crop",
    "blur", "sharpen", "brighten", "darken", "rotate",
    "inpaint", "outpaint", "add_text", "remove_text"
]



@dataclass
class SubInstruction:
    action: str
    object: str
    attribute: Optional[str] = None
    value: Optional[str] = None
    qualifiers: List[str] = field(default_factory=list)
    requires_region: bool = False
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, str) and not v.strip():
                d[k] = None
        return d

@dataclass
class ParseResult:
    original_text: str
    sub_instructions: List[SubInstruction]
    ambiguities: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "sub_instructions": [si.to_dict() for si in self.sub_instructions],
            "ambiguities": self.ambiguities,
            "notes": self.notes
        }
