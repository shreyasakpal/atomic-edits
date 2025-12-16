from typing import Dict

def build_edit_prompt(step: Dict) -> str:
    action   = (step.get("action") or "").strip()
    obj      = (step.get("object") or "image").strip()
    attr     = (step.get("attribute") or "").strip()
    value    = (step.get("value") or "").strip()
    quals    = " ".join(step.get("qualifiers", [])) if step.get("qualifiers") else ""
    where    = f" {quals}".strip()

    def _target(default="image"):
        return obj if obj and obj.lower() != "null" else default

    a = action.lower()

    if a in {"change_color","recolor"} and value:
        return f"change the {_target()} color to {value}{(' ' + quals) if quals else ''}"
    if a == "remove":
        return f"remove the {_target()}{(' ' + quals) if quals else ''}"
    if a == "add":
        return f"add {_target()}{(' ' + quals) if quals else ''}"
    if a == "replace":
        return f"replace the {_target()}{(' ' + quals) if quals else ''}"
    if a in {"increase","decrease"} and attr:
        verb = "increase" if a == "increase" else "decrease"
        return f"{verb} {attr} of the {_target('image')}{(' ' + quals) if quals else ''}"
    if a == "brighten":
        return f"brighten the {_target('image')}{(' ' + quals) if quals else ''}"
    if a == "darken":
        return f"darken the {_target('image')}{(' ' + quals) if quals else ''}"
    if a == "sharpen":
        return f"sharpen the {_target('image')}{(' ' + quals) if quals else ''}"
    if a == "blur":
        return f"blur the {_target('background')}{(' ' + quals) if quals else ''}"
    if a == "crop":
        return f"crop tighter on the {_target('subject')}{(' ' + quals) if quals else ''}"
    if a == "resize":
        return f"resize the {_target('subject')}{(' ' + quals) if quals else ''}"
    if a == "rotate":
        return f"straighten or rotate the {_target('subject')}{(' ' + quals) if quals else ''}"
    if a in {"add_text","remove_text"}:
        return f"{'add' if a=='add_text' else 'remove'} text {('\"'+value+'\" ') if value else ''}{('on ' + _target()) if _target()!='image' else ''}"

    # fallback – keep it short and specific
    return f"{action} the {_target()}{(' ' + quals) if quals else ''}"
