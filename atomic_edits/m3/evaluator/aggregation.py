from dataclasses import dataclass

@dataclass
class Thresholds:
    vqa_yes_threshold: float = 0.60
    clip_margin_threshold: float = 0.05

def decide_pass_polarity(
    vqa_yes_conf: float,
    clip_margin: float,
    thresholds: Thresholds,
    expect_present: bool = True,
):
    """
    Polarity-aware rule:

    - If expect_present=True (e.g., "make the shirt blue"):
        PASS when   VQA_yes_conf >= t_vqa   AND   CLIP_margin >= +t_clip
    - If expect_present=False (e.g., "remove the logo"):
        PASS when   (1 - VQA_yes_conf) >= t_vqa   AND   CLIP_margin <= -t_clip

    Otherwise tie-break by the larger normalized gap.
    """
    t_vqa, t_clip = thresholds.vqa_yes_threshold, thresholds.clip_margin_threshold

    vqa_score = vqa_yes_conf if expect_present else (1.0 - vqa_yes_conf)
    clip_ok   = (clip_margin >=  t_clip) if expect_present else (clip_margin <= -t_clip)

    if (vqa_score >= t_vqa) and clip_ok:
        return True, "both_pass"

    if (vqa_score < t_vqa) and (not clip_ok):
        return False, "both_fail"

    # tie-break: sum of signed gaps
    vqa_gap  = (vqa_score - t_vqa)
    clip_gap = (clip_margin - t_clip) if expect_present else (-clip_margin - t_clip)
    return (vqa_gap + clip_gap) > 0, "tie_break"
