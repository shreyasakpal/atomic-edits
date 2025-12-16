#!/usr/bin/env python3
"""
Prepare all examples for M3 evaluation.
Works with your actual folder structure: artifacts/examples/*/
"""

import json
import shutil
from pathlib import Path

def prepare_example(source_dir: Path) -> bool:
    """Prepare one example for M3."""
    
    # Check if we have the required files
    if not (source_dir / "edits/final.png").exists():
        print(f"⚠️  Skipping {source_dir.name}: no final.png")
        return False
    
    if not (source_dir / "parsed/parse.json").exists():
        print(f"⚠️  Skipping {source_dir.name}: no parse.json")
        return False
    
    # 1. Copy/link edited image (M3 expects "edited.png" in the root)
    edited_src = source_dir / "edits/final.png"
    edited_dst = source_dir / "edited.png"
    if not edited_dst.exists():
        shutil.copy(edited_src, edited_dst)

    masks_src = source_dir / "edits/masks"
    if masks_src.exists():
        masks_dst = source_dir / "masks"
        if not masks_dst.exists():
            shutil.copytree(masks_src, masks_dst)
        print(f"   Copied {len(list(masks_src.glob('*.png')))} masks")
    
    # 2. Get original image from plan and copy to root
    original_dst = source_dir / "original.png"
    if not original_dst.exists() and (source_dir / "plans/plan.json").exists():
        try:
            plan = json.load(open(source_dir / "plans/plan.json"))
            orig_path = Path(plan.get('image', ''))
            if orig_path.exists():
                shutil.copy(orig_path, original_dst)
        except Exception as e:
            print(f"⚠️  Could not copy original for {source_dir.name}: {e}")
    
    # 3. Convert parse.json to M3-compatible sub_instructions.json
    parse_data = json.load(open(source_dir / "parsed/parse.json"))
    
    m3_subs = []
    for i, s in enumerate(parse_data.get('sub_instructions', []), 1):
        action = s.get('action', '')
        obj = s.get('object', 'object')
        value = s.get('value', '')
        attr = s.get('attribute', '')
        
        # Generate VQA question based on action type
        if action == 'remove':
            vqa = f"Is the {obj} removed from the image?"
            expect = False
        elif action == 'change_color' and value:
            vqa = f"Is the {obj} {value}?"
            expect = True
        elif action in ('increase', 'decrease'):
            direction = 'increased' if action == 'increase' else 'decreased'
            vqa = f"Is the {attr or 'brightness'} of the {obj} {direction}?"
            expect = True
        elif action == 'blur':
            vqa = f"Is the {obj} blurred?"
            expect = True
        elif action == 'brighten':
            vqa = f"Is the {obj} brighter?"
            expect = True
        elif action == 'darken':
            vqa = f"Is the {obj} darker?"
            expect = True
        else:
            vqa = f"Is the {obj} modified correctly?"
            expect = True
        
        # Target phrase for CLIP
        if action == 'change_color' and value:
            target_phrase = f"{value} {obj}"
        else:
            target_phrase = obj
        
        m3_subs.append({
            'id': f's{i}',
            'order': s.get('order', i),
            'text': s.get('text', f"{action} {obj}").strip(),
            'action': action,
            'object': obj,
            'attribute': attr,
            'value': value,
            'qualifiers': s.get('qualifiers', []),
            'requires_region': s.get('requires_region', False),
            'vqa_question': vqa,
            'target_phrase': target_phrase,
            'expect_present': expect
        })
    
    m3_data = {
        'original_text': parse_data.get('original_text', ''),
        'sub_instructions': m3_subs,
        'ambiguities': parse_data.get('ambiguities', []),
        'notes': parse_data.get('notes', '')
    }
    
    output_path = source_dir / "sub_instructions.json"
    json.dump(m3_data, open(output_path, 'w'), indent=2)
    
    print(f"✅ {source_dir.name}")
    return True

def main():
    """Process all examples in artifacts/examples/"""
    examples_root = Path("artifacts/examples")
    
    if not examples_root.exists():
        print(f"❌ Not found: {examples_root}")
        return
    
    count = 0
    for example_dir in sorted(examples_root.iterdir()):
        if example_dir.is_dir():
            if prepare_example(example_dir):
                count += 1
    
    print(f"\n✅ Prepared {count} examples for M3")
    print(f"📁 Ready to run: python -m atomic_edits.m3.evaluate")

if __name__ == "__main__":
    main()