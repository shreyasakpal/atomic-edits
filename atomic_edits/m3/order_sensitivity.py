import json
import random
import subprocess
from pathlib import Path
from itertools import permutations
import numpy as np

def test_order_sensitivity(example="artifacts/examples/test_animals", num_permutations=3):
    """Test if instruction order affects results"""
    
    # Load original instructions
    data = json.load(open(Path(example) / "sub_instructions.json"))
    instructions = data['sub_instructions']
    
    if len(instructions) < 2:
        print("Need at least 2 instructions for order testing")
        return
    
    results = []
    orig_image = Path(example) / "original.png"
    
    # Test different orderings
    if len(instructions) <= 3:
        # Test all permutations for small sets
        orders = list(permutations(range(len(instructions))))
    else:
        # Random sample for larger sets
        orders = [list(range(len(instructions)))]  # Original
        for _ in range(num_permutations - 1):
            order = list(range(len(instructions)))
            random.shuffle(order)
            orders.append(order)
    
    for i, order in enumerate(orders):
        print(f"Testing order {i+1}: {order}")
        
        # Create reordered instructions
        reordered = [instructions[j] for j in order]
        
        # Save as temporary parse file
        temp_parse = {
            'sub_instructions': [
                {
                    'action': inst['text'].split()[0],
                    'object': inst['text'].split()[1] if len(inst['text'].split()) > 1 else '',
                    'value': inst.get('value', ''),
                    'order': idx + 1
                }
                for idx, inst in enumerate(reordered)
            ]
        }
        
        temp_dir = Path(f"artifacts/order_test/order_{i}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        parse_path = temp_dir / "parse.json"
        json.dump(temp_parse, open(parse_path, "w"))
        
        # Create plan
        plan = {
            'image': str(orig_image),
            'seed': 42,
            'steps': 10
        }
        plan_path = temp_dir / "plan.json"
        json.dump(plan, open(plan_path, "w"))
        
        # Run edit with this order
        cmd = [
            "python", "-m", "atomic_edits.editor.cli_region",
            "--image", str(orig_image),
            "--json", str(plan_path),
            "--outdir", str(temp_dir / "edits"),
            "--steps", "10",
            "--seed", "42"
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        # Store result path
        results.append({
            'order_id': i,
            'order': order,
            'instruction_sequence': [inst['text'] for inst in reordered],
            'output_path': str(temp_dir / "edits" / "final.png")
        })
    
    # Compare all results
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    
    comparison = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            if Path(results[i]['output_path']).exists() and Path(results[j]['output_path']).exists():
                img1 = np.array(Image.open(results[i]['output_path']).convert('L')) / 255.0
                img2 = np.array(Image.open(results[j]['output_path']).convert('L')) / 255.0
                
                similarity = ssim(img1, img2, data_range=1.0)
                comparison.append({
                    'order1': results[i]['order'],
                    'order2': results[j]['order'],
                    'ssim': float(similarity),
                    'similar': similarity > 0.95
                })
    
    return {
        'results': results,
        'comparisons': comparison,
        'order_matters': any(c['ssim'] < 0.95 for c in comparison)
    }

if __name__ == "__main__":
    study = test_order_sensitivity()
    json.dump(study, open("artifacts/m3/order_sensitivity.json", "w"), indent=2)
    
    print(f"\n=== ORDER SENSITIVITY RESULTS ===")
    print(f"Order affects results: {study['order_matters']}")
    for comp in study['comparisons']:
        print(f"  Orders {comp['order1']} vs {comp['order2']}: SSIM={comp['ssim']:.3f}")