import json
import os
import random
from pathlib import Path

def create_rand_sn_config(json_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "floatBody" not in data:
        print("Error: 'floatBody' not found in JSON.")
        return

    # Base solidity ratio
    base_sn = 0.112
    # Variation range (e.g., 0.1 for +/- 10%)
    variation_range = 0.1

    for i, body in enumerate(data["floatBody"]):
        print(f"Processing body {i+1}: {body.get('name', 'unnamed')}")
        
        # Determine number of nodes
        if "attachNodePos" in body:
            pos_data = body["attachNodePos"]
        elif "attachNodePos_init" in body:
            pos_data = body["attachNodePos_init"]
        else:
            print(f"  Warning: No attachNodePos found for body {i+1}. Skipping.")
            continue
            
        # Assuming pos_data is [x_coords, y_coords, z_coords]
        if not pos_data or not isinstance(pos_data, list):
             print(f"  Warning: Invalid attachNodePos format for body {i+1}. Skipping.")
             continue
             
        n_nodes = len(pos_data[0])
        print(f"  Found {n_nodes} nodes.")
        
        # Initialize Sn array
        attach_node_sn = [0.0] * n_nodes
        
        el_types = body.get("ElType", [])
        el_indices = body.get("ElIndex", [])
        
        if len(el_types) != len(el_indices):
            print(f"  Warning: ElType and ElIndex length mismatch for body {i+1}.")
        
        for etype, indices in zip(el_types, el_indices):
            if len(indices) != 2:
                print(f"  Warning: Invalid index range {indices} for type {etype}.")
                continue
                
            start_idx_1based = indices[0]
            end_idx_1based = indices[1]
            
            # Convert to 0-based
            start_idx = start_idx_1based - 1
            end_idx = end_idx_1based
            
            count = 0
            for k in range(start_idx, end_idx):
                if k < len(attach_node_sn):
                    if etype == "net":
                        # Create small random variation
                        # random.uniform(a, b) returns a random floating point number N such that a <= N <= b
                        variation = random.uniform(-variation_range, variation_range)
                        val = base_sn * (1.0 + variation)
                    elif etype == "cylinder":
                        val = 0.0
                    else:
                        val = 0.0
                        
                    attach_node_sn[k] = val
                    count += 1
            
            print(f"  Processed {count} nodes for type '{etype}' (indices {start_idx_1based}-{end_idx_1based})")

        # Add to body
        body["attachNodeSn"] = attach_node_sn
    
    # Save new JSON
    output_path = str(json_path).replace(".json", "_randSn.json")
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    # Same config file as add_sn_config.py
    json_file = script_dir / "examples/usr0/sys6cage_DgEls_wT1st.json"
    
    if not json_file.exists():
        # Try absolute path if relative fails
        json_file = Path("/Users/lykkefish/Library/CloudStorage/OneDrive-UniversitetetiStavanger/UiS_PhD_workingspace/git_workspace/MC_fishCageArray/offarmpy/examples/usr0/sys6cage_DgEls_wT1st.json")
    
    if json_file.exists():
        create_rand_sn_config(json_file)
    else:
        print(f"Could not locate the JSON file at {json_file}")
