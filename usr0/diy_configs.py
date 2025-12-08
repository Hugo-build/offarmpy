"""
DIY Config Function for OffarmLab Python.

This is the Python equivalent of myFuncDIYconfigs.m in MATLAB.
Modify this file to customize configurations at runtime.

The function receives the loaded configs and can modify:
- offsys: Offshore system (bodies, mooring, positions)
- line_types: Line type properties
- env: Environment (wave, current, wind)

Return the modified (or original) objects.
"""

import numpy as np


def diy_configs(configs, offsys, line_types, env):
    """
    User-defined configuration modifications.
    
    This function is called after configs are loaded but before
    force calculators are built, mirroring MATLAB's myFuncDIYconfigs.
    
    Args:
        configs: Full Configs object (for reference)
        offsys: OffSys object to modify
        line_types: LineTypes object to modify
        env: Env object to modify
        
    Returns:
        Tuple of (offsys, line_types, env) - modified or original
    """
    print("  Running DIY configs...")
    
    # =========================================================================
    # MOORING CONFIGURATION (mirrors myFuncDIYconfigs.m)
    # =========================================================================
    
    # Line type indices (1-based like MATLAB, converted to 0-based for Python)
    itypeA = 0  # Anchor line type (index 0 = first line type)
    itypeS = 1  # Shared line type (index 1 = second line type)
    
    # Get line properties for anchor calculation
    if len(line_types) > itypeA:
        thisX2F = line_types[itypeA].X2F  # Horizontal distance to fairlead
        thisZ2F = line_types[itypeA].Z2F  # Vertical distance to fairlead
        print(f"    Anchor line type: X2F={thisX2F:.1f}m, Z2F={thisZ2F:.1f}m")
    else:
        print("    WARNING: Not enough line types defined")
        thisX2F = 1000.0
        thisZ2F = -50.0
    
    # Set line types for all lines (1-based indexing for consistency with MATLAB)
    # Note: The force calculators expect 1-based indices
    n_anchor = offsys.nAnchorLine
    n_shared = offsys.nSharedLine or 0
    
    # These will be used in offsys_dict for force calculators
    # Store as attributes on offsys (we'll need to handle this in the simulation)
    offsys.anchorLineType = [itypeA + 1] * n_anchor  # 1-based
    offsys.sharedLineType = [itypeS + 1] * n_shared  # 1-based
    print(f"    Assigned anchor line type {itypeA + 1} to {n_anchor} anchor lines")
    print(f"    Assigned shared line type {itypeS + 1} to {n_shared} shared lines")
    
    # =========================================================================
    # ANCHOR POSITIONS (mirrors myFuncDIYconfigs.m geometry)
    # =========================================================================
    # Calculate anchor positions based on fairlead positions and line geometry
    # This replicates the MATLAB logic for a 6-cage fish farm layout
    
    fairlead_pos = offsys.fairleadPos_init  # Shape: (3, nFairleads)
    
    # Helper distances
    dis1 = np.sqrt(thisX2F**2 - 500**2) if thisX2F > 500 else thisX2F
    dis2 = 500.0
    
    # Calculate anchor positions (10 anchors for 6-cage system)
    # Each anchor is offset from its corresponding fairlead
    # The pattern follows the MATLAB configuration:
    
    def deg2rad(deg):
        return np.deg2rad(deg)
    
    # Anchor positions relative to fairleads
    # Format: [X, Y, Z] offset + fairlead position
    anchor_offsets = [
        # Anchor 1: 240° from fairlead 4
        (np.array([thisX2F * np.cos(deg2rad(240)), 
                   thisX2F * np.sin(deg2rad(240)), 
                   thisZ2F]), 3),  # fairlead index 4 (0-based: 3)
        
        # Anchor 2: side offset from fairlead 4
        (np.array([dis2, -dis1, thisZ2F]), 3),
        
        # Anchor 3: 300° from fairlead 8
        (np.array([thisX2F * np.cos(deg2rad(300)), 
                   thisX2F * np.sin(deg2rad(300)), 
                   thisZ2F]), 7),  # fairlead index 8 (0-based: 7)
        
        # Anchor 4: side from fairlead 3
        (np.array([-dis1, dis2, thisZ2F]), 2),
        
        # Anchor 5: side from fairlead 5
        (np.array([dis1, dis2, thisZ2F]), 4),
        
        # Anchor 6: side from fairlead 11
        (np.array([-dis1, dis2, thisZ2F]), 10),
        
        # Anchor 7: side from fairlead 13
        (np.array([dis1, dis2, thisZ2F]), 12),
        
        # Anchor 8: 120° from fairlead 18
        (np.array([thisX2F * np.cos(deg2rad(120)), 
                   thisX2F * np.sin(deg2rad(120)), 
                   thisZ2F]), 17),
        
        # Anchor 9: side from fairlead 18
        (np.array([dis2, dis1, thisZ2F]), 17),
        
        # Anchor 10: 60° from fairlead 22
        (np.array([thisX2F * np.cos(deg2rad(60)), 
                   thisX2F * np.sin(deg2rad(60)), 
                   thisZ2F]), 21),
    ]
    
    # Build anchor position matrix
    n_anchors_calc = min(len(anchor_offsets), n_anchor)
    if n_anchors_calc > 0 and fairlead_pos.shape[1] >= max(idx for _, idx in anchor_offsets[:n_anchors_calc]) + 1:
        new_anchor_pos = np.zeros((3, n_anchors_calc))
        for i, (offset, fl_idx) in enumerate(anchor_offsets[:n_anchors_calc]):
            new_anchor_pos[:, i] = offset + fairlead_pos[:, fl_idx]
        
        # Update anchor positions
        offsys.anchorPos_init = new_anchor_pos
        print(f"    Updated {n_anchors_calc} anchor positions")
    else:
        print(f"    Skipping anchor position update (insufficient fairleads or anchors)")
    
    print("  DIY configs complete.")
    
    return offsys, line_types, env


# =========================================================================
# Optional: Additional helper functions can be added below
# =========================================================================

def set_linear_stiffness(offsys, kx=3.7e4, ky=3.7e4):
    """
    Set linear restoring stiffness for all bodies.
    
    This can be used to add horizontal stiffness to floating bodies.
    Uncomment in diy_configs() to enable.
    """
    for body in offsys.floatBodies:
        body.CClin[0, 0] = kx
        body.CClin[1, 1] = ky
    print(f"    Set linear stiffness: kx={kx:.2e}, ky={ky:.2e}")
    return offsys

