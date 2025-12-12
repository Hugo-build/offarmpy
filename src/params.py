from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path
import numpy as np
import json
import os


def printTypeOfAttrs(obj):
    """Debug helper: print attribute types of an object."""
    for attr, value in obj.__dict__.items():
        print(f"{attr}: {type(value)}")


@dataclass
class Current:
    vel:      list[float]
    zlevel:   list[float]
    wakeRatio:float
    propDir:  float

    def __post_init__(self):
        # Validate data
        if len(self.vel) != len(self.zlevel):
            raise ValueError("vel and zlevel arrays must have the same length")
        if not 0 <= self.wakeRatio <= 1:
            raise ValueError("wakeRatio must be between 0 and 1")
        if not 0 <= self.propDir <= 360:
            raise ValueError("propDir must be between 0 and 360 degrees")
        
        # ndarrayfy lists
        self.vel = np.array(self.vel).reshape(-1,1)
        self.zlevel = np.array(self.zlevel).reshape(-1,1)


@dataclass
class Wave:
    """
    Wave parameters dataclass.
    
    Can be initialized with just basic parameters (Hs, Tp, etc.) and will
    compute derived fields (spectrum, wave field) automatically, mirroring
    MATLAB's setSpecData and setWaveField functions.
    """
    # Basic parameters (required)
    specType: str
    Hs:       float
    Tp:       float
    gamma:    float
    propDir:  float
    dfreq:    float
    domega:   float
    
    # Derived parameters (optional - computed if not provided)
    omegaVec: Optional[list[float]] = None
    dirVec:   Optional[list[float]] = None
    freqVec:  Optional[list[float]] = None  # frequency vector (Hz)
    k:        Optional[list[float]] = None
    kX:       Optional[list[float]] = None
    kY:       Optional[list[float]] = None
    Szz:      Optional[list[float]] = None
    SzzDir:   Optional[list[float]] = None

    ZaCal:    Optional[list[float]] = None
    omegaCal: Optional[list[float]] = None
    phaseCal: Optional[list[float]] = None
    kXCal:    Optional[list[float]] = None
    kYCal:    Optional[list[float]] = None
    
    # Random seed for reproducible wave phases
    seed:     Optional[int] = None

    def __post_init__(self):
        g = 9.81  # gravity acceleration
        
        # Helper to check if a value is empty (None or empty list/array)
        def is_empty(val):
            if val is None:
                return True
            if hasattr(val, '__len__') and len(val) == 0:
                return True
            return False
        
        # If omegaVec not provided, generate from domega (like MATLAB setSpecData)
        if is_empty(self.omegaVec):
            self.omegaVec = np.arange(self.domega, 1.5, self.domega).reshape(-1, 1)
        else:
            self.omegaVec = np.array(self.omegaVec).reshape(-1, 1)
        
        # If dirVec not provided, default to single direction [0]
        if is_empty(self.dirVec):
            self.dirVec = np.array([0.0]).reshape(-1, 1)
        else:
            self.dirVec = np.array(self.dirVec).reshape(-1, 1)
        
        m = len(self.omegaVec)
        n = len(self.dirVec)
        
        # Calculate wave number k = omega^2 / g (deep water dispersion)
        if is_empty(self.k):
            self.k = (self.omegaVec ** 2 / g)
        else:
            self.k = np.array(self.k).reshape(-1, 1)
        
        # Calculate kX and kY (wave number components)
        propDir_rad = np.deg2rad(self.propDir)
        if is_empty(self.kX):
            self.kX = self.k * np.cos(self.dirVec.T + propDir_rad)  # (m x n)
        else:
            self.kX = np.array(self.kX).reshape(-1, 1)
        
        if is_empty(self.kY):
            self.kY = self.k * np.sin(self.dirVec.T + propDir_rad)  # (m x n)
        else:
            self.kY = np.array(self.kY).reshape(-1, 1)
        
        # Calculate Szz (spectrum) using JONSWAP or other spectrum types
        if is_empty(self.Szz):
            self.Szz = self._calculate_spectrum()
        else:
            self.Szz = np.array(self.Szz).reshape(-1, 1)
        
        # Calculate directional spectrum SzzDir = Szz * ones(1,n) / n
        if is_empty(self.SzzDir):
            self.SzzDir = self.Szz * np.ones((1, n)) / n  # (m x n)
        else:
            self.SzzDir = np.array(self.SzzDir).reshape(-1, 1)
        
        # Calculate Cal attributes (wave field) - mirrors setWaveField.m
        if is_empty(self.ZaCal) or is_empty(self.omegaCal) or is_empty(self.phaseCal):
            self._set_wave_field()
        else:
            self.ZaCal = np.array(self.ZaCal).reshape(-1, 1)
            self.omegaCal = np.array(self.omegaCal).reshape(-1, 1)
            self.phaseCal = np.array(self.phaseCal).reshape(-1, 1)
            self.kXCal = np.array(self.kXCal).reshape(-1, 1)
            self.kYCal = np.array(self.kYCal).reshape(-1, 1)
    
    def _calculate_spectrum(self) -> np.ndarray:
        """
        Calculate wave spectrum based on specType.
        Mirrors MATLAB waveSpectrum function.
        """
        g = 9.81
        omega = self.omegaVec.flatten()
        
        if self.specType == 'Jonswap':
            Shh = self._jonswap_spectrum(omega, g)
        elif self.specType == 'PM':
            Shh = self._pm_spectrum(omega, g)
        else:
            # Default to JONSWAP
            Shh = self._jonswap_spectrum(omega, g)
        
        return Shh.reshape(-1, 1)
    
    def _jonswap_spectrum(self, omega: np.ndarray, g: float) -> np.ndarray:
        """JONSWAP spectrum calculation (mirrors MATLAB waveSpectrum case 1)."""
        wp = 2 * np.pi / self.Tp
        gamma = self.gamma
        
        # DNV recommended gamma calculation if outside validity range
        if gamma < 1 or gamma > 7:
            if gamma != 0:
                print("Warning: gamma value outside validity range, using DNV formula")
            k_param = 2 * np.pi / (wp * np.sqrt(self.Hs))
            if k_param <= 3.6:
                gamma = 5.0
            elif k_param <= 5.0:
                gamma = np.exp(5.75 - 1.15 * k_param)
            else:
                gamma = 1.0
        
        alpha = 0.2 * self.Hs**2 * wp**4 / g**2
        
        Shh = np.zeros_like(omega)
        for i, w in enumerate(omega):
            if w <= 0:
                continue
            sigma = 0.07 if w < wp else 0.09
            S1 = alpha * g**2 * (w ** -5) * np.exp(-(5/4) * (wp/w)**4)
            S2 = gamma ** (np.exp(-((w - wp)**2) / (2 * (sigma * wp)**2)))
            Shh[i] = S1 * S2
        
        return Shh
    
    def _pm_spectrum(self, omega: np.ndarray, g: float) -> np.ndarray:
        """Pierson-Moskowitz spectrum calculation."""
        wp = 2 * np.pi / self.Tp
        alpha = 0.0081  # Phillips constant
        
        Shh = np.zeros_like(omega)
        for i, w in enumerate(omega):
            if w <= 0:
                continue
            Shh[i] = (alpha * g**2 / w**5) * np.exp(-(5/4) * (wp/w)**4)
        
        # Scale to match Hs
        m0 = np.trapz(Shh, omega)
        if m0 > 0:
            Shh *= (self.Hs / 4)**2 / m0
        
        return Shh
    
    def _set_wave_field(self):
        """
        Set wave field calculation attributes.
        Mirrors MATLAB setWaveField function.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        m = len(self.omegaVec)
        n = len(self.dirVec)
        
        # ZaMat = sqrt(2 * SzzDir * domega)  - wave amplitude matrix (m x n)
        ZaMat = np.sqrt(2 * self.SzzDir * self.domega)
        
        # omegaMat = repmat(omegaVec, 1, n) - omega matrix (m x n)
        omegaMat = np.tile(self.omegaVec, (1, n))
        
        # phaseMat = 2*pi*rand([m,n]) - random phase matrix (m x n)
        phaseMat = 2 * np.pi * np.random.rand(m, n)
        
        # Flatten to Cal vectors (column vectors of size m*n x 1)
        self.ZaCal = ZaMat.reshape(-1, 1, order='F')      # Fortran order like MATLAB
        self.omegaCal = omegaMat.reshape(-1, 1, order='F')
        self.phaseCal = phaseMat.reshape(-1, 1, order='F')
        self.kXCal = self.kX.reshape(-1, 1, order='F')
        self.kYCal = self.kY.reshape(-1, 1, order='F')


@dataclass
class Wind:
    tFrames: list[float]
    vel:     list[float]
    meshPos: list[float]
    propDir: float


@dataclass
class Env:
    current: Current
    wave: Wave
    wind: Optional[Wind] = None

    @classmethod
    def from_dict(cls, dict):
        return cls(
            current=Current(**dict["current"]),
            wave=Wave(**dict["wave"]),
            wind=Wind(**dict["wind"]) if "wind" in dict else None
        )


@dataclass 
class WaveTrans1st:
    dirVec:           list[float]   # direction vector
    Ndir:             int          # number of direction
    periodVec:        list[float]   # period vector
    omegaVec:         list[float]   # omega vector
    Nomega:           int          # number of omega
    Xamp:             list[list[list[float]]]   # amplitude of each direction
    phase:            list[list[list[float]]]   # phase of each direction
    
    def __post_init__(self):
        # ndarray-ify lists
        self.dirVec = np.array(self.dirVec).reshape(-1,1)
        self.periodVec = np.array(self.periodVec).reshape(-1,1)
        self.omegaVec = np.array(self.omegaVec).reshape(-1,1)
        self.Xamp = np.array(self.Xamp)
        self.phase = np.array(self.phase)


@dataclass
class FloatBody:
    """
    Floating body properties from offshore system JSON.
    
    Note on from_dict vs **dict:
    - Use `FloatBody(**data)` when JSON keys match field names exactly and types are correct
    - Use `FloatBody.from_dict(data)` when you need to:
      - Handle missing keys with defaults
      - Transform/rename keys
      - Parse nested objects
    """
    name:             str
    MM:               list[list[float]]     # mass matrix (6x6)
    MMaInf:           list[list[float]]     # added mass matrix (6x6)
    BBlin:            list[list[float]]     # linear damping matrix (6x6)
    CClin:            list[list[float]]     # linear restoring matrix (6x6)
    ZCOG:             float                 # center of gravity z-coordinate
    posGlobal:        list[float]           # global position [x, y, z]
    fairleadPos_init: list[list[float]]     # initial fairlead positions (3 x nFairlead)
    fairleadIndex:    list[int]             # indices of fairleads

    calDoF:             int                 # number of DoFs to calculate
    AlineSlave:         list[int]           # indices of attached anchor lines
    SlineSlave:         list[list[int]]     # indices of shared lines

    attachNodePos_init: list[list[float]]   # initial node positions (3 x nNodes)
    attachNodePos:      list[list[float]]   # current node positions (3 x nNodes)
    attachNodeVec:      list[list[float]]   # node unit vectors (3 x nNodes)
    attachNodeArea:     list[float]         # projected area per node
    attachNodeCd:       list[float]         # drag coefficient per node
    
    ElType:           list[str]             # element types ["cylinder", "net", ...]
    ElIndex:          list[list[int]]       # element index ranges

    # Optional fields
    type2node_L:      Optional[dict] = None  # left index for each element type
    type2node_R:      Optional[dict] = None  # right index for each element type
    waveTrans1st:     Optional[dict] = None  # first order wave transfer function
    attachNodeSn:     Optional[list[float]] = None  # solidity ratio per node

    def __post_init__(self):
        # ndarray-ify lists
        self.MM = np.array(self.MM)
        self.MMaInf = np.array(self.MMaInf)
        self.BBlin = np.array(self.BBlin)
        self.CClin = np.array(self.CClin)
        self.posGlobal = np.array(self.posGlobal).flatten()
        self.fairleadPos_init = np.array(self.fairleadPos_init)

        self.attachNodePos_init = np.array(self.attachNodePos_init)
        self.attachNodeVec = np.array(self.attachNodeVec)
        self.attachNodeArea = np.array(self.attachNodeArea).flatten()
        self.attachNodeCd = np.array(self.attachNodeCd).flatten()
        
        if self.attachNodeSn is not None:
            self.attachNodeSn = np.array(self.attachNodeSn).flatten()

        # Build type2node mappings from ElType and ElIndex
        if self.type2node_L is None:
            self.type2node_L = {}
            self.type2node_R = {}
            for i, el_type in enumerate(self.ElType):
                if len(self.ElIndex[i]) >= 2:
                    self.type2node_L[el_type] = self.ElIndex[i][0]
                    self.type2node_R[el_type] = self.ElIndex[i][1]

        # Validation
        if self.fairleadPos_init.shape != (3, len(self.fairleadIndex)):
            raise ValueError("fairleadPos_init must be a 3xNfairlead matrix")
        if self.attachNodePos_init.shape[0] != 3:
            raise ValueError("attachNodePos_init must be a 3xNattachNode matrix")
        if self.attachNodeVec.shape[0] != 3:
            raise ValueError("attachNodeVec must be a 3xNattachNode matrix")

    @classmethod
    def from_dict(cls, data: dict) -> "FloatBody":
        """
        Create FloatBody from dictionary with flexible key handling.
        
        Use this when JSON structure might vary or have optional fields.
        """
        return cls(
            name=data.get("name", "unnamed"),
            MM=data.get("MM", []),
            MMaInf=data.get("MMaInf", []),
            BBlin=data.get("BBlin", []),
            CClin=data.get("CClin", []),
            ZCOG=data.get("ZCOG", 0.0),
            posGlobal=data.get("posGlobal", [0, 0, 0]),
            fairleadPos_init=data.get("fairleadPos_init", []),
            fairleadIndex=data.get("fairleadIndex", []),
            calDoF=data.get("calDoF", 6),
            AlineSlave=data.get("AlineSlave", []),
            SlineSlave=data.get("SlineSlave", []),
            attachNodePos_init=data.get("attachNodePos_init", []),
            attachNodePos=data.get("attachNodePos", data.get("attachNodePos_init", [])),
            attachNodeVec=data.get("attachNodeVec", []),
            attachNodeArea=data.get("attachNodeArea", []),
            attachNodeCd=data.get("attachNodeCd", []),
            ElType=data.get("ElType", []),
            ElIndex=data.get("ElIndex", []),
            type2node_L=data.get("type2node_L"),
            type2node_R=data.get("type2node_R"),
            waveTrans1st=data.get("waveTrans1st"),
            attachNodeSn=data.get("attachNodeSn"),
        )


@dataclass
class Vessel(FloatBody):
    pass


@dataclass
class OffSys:
    # Required fields (no defaults) - must come first
    nbod:             int
    nDoF:             int
    calDoF:           list[list[int]]
    
    bodPos_init:      list[list[float]]
    fairleadPos_init: list[list[float]]
    anchorPos_init:   list[list[float]]

    nAnchorLine:      int
    anchorLinePair:   list[list[int]]
    
    floatBodies:      list[FloatBody]

    # Optional fields (with defaults) - must come last
    nSharedLine:      Optional[int] = None
    sharedLinePair:   Optional[list[list[int]]] = None
    bodFairleadIdx:   Optional[list[list[int]]] = None
    bodAlineSlave:    Optional[list[list[int]]] = None
    bodSlineSlave:    Optional[list[list[list[int]]]] = None

    def __post_init__(self):
        # ndarray-ify lists
        self.bodPos_init = np.array(self.bodPos_init)
        self.anchorPos_init = np.array(self.anchorPos_init)
        self.fairleadPos_init = np.array(self.fairleadPos_init)
        
        if len(self.calDoF) != self.nbod:
            raise ValueError("calDoF must have the same length as nbod")
        
        if self.bodPos_init.shape != (3, self.nbod):
            raise ValueError("bodPos_init must be a 3xnbod matrix")
        if self.anchorPos_init.shape[0] != 3:
            raise ValueError("anchorPos_init must be a 3xNanchor matrix")
        if self.fairleadPos_init.shape[0] != 3:
            raise ValueError("fairleadPos_init must be a 3xNfairlead matrix")
        
        if self.nAnchorLine != len(self.anchorLinePair):
            raise ValueError("nAnchorLine must be equal to the length of anchorLinePair")
        
        # Validate optional shared line fields
        if self.nSharedLine is not None and self.sharedLinePair is not None:
            if self.nSharedLine != len(self.sharedLinePair):
                raise ValueError("nSharedLine must be equal to the length of sharedLinePair")
        
        print("Have instanced the json dict --> OffSys dataclass")
    
    @classmethod
    def from_dict(cls, data: dict) -> "OffSys":
        # Parse floatBody list into FloatBody objects using from_dict
        float_bodies_data = data.get("floatBody", [])
        float_bodies = [FloatBody.from_dict(fb) for fb in float_bodies_data]
        
        return cls(
            nbod=data.get("nbod", 0),
            nDoF=data.get("nDoF", 0),
            calDoF=data.get("calDoF", []),
            bodPos_init=data.get("bodPos_init", []),
            fairleadPos_init=data.get("fairleadPos_init", []),
            anchorPos_init=data.get("anchorPos_init", []),
            nAnchorLine=data.get("nAnchorLine", 0),
            anchorLinePair=data.get("anchorLinePair", []),
            floatBodies=float_bodies,
            # Optional fields
            nSharedLine=data.get("nSharedLine"),
            sharedLinePair=data.get("sharedLinePair"),
            bodFairleadIdx=data.get("bodFairleadIdx"),
            bodAlineSlave=data.get("bodAlineSlave"),
            bodSlineSlave=data.get("bodSlineSlave"),
        )


@dataclass
class ElSys:
    """
    Element System for Morison force calculations.
    
    Aggregates all slender elements (cylinders, nets) from multiple floating bodies
    for efficient viscous force calculations. Mirrors MATLAB structElsys function.
    """
    nbod:                     int                   # number of bodies
    nNodes4nbod:              int                   # total nodes across all bodies
    nNodesPerBod:             list[int]             # number of nodes per body

    DoF_Tran:                 list[int]             # translational DoF indices (flattened)
    DoF_Rot:                  list[int]             # rotational DoF indices (flattened)
    bod2DoF_Tran:             list[list[int]]       # ith body -> its 3 translational DoFs
    bod2DoF_Rot:              list[list[int]]       # ith body -> its 3 rotational DoFs
    
    Index_cylType:            list[int]             # global indices of cylinder elements
    Index_netType:            list[int]             # global indices of net elements

    Els2bod:                  list[list[int]]       # ith body -> [start_idx, end_idx] in node arrays
    bodPos_globInit:          list[list[float]]     # initial global position of each body (3 x nbod)

    attachNodePos_globInit:   list[list[float]]     # global position of nodes (3 x nNodes4nbod)
    attachNodePos_loc:        list[list[float]]     # local position of nodes (3 x nNodes4nbod)
    attachNodeCd:             list[float]           # drag coefficient per node
    attachNodeVec:            list[list[float]]     # unit vector per node (3 x nNodes4nbod)
    attachNodeArea:           list[float]           # projected area per node
    attachNodeSn:             Optional[list[float]] = None  # solidity ratio per node

    def __post_init__(self):
        """Convert lists to numpy arrays for numerical operations."""
        self.nNodesPerBod = np.array(self.nNodesPerBod)
        self.DoF_Tran = np.array(self.DoF_Tran)
        self.DoF_Rot = np.array(self.DoF_Rot)
        self.bod2DoF_Tran = np.array(self.bod2DoF_Tran)
        self.bod2DoF_Rot = np.array(self.bod2DoF_Rot)
        self.Index_cylType = np.array(self.Index_cylType)
        self.Index_netType = np.array(self.Index_netType)
        self.Els2bod = np.array(self.Els2bod)
        self.bodPos_globInit = np.array(self.bodPos_globInit)
        self.attachNodePos_globInit = np.array(self.attachNodePos_globInit)
        self.attachNodePos_loc = np.array(self.attachNodePos_loc)
        self.attachNodeCd = np.array(self.attachNodeCd)
        self.attachNodeVec = np.array(self.attachNodeVec)
        self.attachNodeArea = np.array(self.attachNodeArea)
        if self.attachNodeSn is not None:
            self.attachNodeSn = np.array(self.attachNodeSn)
        
        print(f"ElSys initialized: {self.nbod} bodies, {self.nNodes4nbod} total nodes")

    @classmethod
    def from_floatBodies(cls, floatBodies: list[FloatBody]) -> "ElSys":
        """
        Construct ElSys from a list of FloatBody objects.
        
        Mirrors the MATLAB structElsys function. Aggregates all attach nodes
        from multiple bodies into a single coordinate system for efficient
        force calculations.
        
        Args:
            floatBodies: List of FloatBody objects
            
        Returns:
            ElSys instance with all nodes aggregated
        """
        nbod = len(floatBodies)
        
        # Initialize arrays
        bodPos_globInit = np.zeros((3, nbod))
        nNodesPerBod = np.zeros(nbod, dtype=int)
        
        DoF_Tran = []
        DoF_Rot = []
        bod2DoF_Tran = np.zeros((nbod, 3), dtype=int)
        bod2DoF_Rot = np.zeros((nbod, 3), dtype=int)
        
        attachNodePos_globInit = []
        attachNodePos_loc = []
        attachNodeCd = []
        attachNodeVec = []
        attachNodeArea = []
        attachNodeSn = []
        
        # Check if any body has attachNodeSn
        has_sn = any(getattr(b, 'attachNodeSn', None) is not None for b in floatBodies)
        
        # First pass: collect all nodes and compute sizes
        nNodes4nbod = 0
        for ibod, body in enumerate(floatBodies):
            # Body global position
            bodPos_globInit[:, ibod] = np.array(body.posGlobal).flatten()[:3]
            
            # DoF indices (0-based, 6 DoF per body)
            dof_base = 6 * ibod
            DoF_Tran.extend([dof_base, dof_base + 1, dof_base + 2])
            DoF_Rot.extend([dof_base + 3, dof_base + 4, dof_base + 5])
            bod2DoF_Tran[ibod, :] = [dof_base, dof_base + 1, dof_base + 2]
            bod2DoF_Rot[ibod, :] = [dof_base + 3, dof_base + 4, dof_base + 5]
            
            # Node positions (local and global)
            local_pos = np.array(body.attachNodePos_init)[:3, :]
            global_pos = local_pos + bodPos_globInit[:, ibod:ibod+1]
            
            attachNodePos_loc.append(local_pos)
            attachNodePos_globInit.append(global_pos)
            
            # Node properties
            attachNodeVec.append(np.array(body.attachNodeVec)[:3, :])
            attachNodeCd.extend(np.array(body.attachNodeCd).flatten())
            attachNodeArea.extend(np.array(body.attachNodeArea).flatten())
            
            # Handle optional attachNodeSn
            if has_sn:
                if getattr(body, 'attachNodeSn', None) is not None:
                    attachNodeSn.extend(np.array(body.attachNodeSn).flatten())
                else:
                    # Pad with zeros if this body lacks Sn but others have it
                    n_current_nodes = local_pos.shape[1]
                    attachNodeSn.extend([0.0] * n_current_nodes)
            
            # Count nodes
            n_nodes = local_pos.shape[1]
            nNodesPerBod[ibod] = n_nodes
            nNodes4nbod += n_nodes
        
        # Concatenate node arrays
        attachNodePos_globInit = np.hstack(attachNodePos_globInit)
        attachNodePos_loc = np.hstack(attachNodePos_loc)
        attachNodeVec = np.hstack(attachNodeVec)
        attachNodeCd = np.array(attachNodeCd)
        attachNodeArea = np.array(attachNodeArea)
        
        attachNodeSn_final = None
        if has_sn:
            attachNodeSn_final = np.array(attachNodeSn)
        
        # Second pass: compute element indices
        Els2bod = np.zeros((nbod, 2), dtype=int)
        Index_cylType = []
        Index_netType = []
        
        for ibod, body in enumerate(floatBodies):
            # Element range for this body (0-based indices)
            start_idx = int(np.sum(nNodesPerBod[:ibod]))
            end_idx = start_idx + nNodesPerBod[ibod] - 1
            Els2bod[ibod, :] = [start_idx, end_idx]
            
            # Cylinder type indices
            if hasattr(body, 'type2node_L') and 'cylinder' in body.type2node_L:
                left = start_idx + body.type2node_L['cylinder']
                right = start_idx + body.type2node_R['cylinder']
                Index_cylType.extend(range(left, right + 1))
            
            # Net type indices
            if hasattr(body, 'type2node_L') and 'net' in body.type2node_L:
                left = start_idx + body.type2node_L['net']
                right = start_idx + body.type2node_R['net']
                Index_netType.extend(range(left, right + 1))
        
        return cls(
            nbod=nbod,
            nNodes4nbod=nNodes4nbod,
            nNodesPerBod=nNodesPerBod.tolist(),
            DoF_Tran=DoF_Tran,
            DoF_Rot=DoF_Rot,
            bod2DoF_Tran=bod2DoF_Tran.tolist(),
            bod2DoF_Rot=bod2DoF_Rot.tolist(),
            Index_cylType=Index_cylType,
            Index_netType=Index_netType,
            Els2bod=Els2bod.tolist(),
            bodPos_globInit=bodPos_globInit.tolist(),
            attachNodePos_globInit=attachNodePos_globInit.tolist(),
            attachNodePos_loc=attachNodePos_loc.tolist(),
            attachNodeCd=attachNodeCd.tolist(),
            attachNodeVec=attachNodeVec.tolist(),
            attachNodeArea=attachNodeArea.tolist(),
            attachNodeSn=attachNodeSn_final.tolist() if attachNodeSn_final is not None else None,
        )


#=============================================================================
#                      Line Types and Mooring System
#=============================================================================

@dataclass
class LineType:
    """Mooring line type properties from lineTypes JSON."""
    Nseg:           int                                      # number of segments
    NnodesPerSeg:   int   | list[int]                        # nodes per segment
    Nnodes:         int                                      # total nodes
    E:              float | list[float]                      # elastic modulus per segment [Pa]
    A:              float | list[float]                      # cross-sectional area per segment [m²]
    w:              float | list[float]                      # weight per length per segment [N/m]
    s:              float | list[float]                      # segment lengths [m]
    length:         float                                    # total line length [m]
    touchDownSeg:   int   | list[int]                        # touchdown segment index (1-based)
    X2F:            float                                    # horizontal distance to fairlead [m]
    Z2F:            float                                    # vertical distance to fairlead [m]
    HH:             list[float]  | list[list[float]]         # horizontal tension table [N]
    VV:             list[float]  | list[list[float]]         # vertical tension table [N]
    SS:             list[float]  | list[list[float]]         # arc length table [m]
    XX2anch:        list[float]                              # horizontal distance table [m]
    XXF2F:          Optional[list[float]] = None            # horizontal distance table [m]
    F2B:            Optional[list] = None                    # fairlead to body (optional)
    guessSol:       Optional[list] = None                    # guess solution (optional)

    def __post_init__(self):
        # ndarrayfy lists for numerical operations
        self.E = np.array(self.E)
        self.A = np.array(self.A)
        self.w = np.array(self.w)
        self.s = np.array(self.s)
        self.HH = np.array(self.HH)
        self.VV = np.array(self.VV)
        self.SS = np.array(self.SS)
        self.XX2anch = np.array(self.XX2anch)

    @classmethod
    def from_dict(cls, data: dict) -> "LineType":
        """Create LineType from dictionary."""
        return cls(
            Nseg=data.get("Nseg", 1),
            NnodesPerSeg=data.get("NnodesPerSeg", []),
            Nnodes=data.get("Nnodes", 0),
            E=data.get("E", []),
            A=data.get("A", []),
            w=data.get("w", []),
            s=data.get("s", []),
            length=data.get("length", 0.0),
            touchDownSeg=data.get("touchDownSeg", 1),
            X2F=data.get("X2F", 0.0),
            Z2F=data.get("Z2F", 0.0),
            HH=data.get("HH", []),
            VV=data.get("VV", []),
            SS=data.get("SS", []),
            XX2anch=data.get("XX2anch", []),
            XXF2F=data.get("XXF2F", []),
            F2B=data.get("F2B"),
            guessSol=data.get("guessSol"),
        )


@dataclass
class LineTypes:
    """Container for all line types from lineTypes JSON."""
    lineType: list[LineType]

    @classmethod
    def from_dict(cls, data: dict) -> "LineTypes":
        """Create LineTypes from dictionary."""
        line_types_data = data.get("lineType", [])
        if isinstance(line_types_data, dict):
            line_types_data = [line_types_data]
        line_types = [LineType.from_dict(lt) for lt in line_types_data]
        return cls(lineType=line_types)

    def __getitem__(self, idx: int) -> LineType:
        return self.lineType[idx]
    
    def __len__(self) -> int:
        return len(self.lineType)


#=============================================================================
#                      Simulation Configuration
#=============================================================================

@dataclass
class StaticSimuConfig:
    """Static simulation parameters."""
    enabled:    bool = False
    parameters: Optional[dict] = None

    @classmethod
    def from_dict(cls, data: dict) -> "StaticSimuConfig":
        return cls(
            enabled=data.get("enabled", False),
            parameters=data.get("parameters"),
        )


@dataclass
class DynamicSimuConfig:
    """Dynamic simulation parameters."""
    enabled:          bool = True
    numerical_method: Optional[dict] = None
    time_settings:    Optional[dict] = None

    @property
    def tStart(self) -> float:
        return self.time_settings.get("tStart", 0.0) if self.time_settings else 0.0
    
    @property
    def tEnd(self) -> float:
        return self.time_settings.get("tEnd", 100.0) if self.time_settings else 100.0
    
    @property
    def dt(self) -> float:
        return self.time_settings.get("dt", 0.1) if self.time_settings else 0.1

    @classmethod
    def from_dict(cls, data: dict) -> "DynamicSimuConfig":
        return cls(
            enabled=data.get("enabled", True),
            numerical_method=data.get("numerical_method"),
            time_settings=data.get("time_settings"),
        )


@dataclass
class SimuConfig:
    """Complete simulation configuration from simu_config.json."""
    simulator:    str = "offarmLab"
    static_simu:  Optional[StaticSimuConfig] = None
    dynamic_simu: Optional[DynamicSimuConfig] = None
    QSmoor_calc:  Optional[dict] = None

    @property
    def static_enabled(self) -> bool:
        return self.static_simu.enabled if self.static_simu else False

    @property
    def dynamic_enabled(self) -> bool:
        return self.dynamic_simu.enabled if self.dynamic_simu else False

    @property
    def qsmoor_enabled(self) -> bool:
        return self.QSmoor_calc.get("enabled", False) if self.QSmoor_calc else False

    @classmethod
    def from_dict(cls, data: dict) -> "SimuConfig":
        static_data = data.get("static_simu", {})
        dynamic_data = data.get("dynamic_simu", {})
        return cls(
            simulator=data.get("simulator", "offarmLab"),
            static_simu=StaticSimuConfig.from_dict(static_data) if static_data else None,
            dynamic_simu=DynamicSimuConfig.from_dict(dynamic_data) if dynamic_data else None,
            QSmoor_calc=data.get("QSmoor_calc"),
        )


#=============================================================================
#                      Variable System
#=============================================================================

@dataclass
class VarSys:
    """Variable system configuration from varSys.json."""
    vars:           list[str]              # variable names (var1, var2, ...)
    varNames:       list[str]              # human-readable names (Hs, Tp, ...)
    varUnits:       list[str]              # units (m, s, ...)
    varDescription: list[str]              # descriptions
    exprs:          list[str]              # expressions for derived variables
    bounds:         list[list[float]]      # [min, max] bounds for each variable
    method:         str = "random"         # sampling method

    @property
    def nVars(self) -> int:
        return len(self.vars)

    def __post_init__(self):
        self.bounds = np.array(self.bounds) if self.bounds else np.array([])

    @classmethod
    def from_dict(cls, data: dict) -> "VarSys":
        return cls(
            vars=data.get("vars", []),
            varNames=data.get("varNames", []),
            varUnits=data.get("varUnits", []),
            varDescription=data.get("varDescription", []),
            exprs=data.get("exprs", []),
            bounds=data.get("bounds", []),
            method=data.get("method", "random"),
        )


#=============================================================================
#                      Input Manifest
#=============================================================================

@dataclass
class InputManifest:
    """Input manifest from INPUT_manifest.json."""
    workspace_path: str
    files:          dict[str, str]

    @classmethod
    def from_dict(cls, data: dict) -> "InputManifest":
        return cls(
            workspace_path=data.get("workspace_path", "."),
            files=data.get("files", {}),
        )

    def get_file_path(self, key: str, base_path: Optional[Path] = None) -> Optional[Path]:
        """Get the full path for a file specified in the manifest."""
        if key not in self.files:
            return None
        rel_path = Path(self.workspace_path) / self.files[key]
        if base_path:
            return base_path / rel_path
        return rel_path


#=============================================================================
#                      Results / Cases
#=============================================================================

@dataclass
class CaseResult:
    """Case result from resultsPre JSON."""
    nCase:    int
    caseList: list[float]
    caseSeq:  int
    nVar:     int
    varSeq:   list[int]
    infoList: str
    log:      list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "CaseResult":
        return cls(
            nCase=data.get("nCase", 0),
            caseList=data.get("caseList", []),
            caseSeq=data.get("caseSeq", 0),
            nVar=data.get("nVar", 0),
            varSeq=data.get("varSeq", []),
            infoList=data.get("infoList", ""),
            log=data.get("log", []),
        )


#=============================================================================
#                      Helper Functions
#=============================================================================

def load_json(path: Path | str) -> dict:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env_config(path: Path | str) -> Env:
    """Load environment configuration from JSON file."""
    data = load_json(path)
    return Env.from_dict(data)


def load_line_types(path: Path | str) -> LineTypes:
    """Load line types from JSON file."""
    data = load_json(path)
    return LineTypes.from_dict(data)


def load_simu_config(path: Path | str) -> SimuConfig:
    """Load simulation configuration from JSON file."""
    data = load_json(path)
    return SimuConfig.from_dict(data)


def load_var_sys(path: Path | str) -> VarSys:
    """Load variable system from JSON file."""
    data = load_json(path)
    return VarSys.from_dict(data)


def load_offsys(path: Path | str) -> OffSys:
    """Load offshore system from JSON file."""
    data = load_json(path)
    return OffSys(**data)


def load_float_bodies(path: Path | str) -> list[FloatBody]:
    """Load floating bodies from offshore system JSON file."""
    data = load_json(path)
    float_bodies_data = data.get("floatBody", [])
    if isinstance(float_bodies_data, dict):
        float_bodies_data = [float_bodies_data]
    return [FloatBody.from_dict(fb) for fb in float_bodies_data]


def load_manifest(path: Path | str) -> InputManifest:
    """Load input manifest from JSON file."""
    data = load_json(path)
    return InputManifest.from_dict(data)
    




if __name__ == "__main__":
    # Get the usr0 directory path
    script_dir = Path(__file__).parent
    usr0_path = script_dir.parent.parent / "usr0"
    
    print("="*70)
    print(" "*15 + "TESTING PARAMS DATACLASSES WITH usr0 JSONs")
    print("="*70)
    print(f"usr0 path: {usr0_path}")
    print()

    # ----------------------------------------------------------------
    # Testing INPUT_manifest.json
    print("-"*70)
    print(" "*20 + "< testing InputManifest: >")
    manifest = load_manifest(usr0_path / "INPUT_manifest.json")
    print(f"  workspace_path: {manifest.workspace_path}")
    print(f"  files: {list(manifest.files.keys())}")

    # ----------------------------------------------------------------
    # Testing simu_config.json
    print("-"*70)
    print(" "*20 + "< testing SimuConfig: >")
    simu = load_simu_config(usr0_path / "simu_config.json")
    print(f"  simulator: {simu.simulator}")
    print(f"  static_enabled: {simu.static_enabled}")
    print(f"  dynamic_enabled: {simu.dynamic_enabled}")
    if simu.dynamic_simu:
        print(f"  tStart: {simu.dynamic_simu.tStart}, tEnd: {simu.dynamic_simu.tEnd}, dt: {simu.dynamic_simu.dt}")

    # ----------------------------------------------------------------
    # Testing varSys.json
    print("-"*70)
    print(" "*20 + "< testing VarSys: >")
    var_sys = load_var_sys(usr0_path / "varSys.json")
    print(f"  nVars: {var_sys.nVars}")
    print(f"  vars: {var_sys.vars}")
    print(f"  varNames: {var_sys.varNames}")
    print(f"  bounds shape: {var_sys.bounds.shape}")

   

    # ----------------------------------------------------------------
    # Testing env_withVar.json (environment with variables)
    print("-"*70)
    print(" "*20 + "< testing Env.wave config: >")
    env = load_env_config(usr0_path / "env_withoutVar.json")

    # test wave config
    print(f"  wave specType: {env.wave.specType}")
    print(f"  wave Hs: {env.wave.Hs}")  # May be a variable string
    print(f"  wave Tp: {env.wave.Tp}")  # May be a variable string

    # Testing current velocity
    print(" "*20 + "< testing Env.current: >")
    print(f"  current vel shape: {env.current.vel.shape}")
    print(f"  current zlevel shape: {env.current.zlevel.shape}")
    print(f"  current vel: {env.current.vel}")
    print(f"  current zlevel: {env.current.zlevel}")
    print(f"  current wakeRatio: {env.current.wakeRatio}")
    print(f"  current propDir: {env.current.propDir}")


    # ----------------------------------------------------------------
    # Testing lineTypes JSON
    print("-"*70)
    print(" "*20 + "< testing LineTypes: >")
    line_types = load_line_types(usr0_path / "lineTypes_23-Feb-2025_12-25-34.json")
    print(f"  Number of line types: {len(line_types)}")
    if len(line_types) > 0:
        lt = line_types[0]
        print(f"  First line type:")
        print(f"    Nseg: {lt.Nseg}")
        print(f"    length: {lt.length}")
        print(f"    touchDownSeg: {lt.touchDownSeg}")
        print(f"    HH shape: {lt.HH.shape}")


    # ----------------------------------------------------------------
    # Testing sys6cage_DgEls_wT1st.json (offshore system)
    print("-"*70)
    print(" "*20 + "< testing OffSys config: >")
    offsys_data = load_json(usr0_path / "sys6cage_DgEls_wT1st.json")
    # Extract just the OffSys fields (excluding floatBody which is nested)
    offsys_fields = {k: v for k, v in offsys_data.items() if k != "floatBody"}
    # Add required fields that may be missing
    offsys_fields.setdefault("bodFairleadIdx", [])
    offsys_fields.setdefault("bodAlineSlave", [])
    offsys_fields.setdefault("bodSlineSlave", [])
    offSys = OffSys(**offsys_fields)
    print(f"  nbod: {offSys.nbod}")
    print(f"  nDoF: {offSys.nDoF}")
    print(f"  nAnchorLine: {offSys.nAnchorLine}")
    print(f"  nSharedLine: {offSys.nSharedLine}")
    print(f"  bodPos_init shape: {offSys.bodPos_init.shape}")
    
    # ----------------------------------------------------------------
    # Testing floatBody from the offshore system
    print("-"*70)
    print(" "*20 + "< testing FloatBody configs: >")
    float_bodies_data = offsys_data.get("floatBody", [])
    if float_bodies_data:
        fb_data = float_bodies_data[0]
        floatBody = FloatBody(**fb_data)
        print(f"  name: {floatBody.name}")
        print(f"  MM shape: {floatBody.MM.shape}")
        print(f"  fairleadIndex: {floatBody.fairleadIndex}")
        print(f"  attachNodePos_init shape: {floatBody.attachNodePos_init.shape}")

    # ----------------------------------------------------------------
    # Testing resultsPre JSON
    print("-"*70)
    print(" "*20 + "< testing CaseResult: >")
    case_data = load_json(usr0_path / "resultsPre_24-Feb-2025_12-55-13.json")
    case_result = CaseResult.from_dict(case_data)
    print(f"  nCase: {case_result.nCase}")
    print(f"  caseList: {case_result.caseList}")
    print(f"  infoList: {case_result.infoList}")

    print("="*70)
    print(" "*20 + "ALL TESTS PASSED")
    print("="*70)


