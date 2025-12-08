# Import important libraries and set up the class for cable 
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import ode
from scipy import interpolate

# Optional numba import for JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


"""
============================================================================================================================
This code is a Python implementation for
    -a class named `Cable` to formulate the linear elastic
     catenary equation, which is a model for the shape of a hanging flexible cable or chain under its own weight.
     The class provides methods to calculate the catenary shape, tension, and other properties based on given parameters.
    - a class named `CableLM` unfinished

To Do List:
    - [] finish the CableLM class
    - [] test both classes with some examples
============================================================================================================================
"""


def array2dict(obj, attr_list):
    """
    Convert multiple array attributes of an object to a dictionary.
    
    Parameters:
    -----------
    obj : object
        The object containing the attributes to convert
    attr_list : list
        List of attribute names to convert
        
    Returns:
    --------
    dict
        Dictionary with attribute names as keys and converted values
    """
    result = {}
    for attr in attr_list:
        if hasattr(obj, attr) and getattr(obj, attr) is not None:
            value = getattr(obj, attr)
            result[attr] = value.tolist() if isinstance(value, np.ndarray) else value
    return result

# Help function for Catenary
def Catenary(H, V, w, s, E, A):
    """
    Calculate the catenary shape for a cable segment.

    Parameters:
    H (float): Horizontal force
    V (float): Vertical force
    w (float): Weight per unit length
    s (float): Segment length
    E (float): Young's modulus
    A (float): Cross-sectional area

    Returns:
    tuple: (l, h) where l is the horizontal span and h is the vertical drop
    """
    l = H * s / (E * A) + \
        H / w * (np.arcsinh(V / H) - np.arcsinh((V - w * s) / H))

    h = -(w * s / (E * A) * (V / w - s / 2) + \
         H / w * (np.sqrt(1 + (V / H)**2) - np.sqrt(1 + ((V - w * s) / H)**2)))

    return l, h

    # Commented out alternative calculation for h
    # h = -(w * s**2 / (E * A) * (V / (w * s) - 1/2) + \
    #      H / w * (np.sqrt(1 + (V / H)**2) - np.sqrt(1 + ((V - w * s) / H)**2)))

# Class for quasi-static cable analysis
class Cable:
    """
    ------------------------------------------------------------------------
    Cable class for modeling cable dynamics.

    Logs:
        2024-06-02 -> [V] node separation is done for joint between two segs
        2024-07-23 -> [V] Fix some warnings regarding typos     
        2024-07-24 -> [V] Add Tmaxcat, Tproof, Tbreak as properties
        2024-09-11 -> [V] convert from matlab to python, implementing
                        using scipy.optimize.fsolve() and
                        Lambda functions
        2024-09-12 -> [V] tested with examples of an anchored dual-segment 
                        cable, and a non-anchored  single-segment cable
    ------------------------------------------------------------------------
    """
    
    def __init__(self):
        print("#############################################################################")
        print("MAKING A CABLE LINE ... ")
        print()

        # Initialize properties
        self.name = None
        self.config = None # 1: anchored line, 2: non-anchored line
        self.Nseg = None
        self.NnodesPerSeg = None
        self.Nnodes = None
        self.E = None
        self.A = None
        self.w = None
        self.s = None
        self.length = None
        self.touchDownSeg = None
        self.X2F = None
        self.Z2F = None
        self.F2B = None
        self.guessSol = None
        self.HH = None
        self.VV = None
        self.SS = None
        self.XX2anch = None
        self.ZZ2anch = None
        self.XXF2F = None
        self.ZZF2B = None
        self.TmaxCat = None
        self.Tproof = None
        self.Tbreak = None
        self.shapeLoc = None
        self.shapeLoc_init = None
        self.info = None


    
    def __str__(self):
        return "\n".join(f"{attr}: {value}" for attr, value in vars(self).items())

    def help(self):
        print("!")

    def fromJson(self, jsonStruct:dict):
        """
        Convert a JSON structure to a Cable object.

        Args:
            jsonStruct (dict): A dictionary containing the cable properties.
        """
        # Must have fields
        # Convert single values to lists if needed
        for attr in ["E", "A", "w", "s", "NnodesPerSeg"]:
            value = jsonStruct[attr]
            if not isinstance(value, list):
                # If single value, convert to list with Nseg copies
                setattr(self, attr, [value])
            else:
                # If already a list, use as is
                setattr(self, attr, value)
        self.config = jsonStruct["config"]
        #self.config = 1
        self.Nseg = jsonStruct["Nseg"]
        self.Nnodes = jsonStruct["Nnodes"]
        self.touchDownSeg = jsonStruct["touchDownSeg"]
        self.length = jsonStruct["length"]
        self.X2F = jsonStruct["X2F"]
        self.Z2F = jsonStruct["Z2F"]

        # Nice to have fields
        self.guessSol = jsonStruct.get("guessSol", None)

        # check if the lookup table is provided
        if "HH" in jsonStruct:
            self.HH = np.array(jsonStruct["HH"])
        if "VV" in jsonStruct:
            self.VV = np.array(jsonStruct["VV"])
        if "SS" in jsonStruct:
            self.SS = np.array(jsonStruct["SS"])
        if "XX2anch" in jsonStruct:
            self.XX2anch = np.array(jsonStruct["XX2anch"])
        if "ZZ2anch" in jsonStruct:
            self.ZZ2anch = np.array(jsonStruct["ZZ2anch"])
        if "XXF2F" in jsonStruct:
            self.XXF2F = np.array(jsonStruct["XXF2F"])
        if "ZZF2B" in jsonStruct:
            self.ZZF2B = np.array(jsonStruct["ZZF2B"])


    def toDict(self):
        """
        Convert Cable object to a dictionary for JSON serialization.
        Handles numpy arrays by converting them to lists.
        """
        # Basic properties
        cable_dict = {
            "Nseg": self.Nseg,
            "NnodesPerSeg": self.NnodesPerSeg.tolist() if isinstance(self.NnodesPerSeg, np.ndarray) else self.NnodesPerSeg,
            "Nnodes": self.Nnodes,
            "A": self.A.tolist() if isinstance(self.A, np.ndarray) else self.A,
            "E": self.E.tolist() if isinstance(self.E, np.ndarray) else self.E,
            "w": self.w.tolist() if isinstance(self.w, np.ndarray) else self.w,
            "s": self.s.tolist() if isinstance(self.s, np.ndarray) else self.s,
            "length": self.length,
            "touchDownSeg": self.touchDownSeg,
            "X2F": self.X2F,
            "Z2F": self.Z2F
        }
        
        # Add calculated results using the helper function
        calculated_attrs = ["HH", "VV", "XX2anch", "shape_loc", "tension", "position"]
        cable_dict.update(array2dict(self, calculated_attrs))
        
        return cable_dict

    def loadDisCal(self, config=None, NXstep=30):
        """
        Calculate load-displacement relationship.

        Args:
        config (int): The style of the cable
        NXstep (int): The number of load-dis pairs needed (default 30)
        """
        if config is None:
            config = self.config

        if config == 1:  # with anchor
            # Implementation for config 1 goes here
            # =============================
            #        V                    %   
            #     H__|                    %   
            #         \                   %   
            #          \                  %   
            #           \                 %
            #             \               %
            #               \             %
            #                 \ _ _ _ __  %
            #                             %
            #   z                         %
            #   |__x                      %
            # =============================
            
            if self.touchDownSeg is None:
                print('Please tell the program the segment that starts to touch the bottom')
                return
            if self.X2F is None:
                print('Please give the distance from fairlead to anchor')
                return

            V = np.dot(self.w[:self.touchDownSeg], self.s[:self.touchDownSeg])

            def equations(H):
                xVec = []
                zVec = []
                for i in range(self.Nseg):
                    x, z = Catenary(H, V + self.w[i] * self.s[i] - np.dot(self.w[:i+1], self.s[:i+1]),
                                    self.w[i], self.s[i], self.E[i], self.A[i])
                    xVec.append(x)
                    zVec.append(z)
                eXF = -self.X2F + sum(xVec)
                eZF = -self.Z2F + sum(zVec)
                return eZF

            sol_H = fsolve(equations, 1e5)
            Hmax = abs(sol_H[0])
            print(f'Have found a maximum-tensioned catenary shape! Hmax = {Hmax}')

            self.HH = np.linspace(1e4, Hmax, NXstep)
            self.SS = np.zeros(len(self.HH))
            self.XX2anch = np.zeros(len(self.HH))
            self.ZZ2anch = np.zeros(len(self.HH))

            for iH, H in enumerate(self.HH):
                def equations(S):
                    V = self.w[self.touchDownSeg-1] * S + np.dot(self.w[:self.touchDownSeg-1], self.s[:self.touchDownSeg-1])
                    xVec = []
                    zVec = []
                    for i in range(self.Nseg):
                        if i == self.touchDownSeg - 1:
                            x, z = Catenary(H, V + self.w[i] * self.s[i] - np.dot(self.w[:i+1], self.s[:i+1]),
                                            self.w[i], S, self.E[i], self.A[i])
                        else:
                            x, z = Catenary(H, V + self.w[i] * self.s[i] - np.dot(self.w[:i+1], self.s[:i+1]),
                                            self.w[i], self.s[i], self.E[i], self.A[i])
                        xVec.append(x)
                        zVec.append(z)
                    eZF = -self.Z2F + sum(zVec)
                    return eZF

                sol_S = fsolve(equations, self.length / 2)
                self.SS[iH] = sol_S[0]
                
                V = self.w[self.touchDownSeg-1] * self.SS[iH] + np.dot(self.w[:self.touchDownSeg-1], self.s[:self.touchDownSeg-1])
                xVec = []
                for i in range(self.Nseg):
                    if i == self.touchDownSeg - 1:
                        x, _ = Catenary(H, V + self.w[i] * self.s[i] - np.dot(self.w[:i+1], self.s[:i+1]),
                                        self.w[i], self.SS[iH], self.E[i], self.A[i])
                    else:
                        x, _ = Catenary(H, V + self.w[i] * self.s[i] - np.dot(self.w[:i+1], self.s[:i+1]),
                                        self.w[i], self.s[i], self.E[i], self.A[i])
                    xVec.append(x)
                self.XX2anch[iH] = sum(xVec) + self.length - self.SS[iH] - sum(self.s[:self.touchDownSeg-1])
                self.ZZ2anch[iH] = 0  # By definition of the solution
            print("The maximum horizontal X-disance between fairlead and anchor is ", self.XX2anch[-1])
            self.HH[self.SS == 0] = 0
            if self.touchDownSeg == 1:
                self.VV = self.w[self.touchDownSeg-1] * self.SS
            else:
                self.VV = np.dot(self.w[:self.touchDownSeg-1], self.s[:self.touchDownSeg-1]) + self.w[self.touchDownSeg-1] * self.SS
            self.TmaxCat = np.sqrt(np.max(self.HH)**2 + np.max(self.VV)**2)

            self.s[self.touchDownSeg-1] = np.interp(self.X2F, self.XX2anch, self.SS)
            H0 = np.interp(self.X2F, self.XX2anch, self.HH)
            V0 = np.dot(self.w[:self.touchDownSeg], self.s[:self.touchDownSeg])

            print('------------------ Initial solution found -----------------------------')
            print(f'Horizontal distance fairlead to anchor == {self.X2F} [m]')
            print(f'Vertical distance fairlead to anchor  == {self.Z2F} [m]')
            print(f'Horizontal tension at fairlead    == {H0} [N]')
            print(f'Vertical tension at fairlead 1    == {V0} [N]')
            print(f'Total tension at fairlead 1    == {np.sqrt(H0**2 + V0**2)} [N]')
            print('-----------------------------------------------------------------------')

        elif config == 2:
            # Implementation for config 2 goes here
            # ===========================
            #    V                       %
            # H__|                       %
            #     \                      %
            #      \               /     % 
            #       \             /      %
            #         \         /        %
            #           \ _ _ /          %
            #                            %
            #   z                        %
            #   |__x                     %
            # ============================
            xVec = [lambda H, V: Catenary(H, 
                                          V + self.w[i]*self.s[i] - np.dot(self.w[:i+1], self.s[:i+1]),
                                          self.w[i], self.s[i], self.E[i], self.A[i])[0] 
                    for i in range(self.Nseg)]
            zVec = [lambda H, V: Catenary(H, 
                                          V + self.w[i]*self.s[i] - np.dot(self.w[:i+1], self.s[:i+1]),
                                          self.w[i], self.s[i], self.E[i], self.A[i])[1] 
                    for i in range(self.Nseg)]

            Gravity = np.dot(self.w, self.s)

            # Solve the range of cable tension vs. fairleads' relative positions
            self.XXF2F = np.linspace(0.6*self.length, 0.99*self.length, NXstep)
            self.HH = np.zeros((2, len(self.XXF2F)))
            self.VV = np.zeros((2, len(self.XXF2F)))
            self.SS = np.zeros((2, len(self.XXF2F)))
            self.ZZF2B = np.zeros((2, len(self.XXF2F)))

            # Iterate over the range of fairlead-to-anchor distances
            for iX, XXF2F in enumerate(self.XXF2F):
                def equations(vars):
                    H, V = vars
                    eXF = -XXF2F + sum(x(H, V) for x in xVec)
                    eZF = -self.Z2F + sum(z(H, V) for z in zVec)
                    return [eXF, eZF]

                try:
                    sols = fsolve(equations, self.guessSol)
                    self.HH[:, iX] = sols[0]
                    self.VV[0, iX] = sols[1]
                    self.VV[1, iX] = Gravity - self.VV[0, iX]
                except:
                    print('Solution is not found !')
                    print('Cannot determine an initial catenary shape!')
                    print('suggest changing guessed solution range!')

            self.TmaxCat = np.sqrt(np.max(self.HH, axis=1)**2 + np.max(self.VV, axis=1)**2)

            k = np.where(self.HH[0, :] != 0)[0]
            H0 = np.interp(self.X2F, self.XXF2F[k], self.HH[0, k])
            if self.Z2F == 0:
                V0 = 0.5 * np.dot(self.w, self.s)
            else:
                V0 = np.interp(self.X2F, self.XXF2F[k], self.VV[0, k])

            if H0 < 0 or V0 < 0:
                print('Negative solutions :: plz examine')
                print('    |_______ 1-Buoyancy and gravity setup may be not appropriate !')
                print('    |_______ 2-Initial positions of fairleads may be not appropriate !')
            else:
                print('------------------ Initial solution found -----------------------------')
                print(f'horizontal distance between fairleads == {self.X2F} [m]')
                print(f'Horizontal tension at fairlead 1  == {H0} [N]')
                print(f'Horizontal tension at fairlead 2  == {H0} [N]')
                print(f'Vertical tension at fairlead 1    == {V0} [N]')
                print(f'Vertical tension at fairlead 2    == {Gravity - V0} [N]')
                print(f'Total tension at fairlead 1    == {np.sqrt(H0**2 + V0**2)} [N]')
                print(f'Total tension at fairlead 2    == {np.sqrt(H0**2 + (Gravity - V0)**2)} [N]')
                print('-----------------------------------------------------------------------')

        else:
            print("Unsupported configuration")

    def shapeCal(self, config=None, X2F_t=None):
        if config is None:
            config = self.config

        if config == 1:  # Anchored line
            if X2F_t is None:
                self.s[self.touchDownSeg-1] = np.interp(self.X2F, np.unique(self.XX2anch), np.unique(self.SS))
                H0 = np.interp(self.X2F, np.unique(self.XX2anch), np.unique(self.HH))
                V0 = np.dot(self.w[:self.touchDownSeg], self.s[:self.touchDownSeg])
            else:
                self.s[self.touchDownSeg-1] = np.interp(X2F_t, np.unique(self.XX2anch), np.unique(self.SS))
                H0 = np.interp(X2F_t, np.unique(self.XX2anch), np.unique(self.HH))  
                V0 = np.dot(self.w[:self.touchDownSeg], self.s[:self.touchDownSeg])

            shape_loc = np.zeros((3, self.Nnodes))
            s = np.linspace(0, self.s[0], self.NnodesPerSeg[0])

            for iseg in range(self.Nseg):
                end_nodes = [sum(self.NnodesPerSeg[:iseg]),
                             sum(self.NnodesPerSeg[:iseg+1])]
                print(f"Segment {iseg}: Nodes {end_nodes[0]} to {end_nodes[1]}")
                
                if iseg > 0:
                    s_this_seg = np.linspace(sum(self.s[:iseg]), sum(self.s[:iseg+1]), self.NnodesPerSeg[iseg] + 1)
                    s = np.concatenate((s, s_this_seg[1:]))

                for is_ in range(end_nodes[0], end_nodes[1]):
                    x_out, z_out = Catenary(H0,
                        V0 + self.w[iseg] * self.s[iseg] - np.dot(self.w[:iseg+1], self.s[:iseg+1]),
                        self.w[iseg], s[is_] - s[end_nodes[0]], self.E[iseg], self.A[iseg])
                    
                    if iseg == 0:
                        shape_loc[0, is_] = x_out
                        shape_loc[2, is_] = z_out
                    else:
                        shape_loc[0, is_] = x_out + shape_loc[0, end_nodes[0]-1]
                        shape_loc[2, is_] = z_out + shape_loc[2, end_nodes[0]-1]

            if self.length > sum(self.s):
                self.shape_loc = np.column_stack((
                    shape_loc,
                    [self.length - sum(self.s) + shape_loc[0, end_nodes[1]-1], 0, -abs(self.Z2F)]
                ))
            else:
                self.shape_loc = shape_loc



        elif config == 2:  # Non-anchored line
            if X2F_t is None:
                k = np.where(self.HH[0, :] != 0)[0]
                H0 = np.interp(self.X2F, np.unique(self.XXF2F[k]), np.unique(self.HH[0, k]))
                if self.Z2F == 0:
                    V0 = 0.5 * np.dot(self.w, self.s)
                else:
                    V0 = np.interp(self.X2F, np.unique(self.XXF2F[k]), np.unique(self.VV[0, k]))
            else:
                k = np.where(self.HH[0, :] != 0)[0]
                H0 = np.interp(X2F_t, np.unique(self.XXF2F[k]), np.unique(self.HH[0, k]))
                if self.Z2F == 0:
                    V0 = 0.5 * np.dot(self.w, self.s)
                else:
                    V0 = np.interp(X2F_t, np.unique(self.XXF2F[k]), np.unique(self.VV[0, k]))

            shape_loc = np.zeros((3, self.Nnodes))
            s = np.linspace(0, self.s[0], self.NnodesPerSeg[0])

            for iseg in range(self.Nseg):
                end_nodes = [sum(self.NnodesPerSeg[:iseg]) - self.NnodesPerSeg[iseg],
                             sum(self.NnodesPerSeg[:iseg+1])]
                
                if iseg > 0:
                    s_this_seg = np.linspace(sum(self.s[:iseg]), sum(self.s[:iseg+1]), self.NnodesPerSeg[iseg] + 1)
                    s = np.concatenate((s, s_this_seg[1:]))

                for is_ in range(end_nodes[0], end_nodes[1]):
                    x_out, z_out = Catenary(H0,
                        V0 + self.w[iseg] * self.s[iseg] - np.dot(self.w[:iseg+1], self.s[:iseg+1]),
                        self.w[iseg], s[is_] - s[end_nodes[0]], self.E[iseg], self.A[iseg])
                    
                    if iseg == 0:
                        shape_loc[0, is_] = x_out
                        shape_loc[2, is_] = z_out
                    else:
                        shape_loc[0, is_] = x_out + shape_loc[0, end_nodes[0]-1]
                        shape_loc[2, is_] = z_out + shape_loc[2, end_nodes[0]-1]

            self.shape_loc = shape_loc
    
    def loadDisCal2D(self, config, Nstep = [10, 30]):
        if config == 1:
            # 1-anchored line
            ZZF2Anch = self.Z2F + np.linspace(-10, 10, Nstep[0])
            XXF2Anch = np.linspace(0.6*self.length, 0.99*self.length, Nstep[1])

            Vmesh = np.zeros((len(ZZF2Anch), len(XXF2Anch)))
            Hmesh = np.zeros((len(ZZF2Anch), len(XXF2Anch)))
            
            for iZ, ZF2Anch in enumerate(ZZF2Anch):
                self.Z2F = ZF2Anch
                self.loadDisCal(1)
                Vmesh[iZ, :] = self.VV[:]
                Hmesh[iZ, :] = self.HH[:]
            
            self.Vmesh = Vmesh
            self.Hmesh = Hmesh
            self.ZZF2Anch = ZZF2Anch
            self.XXF2Anch = XXF2Anch
            
        elif config == 2:
            # 2 fairlead line
            ZZF2F = self.Z2F + np.linspace(-5, 5, Nstep[0])
            XXF2F = np.linspace(0.6*self.length, 0.99*self.length, Nstep[1])

            Vmesh = np.zeros((len(ZZF2F), len(XXF2F)))
            Hmesh = np.zeros((len(ZZF2F), len(XXF2F)))

            for iZ, ZF2F in enumerate(ZZF2F):
                self.Z2F = ZF2F
                self.loadDisCal(2)
                Vmesh[iZ, :] = self.VV[0, :]
                Hmesh[iZ, :] = self.HH[0, :]
            
            self.Vmesh = Vmesh
            self.Hmesh = Hmesh  
            self.ZZF2F = ZZF2F
            self.XXF2F = XXF2F
        return Vmesh, Hmesh
            
    def getInfo(self):
        self.info = f'mass={self.w/9.81} [kg] | length={self.length} [m]'



"""
The following functions are used to calculate the tension in the cable

    - getTension:
        Calculate the tension at a given position for a single cable
    - getTension2ends:
        Calculate the tension at the two ends of a cable
    - fast_interp:
        A fast interpolation function
    - FQSmoor:
        The main function to calculate the force on the float body
"""

@njit
def getTension_numba(XX2anch, SS, HH, w, s, touchDownSeg, X2F_local):
    s[touchDownSeg-1] = np.interp(X2F_local, 
                                  np.unique(XX2anch), 
                                  np.unique(SS))
    
    H0 = np.interp(X2F_local, 
                   np.unique(XX2anch), 
                   np.unique(HH))
    
    V0 = np.sum(w[0:touchDownSeg] * s[0:touchDownSeg])

    return H0, V0

@njit
def getTension2ends_numba(XXF2F, w, s, HH, VV, X2F_local):
    kN0 = np.nonzero(HH[0, :])[0]
    H01 = np.interp(X2F_local, 
                    np.unique(XXF2F[kN0]), 
                    np.unique(HH[0, kN0]))
    
    if VV.shape[0] == 1:
        H02 = H01
        print(w*s)
        V01 = np.sum(w * s) / 2
        V02 = V01


    else:
        V01 = np.interp(X2F_local, XXF2F[kN0], VV[0, kN0])
        kN0 = np.nonzero(HH[1, :])[0]
        H02 = np.interp(X2F_local, XXF2F[kN0], HH[1, kN0])
        V02 = np.interp(X2F_local, XXF2F[kN0], VV[1, kN0])

    H0 = np.array([H01, H02])
    V0 = np.array([V01, V02])

    return H0, V0

def getTension(cable, X2F_local, use_numba=False):
    if use_numba:
        return getTension_numba(cable.XX2anch, cable.SS, cable.HH, 
                                np.array(cable.w), np.array(cable.s), 
                                cable.touchDownSeg, X2F_local)
    else:
        cable.s[cable.touchDownSeg-1] = np.interp(X2F_local, 
                                                np.unique(cable.XX2anch), 
                                                np.unique(cable.SS))
        
        # print(cable.s)  # For Debug
        H0 = np.interp(X2F_local, 
                    np.unique(cable.XX2anch), 
                    np.unique(cable.HH))
        
        # Convert cable.w and cable.s to NumPy arrays if they're not already
        w = np.array(cable.w)
        s = np.array(cable.s)
        
        V0 = np.sum(w[0:cable.touchDownSeg] * s[0:cable.touchDownSeg])

        return H0, V0
     
def getTension2ends(cable, X2F_local, use_numba=False):
    if use_numba:
        return getTension2ends_numba(cable.XXF2F, np.array(cable.w), np.array(cable.s), 
                                     cable.HH, cable.VV, X2F_local)
    else:
        kN0 = np.nonzero(cable.HH[0, :])[0]
        H01 = np.interp(X2F_local, 
                        np.unique(cable.XXF2F[kN0]), 
                        np.unique(cable.HH[0, kN0]))
        
        # Convert cable.w and cable.s to NumPy arrays if they're not already
        w = np.array(cable.w)
        s = np.array(cable.s)
        
        if cable.Z2F == 0:
            H02 = H01
            V01 = w * s.T / 2
            """
            print(H01.shape) # For Debug
            print(V01.shape) # For Debug
            print(V01)
            """
            if V01.shape[0] == 1:
                V01 = V01[0]
            V02 = V01
            
        else:
            V01 = np.interp(X2F_local, 
                            np.unique(cable.XXF2F[kN0]), 
                            np.unique(cable.VV[0, kN0]))
            kN0 = np.nonzero(cable.HH[1, :])[0]
            H02 = np.interp(X2F_local, 
                            np.unique(cable.XXF2F[kN0]), 
                            np.unique(cable.HH[1, kN0]))
            V02 = np.interp(X2F_local, 
                            np.unique(cable.XXF2F[kN0]), 
                            np.unique(cable.VV[1, kN0]))

        H0 = np.array([H01, H02])
        V0 = np.array([V01, V02])

    return H0, V0

def fast_interp(x, xp, fp):
    # Can use fast_interp instead of np.interp in the function
    indices = np.searchsorted(xp, x)
    indices = np.clip(indices, 1, len(xp) - 1)
    return fp[indices - 1] + (x - xp[indices - 1]) * (fp[indices] - fp[indices - 1]) / (xp[indices] - xp[indices - 1])

def FQSmoor(x, sys, lineType, iconfig):
    F = np.zeros((len(x) // 2, 1))

    # Update position of fairleads
    for ibod in range(sys.nbod):
        sys.fairleadPos[0:2, sys.bodFairleadIdx[ibod]] = (
            x[sys.calDoF(ibod, 1):sys.calDoF(ibod, 1) + 2] +
            sys.fairleadPos_init[0:2, sys.bodFairleadIdx[ibod]]
        )

    # Anchor line force calculation
    xF2Anch_Vec_temp = np.zeros((2, sys.anchorLinePair.shape[0]))
    sys.Aline_FH = np.zeros((1, sys.anchorLinePair.shape[0]))
    sys.Aline_FV = np.zeros((1, sys.anchorLinePair.shape[0]))
    sys.Aline_proj2xy = np.zeros((2, sys.anchorLinePair.shape[0]))

    for iline in range(sys.anchorLinePair.shape[0]):
        xF2Anch_Vec_temp[:, iline] = (
            sys.anchorPos[0:2, sys.anchorLinePair[iline, 0]] -
            sys.fairleadPos[0:2, sys.anchorLinePair[iline, 1]]
        )
        xF2Anch_VecNorm_temp = np.linalg.norm(xF2Anch_Vec_temp[:, iline])

        sys.Aline_FH[0, iline], sys.Aline_FV[0, iline] = getTension(lineType[sys.anchorLineType[iline]], xF2Anch_VecNorm_temp)
        sys.Aline_proj2xy[:, iline] = xF2Anch_Vec_temp[:, iline] / xF2Anch_VecNorm_temp

    # Sharedline force calculation
    if sys.sharedLinePair.shape[0] == 0:
        # Assemble projected force in global X and Y to F vector
        for ibod in range(sys.nbod):
            F[sys.calDoF(ibod, 1):sys.calDoF(ibod, 1) + 2, 0] += np.dot(
                sys.Aline_FH[0, sys.AlineSlave[ibod]],
                sys.Aline_proj2xy[:, sys.AlineSlave[ibod]].T
            )
    else:
        xF2F_Vec_temp = np.zeros((2, sys.sharedLinePair.shape[0]))
        sys.Sline_FH = np.zeros((2, sys.sharedLinePair.shape[0]))
        sys.Sline_FV = np.zeros((2, sys.sharedLinePair.shape[0]))
        sys.Sline_proj2xy = np.zeros((2, 2 * sys.sharedLinePair.shape[0]))

        for iline in range(sys.sharedLinePair.shape[0]):
            xF2F_Vec_temp[:, iline] = (
                sys.fairleadPos[0:2, sys.sharedLinePair[iline, 1]] -
                sys.fairleadPos[0:2, sys.sharedLinePair[iline, 0]]
            )
            xF2F_VecNorm_temp = np.linalg.norm(xF2F_Vec_temp[:, iline])

            sys.Sline_FH[:, iline], sys.Sline_FV[:, iline] = getTension2ends(lineType[sys.sharedLineType[iline]], xF2F_VecNorm_temp)
            sys.Sline_proj2xy[:, 2*iline] = xF2F_Vec_temp[:, iline] / xF2F_VecNorm_temp
            sys.Sline_proj2xy[:, 2*iline + 1] = -sys.Sline_proj2xy[:, 2*iline]

        # Assemble projected force in global X and Y to F vector
        for ibod in range(sys.nbod):
            F[sys.calDoF(ibod, 1):sys.calDoF(ibod, 1) + 2, 0] += (
                np.dot(
                    sys.Aline_FH[0, sys.AlineSlave[ibod]],
                    sys.Aline_proj2xy[:, sys.AlineSlave[ibod]].T
                ) +
                np.dot(
                    sys.Sline_FH[0, sys.SlineSlave[ibod][0, :]],
                    sys.Sline_proj2xy[:, 2*sys.SlineSlave[ibod][0, :] - 
                                      sys.SlineSlave[ibod][1, :]].T
                )
            )

    return F




    
##########################################################################################
#  1. test for anchored line
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rho = 1025
    g = 9.81
    lineType = Cable()
    lineType.Nseg = 2
    lineType.NnodesPerSeg = [30, 30]
    lineType.Nnodes = sum(lineType.NnodesPerSeg)
    lineType.A = [0.01, 0.006]
    lineType.E = [1e9/0.01, 7.64e8/0.006]
    lineType.w = [element * g for element in [3-np.pi/4*0.013**2*rho, 100]]
    lineType.s = [100, 1000]
    lineType.length = sum(lineType.s)
    lineType.touchDownSeg = 2
    lineType.X2F = 1030#979.4
    lineType.Z2F = -290
    lineType.Nnodes = sum(lineType.NnodesPerSeg)

    lineType.loadDisCal(1,50)
    lineType.shapeCal(1)
     
    """     
    print(lineType)
    plt.figure(figsize=(10, 6))
    plt.scatter(lineType.XX2anch, lineType.SS)
    plt.xlabel('Distance from anchor to fairlead [m]')
    plt.ylabel('Segment length [m]')
    plt.title('Segment length vs distance')
    plt.grid(True)
    plt.show() 
    """

    plt.figure(figsize=(10, 6))
    plt.plot(lineType.shape_loc[0,:], lineType.shape_loc[2,:], color='black')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Catenary Shape')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(lineType.XX2anch, lineType.HH)
    plt.xlabel('Distance from anchor to fairlead [m]')
    plt.ylabel('Horizontal tension [N]')
    plt.title('Horizontal tension vs distance')
    plt.grid(True)
    plt.show() 



##########################################################################################
#  2. Test for non-anchored line
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rho = 1025
    g = 9.81
    lineType = Cable()
    lineType.Nseg = 1
    lineType.NnodesPerSeg = [100]
    lineType.Nnodes = sum(lineType.NnodesPerSeg)
    lineType.A = [0.01]  # Make this a list to match Nseg
    lineType.E = [1e11/0.01]  # Make this a list to match Nseg
    lineType.w = [100*g]  # Make this a list to match Nseg
    lineType.s = [1050]  # Make this a list to match Nseg
    lineType.length = sum(lineType.s)
    lineType.touchDownSeg = 1
    lineType.X2F = 1000-55-55
    lineType.Z2F = 0
    lineType.F2B = [-320, -320]
    lineType.guessSol = [1e5, 1e5] 

    lineType.loadDisCal(2, 50)  # Assuming you want to use config 2

    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'Times New Roman'
    # Plot the load-displacement curve
    plt.figure(figsize=(10, 6))
    plt.scatter(lineType.XXF2F, lineType.HH[0,:])
    plt.xlabel('Distance between fairleads [m]')
    plt.ylabel('Horizontal tension [N]')
    plt.title('Horizontal tension vs distance')
    plt.grid(True)
    plt.show()
    
    lineType.shapeCal(2) 

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(lineType.shape_loc[0,:], lineType.shape_loc[2,:], color='black', linewidth=2)
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Catenary Shape')
    plt.grid(False)
    plt.show()





##########################################################################################
# 3. Test for anchored line from json
if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from Cable import Cable

    ltJson = {
        "config":1,
        "Nseg":2,
        "NnodesPerSeg":[50, 50],
        "Nnodes":100,
        "E":[70000000000, 50000000000],
        "A":[0.00013273228961416876, 0.004201784002972668],
        "w":[28.09534364485713, 1095.7100114041098],
        "s":[100, 402.2954463461476],
        "length":1100,
        "touchDownSeg":2,
        "X2F":990,
        "Z2F":-290,
        "F2B":[-320, -320],
        "guessSol":[1e5, 1e5],
        "HH":[],
        "VV":[],
        "SS":[],
        "XX2anch":[],
        "ZZ2anch":[],
        "XXF2F":[],
        "ZZF2B":[],
        "TmaxCat":2493833.1522086263,
        "Tproof":[],
        "Tbreak":[],
        "shapeLoc":[],
        "shapeLoc_init":[],
        "info":"mass=2.863949      111.6932 [kg] | length=1100 [m]"
    }

    lineType = Cable()
    lineType.fromJson(ltJson)
    lineType.loadDisCal(NXstep=50)

    plt.figure(figsize=(10, 6))
    plt.scatter(lineType.XX2anch, lineType.HH)
    plt.xlabel('Distance from anchor to fairlead [m]')
    plt.ylabel('Horizontal tension [N]')
    plt.title('Horizontal tension vs distance')
    plt.grid(True)
    plt.show()
    
    lineType.shapeCal(1, 950)

    plt.figure(figsize=(10, 6))
    plt.plot(lineType.shape_loc[0,:], lineType.shape_loc[2,:], color='black', linewidth=2)
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Catenary Shape')
    plt.grid(False)
    plt.show()


##########################################################################################
# 4. Test for 2D load-displacement curve ( anchored line case)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib

    rho = 1025
    g = 9.81
    lineType = Cable()
    lineType.Nseg = 2
    lineType.NnodesPerSeg = [30, 30]
    lineType.Nnodes = sum(lineType.NnodesPerSeg)
    lineType.A = [0.01, 0.006]
    lineType.E = [1e9/0.01, 7.64e8/0.006]
    lineType.w = [element * g for element in [3-np.pi/4*0.013**2*rho, 100]]
    lineType.s = [100, 1000]
    lineType.length = sum(lineType.s)
    lineType.touchDownSeg = 2
    lineType.X2F = 1030#979.4
    lineType.Z2F = -290
    lineType.Nnodes = sum(lineType.NnodesPerSeg)

    lineType.loadDisCal2D(1)
 
    fig = plt.figure(figsize=(10, 6))
    for iZ in range(np.size(lineType.ZZF2Anch)):
        plt.plot(lineType.XX2anch, lineType.Hmesh[iZ,:])
    plt.xlabel('Distance between fairleads [m]')
    plt.ylabel('Horizontal tension [N]')
    plt.title('Horizontal tension vs distance')
    plt.grid(True)
    plt.show()

    


# 5. Test for 2D load-displacement curve (shared line case)
#%matplotlib widget #if view in jupyter interactive mode
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    

    rho = 1025
    g = 9.81
    lineType = Cable()
    lineType.Nseg = 1
    lineType.NnodesPerSeg = [100]
    lineType.Nnodes = sum(lineType.NnodesPerSeg)
    lineType.A = [0.01]  # Make this a list to match Nseg
    lineType.E = [1e11/0.01]  # Make this a list to match Nseg
    lineType.w = [100*g]  # Make this a list to match Nseg
    lineType.s = [1050]  # Make this a list to match Nseg
    lineType.length = sum(lineType.s)
    lineType.touchDownSeg = 1
    lineType.X2F = 1000-55-55
    lineType.Z2F = 0
    lineType.F2B = [-320, -320]
    lineType.guessSol = [1e5, 1e5] 
     
    lineType.loadDisCal2D(2)

    # Use a single figure rather than creating two
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Create the mesh grid
    Xmesh, Zmesh = np.meshgrid(lineType.XXF2F, lineType.ZZF2F)

    # Create a more appealing surface plot with better coloring and lighting
    surf = ax.plot_surface(Xmesh, Zmesh, lineType.Hmesh, 
                          cmap='viridis',
                          linewidth=0,
                          antialiased=True,
                          alpha=0.8)

    # Improve labels with larger font
    ax.set_xlabel('Distance between fairleads [m]', fontsize=12)
    ax.set_ylabel('Z position of fairlead 1 [m]', fontsize=12)
    ax.set_zlabel('Vertical tension [N]', fontsize=12)

    # Add a title
    ax.set_title('Vertical Tension vs. Fairlead Position', fontsize=14)

    # Add a color bar to show the mapping of colors to values
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Vertical tension [N]')

    # Set an initial view angle for better perspective
    ax.view_init(30, 45)

    # Tighten layout to optimize spacing
    #plt.tight_layout()

    # Show the plot
    plt.show()


##########################################################################################