from itertools import combinations
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from IPython.display import display, Math
import time

# -------------------- Utility Functions --------------------
def indexing(state):
    """
    Convert a binary state representation into an integer index.

    Args:
        state (array-like): Binary state array.

    Returns:
        int: Integer index corresponding to the state.
    """
    n = len(state)
    powers_of_two = 2 ** np.arange(n)
    return int(np.dot(state, powers_of_two))

def inverse_indexing(index, L):
    """
    Convert an integer index back to a binary state.

    Args:
        index (int): Integer index.
        L (int): Number of lattice sites.

    Returns:
        np.ndarray: Binary state array.
    """
    state = np.zeros(2 * L, dtype=bool)
    for i in range(2 * L):
        state[i] = index & 1
        index >>= 1
    return state

def display_state(state):
    """
    Display a binary state as a LaTeX string of arrows.

    Args:
        state (array-like): Binary state array.
    """
    latexstring = r"$"
    for i, x in enumerate(state):
        if x == 1:
            latexstring += r"\uparrow" if i % 2 == 0 else r"\downarrow"
        else:
            latexstring += r"\phantom{\uparrow}" if i % 2 == 0 else r"\phantom{\downarrow}"
        if i % 2 == 1:
            latexstring += r"-"
    latexstring += r"$"
    display(Math(latexstring))

def reverse_dict(d):
    """
    Reverse the keys and values of a dictionary.

    Args:
        d (dict): Input dictionary.

    Returns:
        dict: Reversed dictionary.
    """
    return {v: k for k, v in d.items()}

# -------------------- Basis Generation --------------------
def FixedNStates(L, N):
    """
    Generate all states with a fixed number of electrons.

    Args:
        L (int): Number of lattice sites.
        N (int): Number of electrons.

    Returns:
        np.ndarray: Array of binary states.
    """
    if N > 2 * L:
        raise ValueError("Too many electrons for the number of sites!")

    indices_combinations = combinations(range(2 * L), N)
    binary_arrays = []

    for indices in indices_combinations:
        array = np.zeros(2 * L, dtype=bool)
        array[list(indices)] = 1
        binary_arrays.append(array)

    return np.array(binary_arrays)

def FixedNfixedZStates(L, N, Z):
    """
    Generate all states with a fixed number of electrons and fixed spin Z.

    Args:
        L (int): Number of lattice sites.
        N (int): Number of electrons.
        Z (int): Spin Z value.

    Returns:
        list: List of binary states.
    """
    if N > 2 * L:
        raise ValueError("Too many electrons for the number of sites!")

    indices_combinations = combinations(range(2 * L), N)
    binary_arrays = []

    for indices in indices_combinations:
        array = np.zeros(2 * L, dtype=bool)
        array[list(indices)] = 1
        if SpinZ(array) == Z:
            binary_arrays.append(array)

    return binary_arrays

def baseindices(basis):
    """
    Generate indices for a given basis.

    Args:
        basis (list): List of binary states.

    Returns:
        list: List of indices corresponding to the basis.
    """
    return [indexing(base) for base in basis]

# -------------------- Operators and Observables --------------------
def Repulsion(state, U):
    """
    Calculate the repulsion energy of a state.

    Args:
        state (array-like): Binary state array.
        U (float): Repulsion strength.

    Returns:
        float: Repulsion energy.
    """
    return U * np.dot(state[::2], state[1::2])

def SpinZ(state):
    """
    Compute the spin Z value of a state.

    Args:
        state (array-like): Binary state array.

    Returns:
        int: Spin Z value.
    """
    return int(state[::2].sum() - state[1::2].sum())

def hopping(state):
    """
    Generate all states reachable from a given state by hopping and the sign.

    Args:
        state (array-like): Binary state array.

    Returns:
        tuple: List of new states and their corresponding signs.
    """
    statelist = []
    signs = []
    n = len(state)

    for i in range(n):
        if state[i]:
            nxti = (i + 2) % n
            prvi = (i - 2) % n

            if not state[nxti]:
                tmp = state.copy()
                tmp[i], tmp[nxti] = tmp[nxti], tmp[i]
                statelist.append(tmp)
                signs.append(1 if nxti > i else -1)

            if not state[prvi]:
                tmp = state.copy()
                tmp[i], tmp[prvi] = tmp[prvi], tmp[i]
                statelist.append(tmp)
                signs.append(1 if prvi < i else -1)

    return statelist, signs

# -------------------- Lanczos Algorithm --------------------
def genPsi0(baseindices):
    """
    Generate a random normalized state.

    Args:
        baseindices (list): Indices representing the basis.

    Returns:
        np.ndarray: Random normalized state.
    """
    Nbasis = len(baseindices)
    re = np.random.randn(Nbasis)
    im = np.random.randn(Nbasis)
    Psi = re + 1j * im
    return Psi / np.linalg.norm(Psi)

def LanczosOpt(T, U, tol, basis, psi0, hopping_dict, baseindex_map, inversebaseindex_map, maxiter):
    """
    Perform the Lanczos algorithm to find the ground state energy.

    Args:
        T (float): Hopping term.
        U (float): Repulsion strength.
        tol (float): Convergence tolerance.
        basis (list): Basis states.
        psi0 (np.ndarray): Initial state.
        hopping_dict (dict): Hopping matrix in dictionary form.
        baseindex_map (dict): Map from basis indices to positions.
        inversebaseindex_map (dict): Map from positions to basis indices.
        maxiter (int): Maximum number of iterations.

    Returns:
        tuple: Alpha, beta, energy, and final state.
    """
    energy = [U]
    alpha = []
    beta = []

    psi1 = psi0.copy()
    Hpsi = HamiltonianOnPsiOptimized(psi1, T, U, basis, hopping_dict, baseindex_map, inversebaseindex_map)

    alpha.append(np.vdot(psi1, Hpsi))
    psi2 = Hpsi - alpha[-1] * psi1
    beta.append(np.linalg.norm(psi2))
    psi2 /= beta[-1]

    energy.append(tridiagenergy(alpha, beta, energy))

    for _ in range(maxiter):
        Hpsi = HamiltonianOnPsiOptimized(psi2, T, U, basis, hopping_dict, baseindex_map, inversebaseindex_map)
        alpha.append(np.vdot(psi2, Hpsi))
        psi3 = Hpsi - alpha[-1] * psi2 - beta[-1] * psi1
        beta.append(np.linalg.norm(psi3))

        if beta[-1] < tol:
            break

        psi1, psi2 = psi2, psi3 / beta[-1]
        energy.append(tridiagenergy(alpha, beta, energy))

    return alpha, beta, energy, psi2

# -------------------- Example Wrapper --------------------
def total_thing(params, tol=1e-5):
    """
    Example wrapper for running the Lanczos algorithm.

    Args:
        params (tuple): Tuple containing L, N, and Z.
        tol (float): Convergence tolerance.

    Returns:
        list: List of energy values.
    """
    L, N, Z = params
    T = 1.0
    U = 1000.0

    basis = FixedNfixedZStates(L, N, Z)
    baseindicesar = baseindices(basis)
    baseindex_map = {val: idx for idx, val in enumerate(baseindicesar)}
    inversebaseindex_map = reverse_dict(baseindex_map)

    psi1 = genPsi0(baseindices(basis))
    maxiter = len(basis)

    alpha, beta, energy, _ = LanczosOpt(T, U, tol, basis, psi1, {}, baseindex_map, inversebaseindex_map, maxiter)
    return energy

