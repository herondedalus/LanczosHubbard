import matplotlib.pyplot as plt
import numpy as np

def plot_gap_reduction_quarter_filled(SpecterList, ilist):
    """
    Plot the gap reduction for quarter-filled systems.

    Args:
        SpecterList (dict): Dictionary of energy spectra.
        ilist (list): List of lattice sizes to analyze.
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(ilist)))
    Uvals = [1000, 345.510729, 119.377664, 41.2462638, 14.2510267, 4.92388263, 1.70125428, 0.587801607, 
             0.203091762, 0.0701703829, 0.0242446202, 0.0083767764, 0.00289426612, 0.001, 0.]

    plt.figure(figsize=(5, 5))
    for j, i in enumerate(ilist):
        L, N, Z = i, int(i / 2), int(i / 2) % 2
        string, stringZ = f"{L}{N}{Z}", f"{L}{N}{Z + 2}"

        min_energy = [energy[-1] for energy in SpecterList[string]]
        min_energyZ = [energy[-1] for energy in SpecterList[stringZ]]

        plt.plot(Uvals, np.array(min_energyZ) - np.array(min_energy), marker='o', label=f"L={i}", color=colors[j])

    plt.xlabel("U / T", fontsize=12)
    plt.ylabel(r"$\Delta E_{GS}$", fontsize=12)
    plt.title(f"N=L/2 L in {ilist}", fontsize=14)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_mu_reduction_quarter(SpecterList, ilist, offset=0):
    """
    Plot the chemical potential reduction for quarter-filled systems.

    Args:
        SpecterList (dict): Dictionary of energy spectra.
        ilist (list): List of lattice sizes to analyze.
        offset (int): Offset for U-values in the plot.
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(ilist)))
    Uvals = [1000, 345.510729, 119.377664, 41.2462638, 14.2510267, 4.92388263, 1.70125428, 0.587801607, 
             0.203091762, 0.0701703829, 0.0242446202, 0.0083767764, 0.00289426612, 0.001, 0.]

    plt.figure(figsize=(8, 5))
    for j, i in enumerate(ilist):
        L, N, Z = i, int(i / 2), int(i / 2) % 2
        string = f"{L}{N}{Z}"
        string_plus = f"{L}{N + 1}{(N + 1) % 2}"
        string_minus = f"{L}{N - 1}{(N - 1) % 2}"

        min_energy = [energy[-1] for energy in SpecterList[string]]
        min_energy_plus = [energy[-1] for energy in SpecterList[string_plus]]
        min_energy_minus = [energy[-1] for energy in SpecterList[string_minus]]

        chemical_potential = (np.array(min_energy_plus) + np.array(min_energy_minus) - 2 * np.array(min_energy))
        plt.plot(Uvals[offset:], chemical_potential[offset:], marker='o', label=f"L={i}", color=colors[j])

    plt.xlabel("U / T", fontsize=12)
    plt.ylabel(r"$\frac{\partial \mu}{\partial N}$", fontsize=12)
    plt.title(f"N=L/2 L in {ilist}", fontsize=14)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_gap_reduction(SpecterList, ilist):
    """
    Plot the gap reduction for half-filled systems.

    Args:
        SpecterList (dict): Dictionary of energy spectra.
        ilist (list): List of lattice sizes to analyze.
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(ilist)))
    Uvals = [1000, 345.510729, 119.377664, 41.2462638, 14.2510267, 4.92388263, 1.70125428, 0.587801607, 
             0.203091762, 0.0701703829, 0.0242446202, 0.0083767764, 0.00289426612, 0.001, 0.]

    plt.figure(figsize=(5, 5))
    for j, i in enumerate(ilist):
        L, N, Z = i, i, 0
        string, stringZ = f"{L}{N}{Z}", f"{L}{N}{Z + 2}"

        min_energy = [energy[-1] for energy in SpecterList[string]]
        min_energyZ = [energy[-1] for energy in SpecterList[stringZ]]

        plt.plot(Uvals, np.array(min_energyZ) - np.array(min_energy), marker='o', label=f"L={i}", color=colors[j])

    plt.xlabel("U / T", fontsize=12)
    plt.ylabel(r"$\Delta E_{GS}$", fontsize=12)
    plt.title(f"N=L L in {ilist}", fontsize=14)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_gs_u(SpecterList, Params):
    """
    Plot the ground state energy times U.

    Args:
        SpecterList (dict): Dictionary of energy spectra.
        Params (tuple): Parameters (L, N, Z).
    """
    L, N, Z = Params
    string = f"{L}{N}{Z}"

    min_energy = [energy[-1] for energy in SpecterList[string]]
    Uvals = [1000, 345.510729, 119.377664, 41.2462638, 14.2510267, 4.92388263, 1.70125428, 0.587801607, 
             0.203091762, 0.0701703829, 0.0242446202, 0.0083767764, 0.00289426612, 0.001, 0.]

    plt.figure(figsize=(4, 3))
    plt.plot(Uvals, np.array(Uvals) * np.array(min_energy), marker='o', color='b', label="Estimated GS Energy times U")
    plt.xlabel("U / T", fontsize=12)
    plt.ylabel(r"$E_{GS} \cdot U$", fontsize=12)
    plt.title(f"N={N}, L={L}, Sz={Z}", fontsize=14)
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_spin_gaps(SpecterList, Params):
    """
    Plot the spin gaps for a given parameter set.

    Args:
        SpecterList (dict): Dictionary of energy spectra.
        Params (tuple): Parameters (L, N, Z).
    """
    L, N, Z = Params
    string, stringZ = f"{L}{N}{Z}", f"{L}{N}{Z + 2}"

    min_energy = [energy[-1] for energy in SpecterList[string]]
    min_energyZ = [energy[-1] for energy in SpecterList[stringZ]]

    Uvals = [1000, 345.510729, 119.377664, 41.2462638, 14.2510267, 4.92388263, 1.70125428, 0.587801607, 
             0.203091762, 0.0701703829, 0.0242446202, 0.0083767764, 0.00289426612, 0.001, 0.]

    plt.figure(figsize=(8, 5))
    plt.plot(Uvals, min_energy, marker='o', label=f"Sz={Z}", color='b')
    plt.plot(Uvals, min_energyZ, marker='o', label=f"Sz={Z + 2}", color='r')
    plt.xlabel("U / T", fontsize=12)
    plt.ylabel(r"$\Delta E$", fontsize=12)
    plt.title(f"N={N}, L={L}, Sz={Z}, {Z + 1}", fontsize=14)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.plot(Uvals, np.array(min_energyZ) - np.array(min_energy), marker='o', label="Spin Gap", color='r')
    plt.xlabel("U / T", fontsize=12)
    plt.ylabel("GS Gap", fontsize=12)
    plt.title(f"Spin Gap for N={N}, L={L}", fontsize=14)
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_susceptibility(SpecterList, Params, offset=3):
    """
    Plot the susceptibility for a given parameter set.

    Args:
        SpecterList (dict): Dictionary of energy spectra.
        Params (tuple): Parameters (L, N, Z).
        offset (int): Offset for U-values in the plot.
    """
    L, N, Z = Params
    string = f"{L}{N}{Z}"
    string_plus = f"{L}{N + 1}{(N + 1) % 2}"
    string_minus = f"{L}{N - 1}{(N - 1) % 2}"

    min_energy = [energy[-1] for energy in SpecterList[string]]
    min_energy_plus = [energy[-1] for energy in SpecterList[string_plus]]
    min_energy_minus = [energy[-1] for energy in SpecterList[string_minus]]

    Uvals = [1000, 345.510729, 119.377664, 41.2462638, 14.2510267, 4.92388263, 1.70125428, 0.587801607, 
             0.203091762, 0.0701703829, 0.0242446202, 0.0083767764, 0.00289426612, 0.001, 0.]

    plt.figure(figsize=(8, 5))
    plt.plot(Uvals, min_energy, marker='o', label=f"N={N}", color='b')
    plt.plot(Uvals[offset:], min_energy_plus[offset:], marker='o', label=f"N={N + 1}", color='y')
    plt.plot(Uvals, min_energy_minus, marker='o', label=f"N={N - 1}", color='g')
    plt.xlabel("U / T", fontsize=12)
    plt.ylabel("GS Energy", fontsize=12)
    plt.title(f"Energy for N={N}, {N + 1}, {N - 1}, L={L}", fontsize=14)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(4, 4))
    chemical_potential = (np.array(min_energy_plus) + np.array(min_energy_minus) - 2 * np.array(min_energy))
    plt.plot(Uvals[offset:], chemical_potential[offset:], marker='o', label="Chemical Potential", color='r')
    plt.xlabel("U / T", fontsize=12)
    plt.ylabel(r"$\frac{\partial \mu}{\partial N}$", fontsize=12)
    plt.title(f"N={N}, L={L}", fontsize=14)
    plt.xscale('log')
    plt.grid(True)
    plt.show()

