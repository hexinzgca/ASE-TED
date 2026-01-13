import numpy as np

def dynamics_steer_potential(posI: np.ndarray, box: np.ndarray = None, 
    fposI_t: callable = None, k: float = 1.0e6,  t: float = 0.0):
    """
    Example: time-dependent steer potential for choose-I atoms
    Parameters:
        posI: coordinates of choose-I atoms, shape (IatomNum, 3)
        box:  simulation box lengths (optional), shape (3,)
        fposI_t: time-dependent steered position function for choose-I atoms, fposI_t(t)
        k:    force constant for steer potential (eV/Å^2)
        t:    current time
    Returns:
        V_energy: extra potential energy (eV)
        V_forces: extra forces on all atoms, shape (IatomNum+JatomNum, 3) in eV/Å
    """
    # Evaluate instantaneous charges
    posI_t = fposI_t(t) # steering function
    # Compute displacement vectors
    dr = posI - posI_t  # shape (IatomNum, 3)

    # Compute energy
    V_energy = 0.5 * k * np.sum(dr**2)
    # Compute force components
    V_forces = -k * dr  # shape (IatomNum, 3)
    
    return V_energy, V_forces


def dynamics_coulomb_potential(pos: np.ndarray, box: np.ndarray = None, 
    fq_t: callable = None,  t: float = 0.0):
    """
    Example: time-dependent Coulomb potential between choose-I atoms and choose-J atoms
    Parameters:
        pos: [posI, posJ]
            posI: coordinates of choose-I atoms, shape (IatomNum, 3)
            posJ: coordinates of choose-J atoms, shape (JatomNum, 3)
        box:  simulation box lengths (optional), shape (3,)

        fq_t: time-dependent charge function for choose-I atoms, fq_t(t)
        t:    current time
    Returns:
        V_energy: extra potential energy (eV)
        V_forces: extra forces on all atoms, shape (IatomNum+JatomNum, 3) in eV/Å
    """
    # Evaluate instantaneous charges
    posI, posJ = pos
    qI_t = fq_t(t, pos, 0)
    qJ_t = fq_t(t, pos, 1)

    # Initialize energy and force array
    V_energy = 0.0
    # Compute pairwise displacement vectors
    dr = posI[:, None, :] - posJ[None, :, :]  # shape (IatomNum, JatomNum, 3)

    # Apply minimum image convention if box is provided (no Ewald needed for virtual external potential)
    if box is not None:
        box = np.asarray(box)
        dr -= np.rint(dr / box) * box
        r = np.linalg.norm(dr, axis=-1)
    else:
        r = np.linalg.norm(dr, axis=-1)  # shape (IatomNum, JatomNum)

    # Avoid division by zero: replace r=0 with a small value
    r_safe = np.where(r == 0.0, 1e-6, r)

    # Compute pairwise Coulomb energy (k = 1/(4πε₀) ≈ 14.3996 eV·Å)
    k = 14.3996  # eV·Å
    pair_energy = k * qI_t * qJ_t / r_safe  # shape (IatomNum, JatomNum)
    V_energy = np.sum(pair_energy)

    # Compute forces: F = -∇V
    # Force on I atoms: F_I = Σ_J k * qI * qJ * dr / r³
    # Force on J atoms: F_J = -Σ_I k * qI * qJ * dr / r³
    prefactor = k * qI_t * qJ_t / (r_safe ** 3)  # shape (IatomNum, JatomNum)
    F_I = np.sum(prefactor[:, :, None] * dr, axis=1)   # shape (IatomNum, 3)
    F_J = -np.sum(prefactor[:, :, None] * dr, axis=0)  # shape (JatomNum, 3)

    # Store forces in output array
    return V_energy, [F_I, F_J]
