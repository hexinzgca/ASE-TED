#!/usr/bin/env python3
import os, shutil
from typing import List
from copy import deepcopy
import numpy as np
# from ase.calculators.calculator import all_changes
from ase import units
from ase.io.trajectory import Trajectory
from ase.calculators.calculator import Calculator
from .decorator_utils import debug_helper, Timing
from .lammps_utils import load_lammps_data_0

def displacement(posI: np.ndarray, posJ: np.ndarray, box: np.ndarray = None):
    """
    Returns:
        dr: displacement vector between posI and posJ, shape (n, 3)
    """
    dr = posI - posJ
    if box is not None:
        dr -= np.rint(dr / box) * box
    return dr


def safe_norm(dr, axis: int = 1, eps: float = 1.0e-6):
    """
    Returns:
        r_safe: safe norm of dr, shape (n,)
    """
    r_safe = np.linalg.norm(dr, axis=axis)
    r_safe[r_safe == 0] = eps
    return r_safe


@debug_helper(enable=True, print_args=False, print_return=False)
def dynamics_restraint_potential(pos: np.ndarray, box: np.ndarray = None, t: float = 0.0, *args):
    """
    Time-dependent restraint potential between bonded atoms (I atoms and J atoms)
    Returns:
        V_energy: extra potential energy (eV)
        V_forces: extra forces on all atoms, shape (IatomNum+JatomNum, 3) in eV/Å
    """
    assert len(pos) == 2, 'pos should be a list of two arrays, each with shape (n, 3)'
    assert len(args) == 2, 'args should be a list of two arrays, each with shape (n,)'
    posI, posJ = pos
    kcoeff, r0 = args[0], args[1]

    dr = displacement(posI, posJ, box)
    r_safe = safe_norm(dr)
    f_safe = dr / r_safe[:, None]
    
    V_energy = 0.5 * np.sum(kcoeff * (r_safe-r0)**2)
    F_I = -kcoeff[:,None] * (r_safe-r0)[:,None] * f_safe
    F_J = -F_I

    return V_energy, [F_I, F_J] # the reduce of F_I and F_J should be careful that the index can be repeated!


@debug_helper(enable=True, print_args=False, print_return=False)
def dynamics_coulomb4b_potential(pos: np.ndarray, box: np.ndarray = None, t: float = 0.0, *args):
    """
    Time-dependent soft-Coulomb potential for 4 atoms: (A, B, C, D)-type atoms
    Parameters:
        pos: [posA, posB, posC, posD]
            posA: coordinates of A atoms, shape (IatomNum, 3)
            posB: coordinates of B atoms, shape (JatomNum, 3)
            posC: coordinates of C atoms, shape (KatomNum, 3)
            posD: coordinates of D atoms, shape (LatomNum, 3)
        box:  simulation box lengths (optional), shape (3,)
        t:    current time
        qref: scalar charge of the Coulomb 4-body potential
    Returns:
        V_energy: extra potential energy (eV)
        V_forces: extra forces on all atoms, shape (IatomNum+JatomNum, 3) in eV/Å
    """
    # Evaluate instantaneous charges
    # A, B, C, D; AC吸引，BD吸引；只有吸引到一定区域才会触发AB排斥 & CD排斥，不然AB和CD也是吸引的。
    assert len(pos) == 4, 'pos should be a list of four arrays, each with shape (n, 3)'
    assert len(args) == 1, 'args should be a list of one array, each with shape (1,)'
    posA, posB, posC, posD = pos
    qref = args[0] if args[0] >=0 else 0
    
    # Initialize energy and force array
    V_energy = 0.0
    # Compute pairwise displacement vectors
    drAC = displacement(posA[:,None,:], posC[None,:,:], box)
    drBD = displacement(posB[:,None,:], posD[None,:,:], box)
    drAB = displacement(posA[:,None,:], posB[None,:,:], box)
    drCD = displacement(posC[:,None,:], posD[None,:,:], box)
    rAC_safe = safe_norm(drAC, axis=2)
    rBD_safe = safe_norm(drBD, axis=2)
    rAB_safe = safe_norm(drAB, axis=2)
    rCD_safe = safe_norm(drCD, axis=2)

    qIqJ = np.abs(qref)**2
    k = 14.3996  # eV·Å
    bCO = 1.410; bOH = 0.950 # Å
        
    # AC, BD 吸引 (AC has a factor of 2!)
    pairAC_energy = - k * 2 * qIqJ/bCO * (1+np.log(rAC_safe/bCO))/ (rAC_safe/bCO)
    pairBD_energy = - k * qIqJ/bOH * (1+np.log(rBD_safe/bOH))/ (rBD_safe/bOH) # @dangerous
    pairAB_energy = + k * qIqJ/bCO * (1+np.log(rAB_safe/bCO))/ (rAB_safe/bCO)
    pairCD_energy = + k * qIqJ/bOH * (1+np.log(rCD_safe/bOH))/ (rCD_safe/bOH)

    prefactor_AC =  + k * 2 * qIqJ/bCO**2 * np.log(rAC_safe/bCO) / (rAC_safe / bCO)**2
    prefactor_BD =  + k * qIqJ/bOH**2 * np.log(rBD_safe/bOH) / (rBD_safe / bOH)**2
    prefactor_AB =  - k * qIqJ/bCO**2 * np.log(rAB_safe/bCO) / (rAB_safe / bCO)**2
    prefactor_CD =  - k * qIqJ/bOH**2 * np.log(rCD_safe/bOH) / (rCD_safe / bOH)**2

    # fast neglect interaction for A with free C & B with free D
    pairAC_energy[len(posA):] = 0
    pairBD_energy[len(posA):] = 0
    prefactor_AC[len(posA):] = 0
    prefactor_BD[len(posA):] = 0
    
    V_energy = np.sum(pairAC_energy) + np.sum(pairBD_energy) + np.sum(pairAB_energy) + np.sum(pairCD_energy)
    F_A = ( - np.sum(prefactor_AC[:, :, None] * drAC / rAC_safe[:, :, None], axis=1) 
            - np.sum(prefactor_AB[:, :, None] * drAB / rAB_safe[:, :, None], axis=1))
    F_B = ( - np.sum(prefactor_BD[:, :, None] * drBD / rBD_safe[:, :, None], axis=1) 
            + np.sum(prefactor_AB[:, :, None] * drAB / rAB_safe[:, :, None], axis=0))
    F_C = ( - np.sum(prefactor_CD[:, :, None] * drCD / rCD_safe[:, :, None], axis=1) 
            + np.sum(prefactor_AC[:, :, None] * drAC / rAC_safe[:, :, None], axis=0))
    F_D = ( + np.sum(prefactor_CD[:, :, None] * drCD / rCD_safe[:, :, None], axis=0) 
            + np.sum(prefactor_BD[:, :, None] * drBD / rBD_safe[:, :, None], axis=0))
    return V_energy, [F_A, F_B, F_C, F_D]


class NeFFCalculator(Calculator):
    """
    Calculator for NeFF potential.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, calc: Calculator, 
        neff_file: str,
        bond_topo_file: str,
        work_record_file: str, 
        bond_record_file: str):

        self._eqcalc = calc
        self._neff_file = neff_file
        self._bond_topo_file = bond_topo_file
        self._work_record_file = work_record_file
        self._bond_record_file = bond_record_file
        if os.path.exists(self._work_record_file): os.remove(self._work_record_file)
        if os.path.exists(self._bond_record_file): os.remove(self._bond_record_file)

        # read NeFF potential from neff file
        self._neff_data = self.PARSE_NEFF_FILE(neff_file)
        self._ne_potentials = deepcopy(self._neff_data['Coeffs'])

        self._exertList = []
        for _, t in enumerate(self._ne_potentials):
            neff_atom_list = []
            for atom_list in t['lists']:
                for index in atom_list:
                    if index not in neff_atom_list: neff_atom_list.append(index)
            neff_atom_list.sort()
            self._exertList.append(np.array(neff_atom_list))

        # read additional NeFF potential from bond topo file
        if os.path.exists(self._bond_topo_file):
            with open(self._bond_topo_file, 'r') as f:
                self._bond_data = load_lammps_data_0(f.read())
            self._bond_restraint = True
            self._bond_special_type = [9, 10]
            self._bond_type = []
            self._bond_Iinfo = [], [], [] # index, atype(atom type), sym
            self._bond_Jinfo = [], [], [] # index, atype(atom type), sym
            self._bond_k0, self._bond_r0 = [], []
            self._bond_special_indicator = [] # 0 or 1 to indicate if the bond
            self._bond_special_label = [] # label for reaction

            kcalpmol2eV = 0.0433641
            neff_atom_list = []
            _pairs = {}
            for i, t in enumerate(self._bond_data['Bonds']):
                btype = t['type']
                k, r = map(float, self._bond_data['Bond Coeffs'][btype-1]['value'].split())

                aindex = [a-1 for a in t['atom_index']]
                atype = [self._bond_data['Atoms'][ia]['type'] for ia in aindex]
                asym = [self._bond_data['sym'][ia] for ia in aindex]

                # make sure O-H & O-C bond: O is the first atom
                if btype in self._bond_special_type and asym[0] != 'O':
                    aindex[0], aindex[1] = aindex[1], aindex[0]                
                    atype[0], atype[1] = atype[1], atype[0]                    
                    asym[0], asym[1] = asym[1], asym[0]
                
                # for reduction 
                if (aindex[0], aindex[1]) in _pairs:
                    print(f'Warning: duplicated definition for bond!')
                    print(f'previous btype = {_pairs[(aindex[0], aindex[1])]}')
                    print(f'current new btype = {btype}')
                    continue

                for index in aindex:
                    if index not in neff_atom_list: neff_atom_list.append(index)

                self._bond_type.append(btype) # btype count from 1
                self._bond_k0.append(k)
                self._bond_r0.append(r)
                self._bond_Iinfo[0].append(aindex[0]), self._bond_Iinfo[1].append(atype[0]), self._bond_Iinfo[2].append(asym[0])
                self._bond_Jinfo[0].append(aindex[1]), self._bond_Jinfo[1].append(atype[1]), self._bond_Jinfo[2].append(asym[1])

                self._bond_special_indicator.append(1 if btype in self._bond_special_type else 0)

                _pairs[(aindex[0], aindex[1])] = btype

            self._bond_type = np.array(self._bond_type)
            self._bond_k0 = np.array(self._bond_k0) * kcalpmol2eV
            self._bond_r0 = np.array(self._bond_r0)
            self._bond_special_indicator = np.array(self._bond_special_indicator)
            self._bond_special_reacted = np.zeros_like(self._bond_special_indicator)
            self._bond_special_indice = np.where(self._bond_special_indicator == 1)[0]
            for i in self._bond_special_indice:
                print(f'{i}-th special bond, {self._bond_Iinfo[2][i]} - {self._bond_Jinfo[2][i]}')

            self._ne_potentials.append({
                'id': len(self._ne_potentials) + 1,
                'type': 'restraint',
                'func': dynamics_restraint_potential,
                'ncls': 2,
                'lists': [self._bond_Iinfo[0], self._bond_Jinfo[0]]
            })
            neff_atom_list.sort()
            self._exertList.append(np.array(neff_atom_list))
        else:
            self._bond_restraint = False

        # initial values
        self._prev_Coords = None
        self._dCoords = None
        self._prev_Forces = None
        self._prev_Energy_analysis = None
        self._prev_sum_Force_dot_Coords = None
        self._time = 0.0
        self._step = 0
        self._qt = 0.0
        self._bond_k = np.zeros_like(self._bond_k0)
        
        self._results = {
            'sumdE': None, 
            'sumdE_coul': None, 
            'sumdE_bond': None, 
            'sumdE_topo': None, 
            'work': None, 'work_step': None, 'neq_energy': None, 'neq_forces': None}

        # print(f"ExertList: len {len(self._exertList)}\n{self._exertList}")
        super().__init__()

    @classmethod
    def PARSE_NEFF_FILE(cls, filename: str):
        """
        Parse the NEFF block from the NEFF file.
        """
        data = {'Coeffs': [], 'Forces': []}
        with open(filename, 'r') as f: lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Coeffs'):
                i += 1
                while i < len(lines) and lines[i].strip().startswith('#'): i += 1
                while i < len(lines) and lines[i].strip() != '':
                    terms = lines[i].strip().split('#')[0].split()
                    id, type_str, ncls, params = int(terms[0]), terms[1], int(terms[2]), [float(v) for v in terms[3:]]
                    if type_str == 'coulomb4b':
                        func = dynamics_coulomb4b_potential
                    else:
                        raise ValueError(f"Unknown potential type: {type_str}")
                    data['Coeffs'].append({
                        'id': id,
                        'type': type_str, 'func': func,
                        'ncls': ncls, 'lists': [[] for _ in range(ncls)],
                        'params': params,
                    })
                    i += 1
            if lines[i].startswith('Forces'):
                i += 1
                while i < len(lines) and lines[i].strip().startswith('#'): i += 1
                while i < len(lines) and lines[i].strip() != '':
                    terms = lines[i].strip().split('#')[0].split()
                    force_id, neff_id, atom_idx, atom_icls = int(terms[0]), int(terms[1]), int(terms[2]), int(terms[3])
                    neff_ref = data['Coeffs'][neff_id - 1]
                    data['Forces'].append({
                        'id': force_id,
                        'neid': neff_id,
                        'aindex': atom_idx,
                        'acls': atom_icls,
                    })
                    neff_ref['lists'][atom_icls - 1].append(atom_idx - 1)
                    i += 1
        for i, t in enumerate(data['Coeffs']):
            for acls in t['lists']: acls.sort()
        return data

    @property
    def calc(self):
        return self._eqcalc

    @property
    def ne_potentials(self):
        return self._ne_potentials

    @debug_helper(enable=True, print_args=False, print_return=False)
    def update_restraint_topology(self, refered_temperature_K = 300, custom_loggor = None): 
        box = np.diag(self._eqcalc.atoms.get_cell())
        xyz = self._eqcalc.atoms.get_positions()

        posI_sp = xyz[self._bond_Iinfo[0]][self._bond_special_indice].copy()
        posJ_sp = xyz[self._bond_Jinfo[0]][self._bond_special_indice].copy()
    
        dr_sp = displacement(posI_sp, posJ_sp, box)
        r_safe_sp = safe_norm(dr_sp, axis=1)

        D_MAX = 9999
        rc_sp = 0.5 * (posI_sp + posJ_sp)
        dd_sp = displacement(rc_sp[:,None,:], rc_sp[None,:,:], box)

        dist_rc_sp = safe_norm(dd_sp, axis=2)
        dist_rc_sp_screen = np.ones_like(dist_rc_sp) * D_MAX
    
        bondtype_sp = self._bond_type[self._bond_special_indice]
        reacted_sp = self._bond_special_reacted[self._bond_special_indice]

        # move unreactive sites & reacted sites (only react onece!!!)
        special_index_i = np.where( (bondtype_sp == self._bond_special_type[0]) & (reacted_sp == 0))[0]
        special_index_j = np.where( (bondtype_sp == self._bond_special_type[1]) & (reacted_sp == 0))[0]
        dist_rc_sp_screen[np.ix_(special_index_i, special_index_j)] = dist_rc_sp[np.ix_(special_index_i, special_index_j)]
    
        spi, spj = np.unravel_index(np.argmin(dist_rc_sp_screen), dist_rc_sp_screen.shape)
        minrc = dist_rc_sp_screen[spi, spj]
        i = self._bond_special_indice[spi]
        j = self._bond_special_indice[spj]

        if custom_loggor is not None:
            custom_loggor.print(f'min rc bond length = {minrc}')
            custom_loggor.print(f'ith bond information: {i}, {self._bond_Iinfo[0][i]} - {self._bond_Jinfo[0][i]} {self._bond_Iinfo[2][i]}-{self._bond_Jinfo[2][i]}')
            custom_loggor.print(f'jth bond information: {j}, {self._bond_Iinfo[0][j]} - {self._bond_Jinfo[0][j]} {self._bond_Iinfo[2][j]}-{self._bond_Jinfo[2][j]}')
        else:
            print(f'min rc bond length = {minrc}')
            print(f'ith bond information: {i}, {self._bond_Iinfo[0][i]} - {self._bond_Jinfo[0][i]} {self._bond_Iinfo[2][i]}-{self._bond_Jinfo[2][i]}')
            print(f'jth bond information: {j}, {self._bond_Iinfo[0][j]} - {self._bond_Jinfo[0][j]} {self._bond_Iinfo[2][j]}-{self._bond_Jinfo[2][j]}')
        
        energy_old = (0.5 * self._bond_k[i] * (r_safe_sp[spi]-self._bond_r0[i])**2  # use _bond_k other than _bond_k0
                    + 0.5 * self._bond_k[j] * (r_safe_sp[spj]-self._bond_r0[j])**2)

        # after exchange energy Change I-Atom only (make sure all I atoms are O element)
        new_dri = displacement(xyz[self._bond_Iinfo[0][j]], xyz[self._bond_Jinfo[0][i]], box)
        new_drj = displacement(xyz[self._bond_Iinfo[0][i]], xyz[self._bond_Jinfo[0][j]], box)
        new_bi = np.linalg.norm(new_dri)
        new_bj = np.linalg.norm(new_drj)
        energy_new = (0.5 * self._bond_k[i] * (new_bi-self._bond_r0[i])**2  # use _bond_k other than _bond_k0
                    + 0.5 * self._bond_k[j] * (new_bj-self._bond_r0[j])**2)
        
        if custom_loggor is not None:
            custom_loggor.print(f'energy old = {energy_old}')
            custom_loggor.print(f'energy new = {energy_new}')
        else:
            print(f'energy old = {energy_old}')
            print(f'energy new = {energy_new}')
        
        refered_energy_ev = units.kB * refered_temperature_K
        beta_ev_inv = 1.0 / refered_energy_ev

        rand_num = np.random.rand()
        
        if custom_loggor is not None:
            custom_loggor.print(f'random number = {rand_num} vs exp(-deltaE * beta) = {np.exp(-(energy_new - energy_old) * beta_ev_inv)}')
        else:
            print(f'random number = {rand_num} vs exp(-deltaE * beta) = {np.exp(-(energy_new - energy_old) * beta_ev_inv)}')
        
        if minrc < 2.0 and (energy_new < energy_old or rand_num < np.exp(-(energy_new - energy_old) * beta_ev_inv)):
            self._bond_old = self._bond_Iinfo[0].copy()

            self._bond_Iinfo[0][i], self._bond_Iinfo[0][j] = self._bond_Iinfo[0][j], self._bond_Iinfo[0][i]
            self._bond_Iinfo[1][i], self._bond_Iinfo[1][j] = self._bond_Iinfo[1][j], self._bond_Iinfo[1][i]
            assert self._bond_Iinfo[2][i] == 'O' and self._bond_Iinfo[2][j] == 'O'
            self._bond_special_reacted[j] = 1 # label that bond's atom index has been exchanged and make it not back!

            self._bond_new = self._bond_Iinfo[0].copy()
            if np.all(self._bond_old[i] == self._bond_new[i]):
                print(f'fails to change')

            # update the lists of non-equilibrium potential (save topology)
            # print(self._ne_potentials[-1]['type'])
            self._ne_potentials[-1]['lists'] = [np.array(self._bond_Iinfo[0]), np.array(self._bond_Jinfo[0])]
            if custom_loggor is not None:
                custom_loggor.print('****** Reaction occur! bond i(Iatom) and bond j(Iatom) are exchanged ******')
                custom_loggor.print(f'energy old = {energy_old}')
                custom_loggor.print(f'energy new = {energy_new}')
                custom_loggor.print(f'delta energy = {energy_new - energy_old}')
            else:
                print('****** Reaction occur! bond i(Iatom) and bond j(Iatom) are exchanged ******')
                print(f'energy old = {energy_old}')
                print(f'energy new = {energy_new}')
                print(f'delta energy = {energy_new - energy_old}')
        else:
            if custom_loggor is not None:
                custom_loggor.print('Reject exchange by random')
            else:
                print('Reject exchange by random')
            energy_new = energy_old
            
        delta_energy = energy_new - energy_old
        return energy_new - energy_old

    @debug_helper(enable=True, print_args=False, print_return=False)
    def calculate(self, atoms, properties, system_changes):
        with Timing("neff calculate"):
            xyz = atoms.get_positions()
            box = np.diag(atoms.get_cell())

            # step 1: check the change of coordinates
            update_eq_calc = True
            if self._prev_Coords is not None:
                self._dCoord = displacement(atoms.get_positions(), self._prev_Coords, box)
                max_dCoord_norm = np.max(np.linalg.norm(self._dCoord, axis=1))
                if max_dCoord_norm < 1e-8: update_eq_calc = False
            else:
                self._prev_Coords = xyz.copy()
                self._prev_Forces = np.zeros_like(xyz)
                self._dCoord = np.zeros_like(xyz)
                self._prev_sum_Force_dot_Coords = 0.0
            
            # step 2: update eq calculator if needed
            if update_eq_calc:
                self._eqcalc.calculate(atoms=atoms, properties=['energy', 'forces'], system_changes=['positions'])
            else:
                _debug_print("*** No update for eq calculator ***")

            energy = self._eqcalc.results['energy']
            forces = self._eqcalc.results['forces']

            # step 3: calculate non-equilibrium part
            total_neq_energy = []; total_neq_forces = []        
            for _, t in enumerate(self._ne_potentials):
                with Timing(f"neff calculate potentials {t['type']}"):

                    neq_energy = 0.0; neq_forces = np.zeros_like(forces)
                    atoms_pos = [xyz[atom_list] for atom_list in t['lists']] # atom_list can repeat

                    # print(t['type'], len(t['lists']))
                    func = t['func']
                    if t['type'] == 'coulomb4b':
                        neq_energy, fcoll = func(atoms_pos, box, self._time, self._qt) # it's OK!
                    elif t['type'] == 'restraint':
                        neq_energy, fcoll = func(atoms_pos, box, self._time, self._bond_k, self._bond_r0)
                    else:
                        raise ValueError(f"unsupported potential type: {t['type']}")
                    
                    # reduction might be careful!
                    for i, atom_list in enumerate(t['lists']):
                        for j, atom_idx in enumerate(atom_list):
                            neq_forces[atom_idx, :] += fcoll[i][j]

                    total_neq_energy.append(np.sum(neq_energy))
                    total_neq_forces.append(neq_forces)

            # step 4: copy results to self.results (cannot be saved for analysis) & self._results (can saved for analysis)
            reduced_neq_energy = np.sum(total_neq_energy)
            reduced_neq_forces = np.sum(total_neq_forces, axis=0)
            total_energy = energy + reduced_neq_energy
            total_forces = forces + reduced_neq_forces

            self.results['energy'] = total_energy
            self.results['forces'] = total_forces
            self._results['energy'] = total_energy
            self._results['forces'] = total_forces
            self._results['reduced_neq_energy'] = reduced_neq_energy
            self._results['reduced_neq_forces'] = reduced_neq_forces
            self._results['total_neq_energy'] = total_neq_energy
            self._results['total_neq_forces'] = total_neq_forces

            # step 5: cumulate work (for each type external force on each atoms)
            work_step = []
            # only external neq_force (not total force!) on exertCoords contributes to work!!!
            for i in range(len(self._exertList)): # @TODO
                work_step.append(np.sum(self._dCoord[self._exertList[i]] * total_neq_forces[i][self._exertList[i]], axis=1))
            self._results['work_step'] = deepcopy(work_step)
            if self._results['work'] == [] or self._results['work'] is None: 
                self._results['work'] = deepcopy(work_step)
            for i, t in enumerate(work_step):
                self._results['work'][i] += t
            
            # step 5: update prev_coords, exertCoords, Forces & prev_sum_Force_dot_Coords
            self._prev_Coords = atoms.get_positions()
            self._prev_Forces = total_forces.copy()
            self._prev_sum_Force_dot_Coords += np.sum(total_forces * self._dCoord)


    @debug_helper(enable=True, print_args=False, print_return=False)
    def analysis(self, atoms, iterator, custom_loggor, noneq = False):
        custom_loggor.print(f'=== [Step: {iterator.nsteps:6d}] Neff Setups and Analysis ===')
        with Timing("setup neff charge interaction"):
            self._step = iterator.nsteps
            self._time = iterator.nsteps * iterator.dt

            t0, tend, texp, qmax, nhperiod = self._ne_potentials[0]['params']
            t0 = float(t0) * units.fs
            tend = float(tend) * units.fs
            texp = float(texp) * units.fs
            qmax = float(qmax)
            tT = (tend - t0) / int(nhperiod) # suggest nhperiod = 4

            # sumdE is cause by change of status parameters (like qt and bond_k)
            if self._results['sumdE'] is None: 
                self._results['sumdE'] = 0.0
                self._results['sumdE_coul'] = 0.0
                self._results['sumdE_bond'] = 0.0
                self._results['sumdE_topo'] = 0.0
            previous_neq_energy = self._results['reduced_neq_energy']
            previous_coul_energy = self._results['total_neq_energy'][0]
            previous_bond_energy = self._results['total_neq_energy'][1]
            update_neq_energy = previous_neq_energy
            update_coul_energy = previous_coul_energy
            update_bond_energy = previous_bond_energy

            # try to update the status parameters (like qt and bond_k and topology)
            ## 1. update qt for attraction and repulsion intection for reactive sites
            if noneq and self._time >= t0 and self._time <= tend:
                # self._qt = qmax * np.sin(np.pi*(self._time - t0)/tT) * np.exp(-(self._time-t0)/texp)
                xit = (self._time - t0) / tT - 0.5
                xit_int = np.floor(xit)
                self._qt = qmax * (-1)**xit_int * (2*xit_int+2 - 2 * (self._time - t0) / tT)
            else:
                self._qt = 0.0
                
            ## 2. update k for restraint potential for both unreactive & reactive sites
            k1 = (0.25 * np.abs(self._qt)) / (qmax+0.1) # for unreactive sites (strong interaction)
            if self._time >= t0 and self._time <= tend:
                k1 = np.maximum(k1, 0.01)
            k1max = (0.25 * qmax + 0.000) / (qmax+0.1)
            k1min = (0.25 * 0.00 + 0.000) / (qmax+0.1)

            k2 = 1.0 if self._time > tend else np.maximum(np.exp(-4*(tend - self._time)/(tend - t0)), 0.1)  # for reactive sites (weak interaction)

            # @TEMP
            k1 = 0.05
            k2 = 0.05

            self._bond_k = self._bond_k0 * k1
            special_index = np.where(self._bond_special_indicator==1)[0]
            self._bond_k[special_index] = self._bond_k0[special_index] * k2

            ## 3. update topology for current reactive & unreactive sites
            topo_energy = 0.0
            with Timing("neff update restraint topology"):
                topo_energy = self.update_restraint_topology(iterator.temperature_K, custom_loggor)

            neq_energy = 0.0
            with Timing("neff calculate potentials again"):
                xyz = self._eqcalc.atoms.get_positions()
                box = np.diag(self._eqcalc.atoms.get_cell())
                for _, t in enumerate(self._ne_potentials):
                    #print(t['type'], len(t['lists']))
                    atoms_pos = [xyz[atom_list] for atom_list in t['lists']] # list should be updated in update_restraint
                    func = t['func']
                    if t['type'] == 'coulomb4b':
                        #print('1', t['type'], len(t['lists']))
                        e1, _f1 = func(atoms_pos, box, self._time, self._qt)
                        update_coul_energy = e1
                    elif t['type'] == 'restraint':
                        #print('2', t['type'], len(t['lists']))
                        e1, _f1 = func(atoms_pos, box, self._time, self._bond_k, self._bond_r0)
                        update_bond_energy = e1
                    else:
                        raise ValueError(f"unsupported potential type: {t['type']}")
                    neq_energy += e1

            update_neq_energy = neq_energy

            self._results['sumdE'] += (update_neq_energy - previous_neq_energy)
            self._results['sumdE_coul'] += (update_coul_energy - previous_coul_energy)
            self._results['sumdE_bond'] += (update_bond_energy - previous_bond_energy)
            self._results['sumdE_topo'] += topo_energy

            custom_loggor.print(f"time: {self._time / units.fs:.6f} fs | qt = {self._qt:.6f} | k1 = {k1:.6f} | k2 = {k2:.6f}")

        with Timing("neff work analysis"):
            # check for energy
            if self._prev_Energy_analysis is None:
                self._prev_Energy_analysis = self._results['energy']
            total_energy = self._results['energy']
            Ekin = atoms.get_kinetic_energy()
            custom_loggor.print(f"Energys: Pot = {total_energy:.6f} eV | Kin = {Ekin:.6f} eV | Tot = {total_energy + Ekin:.6f} eV")
            
            # check for geometry and force
            dCoord = self._dCoord
            total_forces = np.zeros_like(self._dCoord) if 'forces' not in self._results else self._results['forces']
            max_coord_norm = np.max(np.linalg.norm(dCoord, axis=1))
            max_force_norm = np.max(np.linalg.norm(total_forces, axis=1))
            delta_E = total_energy - self._prev_Energy_analysis
            custom_loggor.print(f"> dCoord max: {max_coord_norm:10.6f} Å")
            custom_loggor.print(f"> Force  max: {max_force_norm:10.6f} eV/Å")
            custom_loggor.print(f"> Delta Epot: {delta_E:10.6f} eV")
            custom_loggor.print(f"> sum(F* dR): {self._prev_sum_Force_dot_Coords:10.6f} eV")
            
            stable_factor = max_force_norm * max_coord_norm
            if stable_factor > 1.0 and max_force_norm > 1e3 and max_coord_norm > 1.0:
                custom_loggor.print(f"* Warning: max force norm > 1e3 eV/Å. This may indicate a problem with the force field.")
                custom_loggor.print(f"* Save a snapshot of the system with max force norm > 1e3 eV/Å and then terminate the simulation.")

                if self._work_record_file.endswith('.work'):
                    self._restart_snapshot = self._work_record_file.replace('.work', '_restart.traj')
                else:
                    self._restart_snapshot = self._work_record_file + '_restart.traj'
                with Trajectory(self._restart_snapshot, 'w') as traj:
                    traj.write(atoms)
                exit(1)

            for i in range(len(self._results['work'])):
                custom_loggor.print(f"{i}-th work sum current: {np.sum(self._results['work_step'][i])}")
                custom_loggor.print(f"{i}-th work sum history: {np.sum(self._results['work'][i])}")
            with open(self._work_record_file, 'a') as f:
                f.write(f"{iterator.nsteps:6d} {self._results['energy']} ")
                f.write(" ".join(f"{ener:12.6e}" for ener in self._results['total_neq_energy']) + " ")
                f.write(f"{self._results['sumdE']:12.6e} ")
                f.write(f"{self._results['sumdE_coul']:12.6e} ")
                f.write(f"{self._results['sumdE_bond']:12.6e} ")
                f.write(f"{self._results['sumdE_topo']:12.6e} ")
                f.write(" ".join(f"{np.sum(ithwork):12.6e}" for ithwork in self._results['work']))
                f.write("\n")
                f.close()

            if 'energy' in self._results:
                self._prev_Energy_analysis = self._results['energy']
                self._prev_sum_Force_dot_Coords = 0.0 # reset to zero

        with Timing("neff bond statistics"):
            xyz = self._eqcalc.atoms.get_positions()
            box = np.diag(atoms.get_cell())
            cutoff = 2.5
            for i in range(1):
                t = self._ne_potentials[0]
                atoms_pos = [xyz[atom_list] for atom_list in t['lists']]
                # C      O      O      H
                siteA, siteB, siteC, siteD = atoms_pos[0], atoms_pos[1], atoms_pos[2], atoms_pos[3]
                custom_loggor.print(f'ABCD: {len(siteA)}, {len(siteB)}, {len(siteC)}, {len(siteD)}')
                lenA = len(siteA)
                custom_loggor.print('A index ' + ' '.join([str(k) for k in t['lists'][0]]))
                custom_loggor.print('B index ' + ' '.join([str(k) for k in t['lists'][1]]))
                custom_loggor.print('C index ' + ' '.join([str(k) for k in t['lists'][2]]))
                custom_loggor.print('D index ' + ' '.join([str(k) for k in t['lists'][3]]))
                custom_loggor.print('VMD index ' 
                    + ' '.join([str(k) for k in t['lists'][0]])
                    + ' ' + ' '.join([str(k) for k in t['lists'][1]])
                    + ' ' + ' '.join([str(k) for k in t['lists'][2][:lenA]])
                    + ' ' + ' '.join([str(k) for k in t['lists'][3][:lenA]])
                )
                
                self.count_free_0 =  len(siteC) // 2
                self.count_free_max = (len(siteB) + len(siteC)) // 2

                dispAB = siteA[:,None,:] - siteB[None,:,:]
                dispAC = siteA[:,None,:] - siteC[None,:,:]
                dispAB = dispAB - np.floor(dispAB / box[None,None,:] + 0.5) * box
                dispAC = dispAC - np.floor(dispAC / box[None,None,:] + 0.5) * box
                distAB = np.linalg.norm(dispAB, axis=2)
                distAC = np.linalg.norm(dispAC, axis=2)

                siteAB_minarg1 = np.argmin(distAB, axis=0) # find A-index
                siteAB_minval1 = np.min(distAB, axis=0)
                siteAC_minarg1 = np.argmin(distAC, axis=0) # find A-index
                siteAC_minval1 = np.min(distAC, axis=0)

                linkflagAB = siteAB_minarg1 // (len(siteA) // 4)
                linkflagAB[np.where(siteAB_minval1 > cutoff)] = -1
                
                linkflagAC = siteAC_minarg1 // (len(siteA) // 4)
                linkflagAC[np.where(siteAC_minval1 > cutoff)] = -1

                count_free1 = np.sum((linkflagAB[:lenA] == -1) & (linkflagAC[:lenA] == -1))
                count_free2 = np.sum((linkflagAC[lenA::2] == -1) & (linkflagAC[lenA+1::2] == -1))
                self.count_free = count_free1 + count_free2

                custom_loggor.print(f"free bond count: {count_free1} + {count_free2} = {self.count_free}")
                custom_loggor.print(f"linkflagAB: {linkflagAB}")
                custom_loggor.print(f"linkdistAB: {siteAB_minval1}")
                custom_loggor.print(f"linkflagAC: {linkflagAC}")
                custom_loggor.print(f"linkdistAC: {siteAC_minval1}")

                # write bond statistics report
                report = f'{iterator.nsteps} {self.count_free} '
                report += " ".join(f"{x}" for x in linkflagAB) + ' ' + ' '.join(f"{x}" for x in linkflagAC) 
                report += " " + " ".join(f"{x:.6f}" for x in siteAB_minval1) + ' ' + " ".join(f"{x:.6f}" for x in siteAC_minval1)
                with open(self._bond_record_file, 'a') as f:
                    f.write(f"{report}")
                    f.write("\n")
                    f.close()

        with Timing("clean temp dir"):
            try:
                if isinstance(self._eqcalc.tmp_dir, str):
                    if os.path.exists(self._eqcalc.tmp_dir):
                        shutil.rmtree(self._eqcalc.tmp_dir)
                        os.makedirs(self._eqcalc.tmp_dir)
                elif isinstance(self._eqcalc.tmp_dir, list):
                    for tmp_dir in self._eqcalc.tmp_dir:
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)
                            os.makedirs(tmp_dir)
            except Exception as e:
                custom_loggor.print(f"Clean Error: {e}")
                pass


if __name__ == '__main__':
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pprint import pprint
    from lammps_utils import parse_lammps_data_to_ase_atoms, load_lammps_data_0

    with open('oplsaa2.data', 'r') as f:
        data = load_lammps_data_0(f.read())
    atoms = parse_lammps_data_to_ase_atoms(data)
    # print(atoms)

    with open('system1.xyz', 'r') as f:
        xyz_block = f.read()
    mol = Chem.MolFromXYZBlock(xyz_block)
    Chem.SanitizeMol(mol)  # Ensure valence states are properly calculated
    mol = Chem.AddHs(mol)  # Add hydrogens explicitly
    # AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # uff_calc = AllChem.UFFGetMoleculeForceField(mol)

    #data = NeFFCalculator.PARSE_NEFF_FILE('system1.neff')
    # pprint(data)
    neff = NeFFCalculator(
        calc=Calculator(),
        neff_file='neff.neff',
        device='cpu'
    )

    def dr_langevin(natom, dt, gamma, temp, seed=42):
        np.random.seed(seed)
        return np.sqrt(2*gamma*temp*dt) * np.random.randn(natom, 3)

    atoms.calc = neff
    energy1 = atoms.get_potential_energy()
    force1 = atoms.get_forces()
    # pprint(atoms.calc.results)

    for _ in range(100):
        xyz = atoms.get_positions()
        dxyz = dr_langevin(len(xyz), 0.01, 1.0, 0.001)
        atoms.set_positions(xyz + dxyz)
        energy2 = atoms.get_potential_energy()
        force2 = atoms.get_forces()

        print('------')
        print(f"de: {np.sum(dxyz * 0.5* (force2 + force1))}")
        print(f"energy1: {energy1}")
        print(f"energy2: {energy2}")
        print(f"energy_diff: {energy2 - energy1}")

        energy1 = energy2
        force1 = force2

        print(f"work_sum: {np.sum(neff._results['work'])}")
