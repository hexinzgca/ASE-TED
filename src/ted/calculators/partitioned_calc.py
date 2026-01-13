#!/usr/bin/env python3

from tkinter import N
import ase
from ase.calculators.calculator import Calculator
from copy import deepcopy  # shallow and deep copy operations
import numpy as np
from .lammps_utils import load_lammps_data_0
from .decorator_utils import debug_helper, Timing
import os


class PartitionedCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    """
    Description
        Partitioned Calculators. Typically, for QM/MM calculation with link atoms.
    """

    LINK_ATOM_LABEL = 'H'
    BOND_PARAM = {
        "CH": 1.090, "OH": 0.950, "NH": 1.008,
        "CC": 1.520, "CO": 1.410, "CN": 1.475, # based on bond-eq oplsaa
    }

    def __init__(self, partCalcs, partFile, device=None):
        self._partCalcs = partCalcs # pass calculator objects into constructor
        self._partFile = partFile # partition file
        self._device = device

        self.tmp_dir = []
        for calcs in self._partCalcs:
            for calc in calcs:
                self.tmp_dir.append(calc.tmp_dir)

        self._atomLabels, self._partTypes, self._atomLinks = self.PARSE_PART_FILE(self._partFile)
        self._atomLabels = np.array(self._atomLabels)
        self._atomCoords = np.zeros((len(self._atomLabels), 3)) # initialize atom coordinates to zeros
        
        self._partTypeUniq = sorted(list(set(self._partTypes)))
        self._partTypeNum = len(self._partTypeUniq)

        assert len(self._partTypeUniq) == len(self._partCalcs), "partition number must match calculator number"
        print(f'detected part types: {self._partTypeUniq} with total number: {self._partTypeNum}')

        self._partInfo = {}
        for parttype in self._partTypeUniq: 
            self._partInfo[parttype] = {'atomList': []}
        for i, parttype in enumerate(self._partTypes): # @note atom index starts from 0 
            self._partInfo[parttype]['atomList'].append(i)

        for parttype in self._partTypeUniq:
            print(f'[+] building partition <{parttype}> for calculator {self._partCalcs[parttype][0]}...')
            if len(self._partCalcs[parttype]) > 1:
                print(f'[-] building partition <{parttype}> for calculator {self._partCalcs[parttype][1]}...')
            
            atomList_part = self._partInfo[parttype]['atomList']
            linkList1_part = []
            linkList2_part = []
            linkRatio = []
            for idx in atomList_part:
                for inbr in self._atomLinks[idx]: # @note inbr from 0 to count
                    if self._partTypes[inbr] == parttype + 1:
                        linkList1_part.append(idx) # inner link
                        linkList2_part.append(inbr) # outer link
                        label = self._atomLabels[idx]
                        link_ratio = self.BOND_PARAM[f'{label}H'] / self.BOND_PARAM[f'C{label}']
                        linkRatio.append(link_ratio)
                    #     # linkList_part.append([idx, inbr])  # inner part --> outer part
                    # elif self._partTypes[inbr] == parttype or self._partTypes[inbr] == parttype - 1:
                    #     pass
                    # else:
                    #     raise ValueError(f'part {parttype} has neighbor atom {inbr} with part type {self._partTypes[inbr]}!')
            
            fullList_part = atomList_part + linkList2_part
            calcList_part = deepcopy(fullList_part)
            if parttype > 0: calcList_part = self._partInfo[parttype-1]['atomList'] + calcList_part
            fullList_part.sort() # sort the list of atoms with link atoms
            calcList_part.sort() # sort the list of atoms with link atoms
            
            self._partInfo[parttype]['linkList1'] = linkList1_part
            self._partInfo[parttype]['linkList2'] = linkList2_part
            self._partInfo[parttype]['linkRatio'] = np.array(linkRatio)
            self._partInfo[parttype]['fullList'] = fullList_part
            self._partInfo[parttype]['calcList'] = calcList_part

            self._partInfo[parttype]['atomLabel'] = self._atomLabels[atomList_part]
            atomLabels_copy = deepcopy(self._atomLabels);
            atomLabels_copy[linkList2_part] = self.LINK_ATOM_LABEL
            self._partInfo[parttype]['fullLabel'] = atomLabels_copy[fullList_part]
            self._partInfo[parttype]['calcLabel'] = atomLabels_copy[calcList_part]

            self._partInfo[parttype]['atomNum'] = len(atomList_part)
            self._partInfo[parttype]['linkNum'] = len(linkList1_part)
            self._partInfo[parttype]['fullNum'] = len(fullList_part)
            self._partInfo[parttype]['calcNum'] = len(calcList_part)

            for tag in ['atom', 'link', 'full', 'calc']:
                print(f'partition [{parttype}] has {self._partInfo[parttype][f"{tag}Num"]} {tag} atoms for calculator {self._partCalcs[parttype][0]}')
            
            print(f'calculator {self._partCalcs[parttype][0].n_atoms} is compilable with {self._partInfo[parttype]["calcNum"]} atoms [-1 means any]')
            if len(self._partCalcs[parttype]) > 1:
                print(f'calculator {self._partCalcs[parttype][1].n_atoms} is compilable with {self._partInfo[parttype]["calcNum"]} atoms [-1 means any]')

        Calculator.__init__(self)

    @debug_helper(enable=True, print_args=False, print_return=False)
    def update_calcAtoms(self, parttype: int, atoms: ase.Atoms):
        """
        Update the atoms in the part with parttype.
        """
        box = np.diag(atoms.get_cell())
        atomCoords = atoms.get_positions().copy()

        # move the atoms to the center of the box with PBC ? not necessary!
        # atomCoords = atomCoords - np.floor(atomCoords / box[None,:]) * box

        assert atomCoords.shape == self._atomCoords.shape, "passaged atoms shape must match full calculator atomCoords shape"
        
        _coord_list = self._partInfo[parttype]['linkList1']
        _coord = atomCoords[_coord_list]
        _neigh_list = self._partInfo[parttype]['linkList2']
        _neigh = atomCoords[_neigh_list]
        _linkRatio = self._partInfo[parttype]['linkRatio']

        # make sure the link atoms are in the same box with PBC symmetry
        _neigh = _neigh - np.floor((_neigh - _coord) / box[None,:] + 0.5) * box # 0.5 is for rounding
        # _neigh = _neigh - np.rint((_neigh - _coord) / box[None,:]) * box # slower speed

        dist = np.linalg.norm(_neigh - _coord, axis=1)
        if not np.all(dist < 2.0):
            _debug_print(f"link atom distance is large {dist} where the box is {box}")
        _h_coord = _coord + (_neigh - _coord) * _linkRatio[:, None]
        atomCoords[_neigh_list] = _h_coord # update with link atom coordinates 

        _calc_list = self._partInfo[parttype]['calcList']
        calcCoords = atomCoords[_calc_list]

        # define atoms for sub calculators        
        if 'calcAtoms' not in self._partInfo[parttype] or self._partInfo[parttype]['calcAtoms'] is None:
            symbols = np.array(atoms.get_chemical_symbols())
            self._partInfo[parttype]['calcAtoms'] = ase.Atoms(
                symbols=symbols[_calc_list],
                positions=calcCoords,
                cell=atoms.get_cell(),
                pbc=atoms.get_pbc(),
            )
        else:
            self._partInfo[parttype]['calcAtoms'].set_positions(calcCoords)
        return


    @debug_helper(enable=True, print_args=False, print_return=False)
    def reduce_calcForces(self, parttype: int, forces: np.ndarray):
        """
        Reduce the forces in the part with parttype.
        """
        #print(f'forces shape: {forces.shape}')
        #print(f'fullNum: {self._partInfo[parttype]["fullNum"]}')
        #print(f'calcNum: {self._partInfo[parttype]["calcNum"]}')
        assert forces.shape[0] == self._partInfo[parttype]['calcNum'], "forces shape must match full calculator forces shape"
        
        _coord_list = self._partInfo[parttype]['linkList1']
        _coord = self._atomCoords[_coord_list]
        _neigh_list = self._partInfo[parttype]['linkList2']
        _neigh = self._atomCoords[_neigh_list]
        _linkRatio = self._partInfo[parttype]['linkRatio']

        fullForces = np.zeros_like(self._atomCoords)
        _calc_list = self._partInfo[parttype]['calcList']
        fullForces[_calc_list] = forces
        linkForces = fullForces[_neigh_list]
        fullForces[_neigh_list] = linkForces * _linkRatio[:, None]
        fullForces[_coord_list] = linkForces  * (1-_linkRatio[:, None])
        
        return fullForces


    @property
    def get_calcAtoms(self, parttype: int):
        """
        Get the atoms in the part with parttype.
        """
        return self._partInfo[parttype]['calcAtoms']

    @property
    def get_calcCoords(self, parttype: int):
        """
        Get the atom coordinates in the part with parttype.
        """
        return self._partInfo[parttype]['calcAtoms'].get_positions()

    @debug_helper(enable=True, print_args=False, print_return=False)
    def calculate(self, atoms=None, properties=None, system_changes=None, force_consistent=False):    
        """
        统一计算入口：一次性调用子计算器获取能量和力，避免重复调用，提升效率
        :param atoms: ASE Atoms 对象
        :param force_consistent: 是否使用力一致能量（对应 ASE 规范）
        """
        # 初始化总能量和总力
        self._result = {}
        # ===================== 核心：单循环合并能量和力的计算，一次性调用子计算器 =====================
        for parttype in self._partTypeUniq:
            self._result[parttype] = {}
            
            # 1. 更新当前 parttype 的计算原子对象
            self.update_calcAtoms(parttype, atoms)
            calc_atoms = self._partInfo[parttype]['calcAtoms']
            
            # 2. 处理第一个子计算器（索引0）：一次性获取 energy1 和 force1
            with Timing(f'{parttype} calculate time for 0'):                
                sub_calc_0 = self._partCalcs[parttype][0]
                calc_atoms.calc = sub_calc_0
                sub_calc_0.calculate(atoms=calc_atoms, properties=['energy', 'forces'], system_changes=system_changes)
                energy1 = sub_calc_0.results['energy']  # 等价于 get_potential_energy，直接读缓存
                force1 = sub_calc_0.results['forces']    # 等价于 get_forces，直接读缓存
                fullforce1 = self.reduce_calcForces(parttype, force1)
            _debug_print(f'{parttype} 0 max force norm: {np.max(np.linalg.norm(force1, axis=1))}')
            
            # 3. 处理第二个子计算器（索引1，若存在）：一次性获取 energy2 和 force2
            with Timing(f'{parttype} calculate time for 1'):
                energy2 = 0.0
                fullforce2 = np.zeros_like(atoms.get_positions())
                if len(self._partCalcs[parttype]) > 1:
                    sub_calc_1 = self._partCalcs[parttype][1]
                    calc_atoms.calc = sub_calc_1
                    sub_calc_1.calculate(atoms=calc_atoms, properties=['energy', 'forces'], system_changes=system_changes)
                    energy2 = sub_calc_1.results['energy']
                    force2 = sub_calc_1.results['forces']
                    print(f'{parttype} 1 max force norm: {np.max(np.linalg.norm(force2, axis=1))}')
                    fullforce2 = self.reduce_calcForces(parttype, force2)

            # 4. 记录 energy1, energy2 和 force1, force2
            self._result[parttype]['energy'] = (energy1, energy2)
            self._result[parttype]['force'] = (fullforce1, fullforce2)

        # 5. 累加总能量和总力
        total_energy = 0.0
        total_forces = np.zeros_like(self._atomCoords)
        for parttype in self._partTypeUniq:
            total_energy += self._result[parttype]['energy'][0] - self._result[parttype]['energy'][1]
        self.results['energy'] = total_energy
        for parttype in self._partTypeUniq:
            total_forces += self._result[parttype]['force'][0] - self._result[parttype]['force'][1]
        self.results['forces'] = total_forces

        self._result['total_energy'] = total_energy
        self._result['total_forces'] = total_forces

        _debug_print(f"total_energy: {total_energy}")
        _debug_print(f"total_forces: {total_forces}")


    def analysis(self, atoms, iterator, custom_loggor):
        custom_loggor.print(f'=== [Step: {iterator.nsteps:6d}] Part Setups and Analysis ===')
        with Timing("part analysis"):
            total_forces = self._result['total_forces']
            total_norm = np.linalg.norm(total_forces, axis=1)
            sym = np.array(atoms.get_chemical_symbols())
            top_idx = np.argsort(-total_norm)[:6]
            top_ele = sym[top_idx]
            top_flag =[]

            for parttype in self._partTypeUniq:
                linkList1 = self._partInfo[parttype]['linkList1']
                linkList2 = self._partInfo[parttype]['linkList2']
                custom_loggor.print(f'{parttype} linkList1: {linkList1}')
                custom_loggor.print(f'{parttype} linkList2: {linkList2}')

            for idx in top_idx:
                for parttype in self._partTypeUniq:
                    linkList1 = self._partInfo[parttype]['linkList1']
                    linkList2 = self._partInfo[parttype]['linkList2']
                    fullList = self._partInfo[parttype]['fullList']
                    if idx in linkList1:
                        top_flag.append(f'K{parttype}')
                        break
                    elif idx in linkList2:
                        top_flag.append(f'L{parttype}')
                        break
                    elif idx in fullList:
                        top_flag.append(f'{parttype}')
                        break
            
            custom_loggor.print(f'top norm: {total_norm[top_idx]}')
            custom_loggor.print(f'top idx : {top_idx}')
            custom_loggor.print(f'top ele : {top_ele}')
            custom_loggor.print(f'top flag: {top_flag}')

            total_energy = self._result['total_energy']
            custom_loggor.print(f'total_energy: {total_energy}')
            for parttype in self._partTypeUniq:
                energy1, energy2 = self._result[parttype]['energy']
                force1, force2 = self._result[parttype]['force']
                norm1 = np.linalg.norm(force1, axis=1)
                norm2 = np.linalg.norm(force2, axis=1)
                custom_loggor.print(f'part {parttype} energy1: {energy1}')
                custom_loggor.print(f'part {parttype} energy2: {energy2}')
                custom_loggor.print(f'part {parttype} force1 norm: {norm1[top_idx]}')
                custom_loggor.print(f'part {parttype} force2 norm: {norm2[top_idx]}')


    @classmethod
    def PARSE_MASK_FILE(cls, file: str, atomCoords: np.ndarray, cutoff: float = 1.8):
        '''
        Parse a “.mask” file that lists every atom and its part type.

        File format (white-space separated, one atom per line):
        -------------------------------------------------------
        # atomLabel  partType(int start from 0)
        C1   1
        O2   2
        N3   3
        …

        atomLabel : arbitrary string that identifies the atom
        partType : single character indicating the part (from 1 to count)
                    1  –  high part: such as high precision QM part
                    2  –  medium part: such as medium precision QM part
                    3  –  low part: such as low precision MM part
                    ...

        The routine also builds neighbor lists: two atoms are considered
        neighbors if their distance is below the supplied cutoff.  When
        the two neighbors belong to different parts the pair is stored
        as a “link” (candidate for a capping atom or a link-atom scheme).

        Returns
        -------
        atomLabels   – list(str)        labels in file order
        partTypes   – list(str)        part flags in file order
        atomLinks    – list[list[int]]  for each atom, indices of
                                        partner atoms in different parts
                                        (index start from 0)
        atomNeighbors– list[list[int]]  for each atom, indices of all
                                        neighbors within cutoff
                                        (index start from 0)
        '''
        assert file.endswith('.mask'), "file must be a mask file"
        assert os.path.exists(file), f"file {file} does not exist"
        assert len(atomCoords.shape) == 2 and atomCoords.shape[1] == 3, "atomCoords must be (n_atoms, 3)"
        assert cutoff > 1.0, "cutoff must be positive"
        n_atoms = atomCoords.shape[0]
        atomLabels = []; partTypes = []
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '': continue
                line = line.strip().split()
                atomLabels.append(line[0])
                partTypes.append(int(line[1]) - 1)
        assert n_atoms == len(atomLabels), "n_atoms must be equal to len(atomLabels)"
        assert n_atoms == len(partTypes), "n_atoms must be equal to len(partTypes)"

        ones_atoms = np.ones(n_atoms)
        distance_matrix = np.sqrt( np.sum((np.einsum('ai,b->abi', atomCoords, ones_atoms) 
            - np.einsum('a,bi->abi', ones_atoms, atomCoords))**2, axis=2) )
        eps = 1.0e-8
        atomLinks = [[] for i in range(n_atoms)]
        atomNeighbors = [[] for i in range(n_atoms)]
        neighbor_mask = (distance_matrix - cutoff) * (distance_matrix - eps) < 0
        neighbor_pair_index = np.argwhere(neighbor_mask)
        for pair in neighbor_pair_index:
            if partTypes[pair[0]] != partTypes[pair[1]]: atomLinks[pair[0]].append(pair[1]) # pair[1] is the link of pair[0]
            atomNeighbors[pair[0]].append(pair[1]) # pair[1] is the neighbor of pair[0]
            
        return atomLabels, partTypes, atomLinks, atomNeighbors

    @classmethod
    def DUMP_MASK_FILE(cls, atomLabels: list, partTypes: list):
        """
        Dump the mask block to a string.
        """
        mask_block = '# atomLabel  partType\n'
        for i in range(len(atomLabels)):
            mask_block += f'{atomLabels[i]}  {partTypes[i] + 1:2d}\n'
        return mask_block

    @classmethod
    def PARSE_PART_FILE(cls, file: str):
        """
        Parse a “.part” file that lists every atom and its part type.

        File format (white-space separated, one atom per line, linkAtomIndex=0 means no link atom,
        generally multiple link atoms is not forbidden, but suggest only one link atom is used):
        -------------------------------------------------------
        # atomLabel  partType  linkAtomIndex1  linkAtomIndex2 ...
        C1   1   2   ...
        O2   2   1   ...
        N3   3   0   ...
        …

        atomLabel : arbitrary string that identifies the atom
        partType : single character indicating the part (from 1 to count)
                    1  –  high part: such as high precision QM part
                    2  –  medium part: such as medium precision QM part
                    3  –  low part: such as low precision MM part
                    ...
        linkAtomIndex1 : index of the link atom in the high part (0 if none)
        linkAtomIndex2 : index of the link atom in the low part (0 if none)

        The routine also builds neighbor lists: two atoms are considered
        neighbors if their distance is below the supplied cutoff.  When
        the two neighbors belong to different parts the pair is stored
        as a “link” (candidate for a capping atom or a link-atom scheme).

        Returns
        -------
        atomLabels   – list(str)          labels in file order
        partTypes   – list(int)          part types in file order (conresponding partCalcs)
        atomLinks    – list[list[int]]    for each atom, indices of partner atoms in different parts
        """
        assert file.endswith('.part'), "file must be a part file"
        assert os.path.exists(file), f"file {file} does not exist"
        atomLabels, partTypes, atomLinks = [], [], []
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '': continue
                line = line.strip().split()
                atomLabels.append(line[0])
                partTypes.append(int(line[1]) - 1)
                atomLinks.append([int(line[i]) - 1 for i in range(2, len(line)) if int(line[i]) > -1])
        return atomLabels, partTypes, atomLinks


    @classmethod
    def DUMP_PART_FILE(cls, atomLabels: list, partTypes: list, atomLinks: list[list[int]]):
        """
        Dump the part block to a string.
        """
        part_block = '# atomLabel  partType  linkAtomIndex1  linkAtomIndex2 ...\n'
        for i in range(len(atomLabels)):
            part_block += f'{atomLabels[i]}  {partTypes[i] + 1:2d}  {" ".join([str(j+1) for j in atomLinks[i]])}\n'
        return part_block


    @classmethod
    def PART_FROM_MASK_WITH_COORDS(cls, fileMask: str, fileCoords: str):
        from coord import Coord
        atomCoords = Coord.readCoords(fileCoords)
        atomLabels, partTypes, atomLinks, _ = cls.PARSE_MASK_FILE(fileMask, atomCoords)
        return DUMP_PART_FILE(atomLabels, partTypes, atomLinks)

if __name__ == '__main__':
    from ReaxFFCalculator import ReaxFFCalculator_LAMMPS
    from OPLSAACalculator import OPLSAACalculator_LAMMPS
    from lammps_utils import load_lammps_data_0, parse_lammps_data_to_ase_atoms

    partCalcs = [
        [
            ReaxFFCalculator_LAMMPS(ff_file='reaxff.ff'), 
            OPLSAACalculator_LAMMPS(data_file='oplsaa1.data', tmp_dir='tmp_opls1')
        ],
        [OPLSAACalculator_LAMMPS(data_file='oplsaa2.data', tmp_dir='tmp_opls2')],
    ]

    with open('oplsaa2.data', 'r') as f:
        data = load_lammps_data_0(f.read())
    atoms = parse_lammps_data_to_ase_atoms(data)

    # partCalcs = [Calculator() for _ in range(2)]
    pcalc = PartitionedCalculator(partCalcs=partCalcs, partFile='system1.part')
    atoms.calc = pcalc
    Energy = atoms.get_potential_energy()
    print(Energy)
