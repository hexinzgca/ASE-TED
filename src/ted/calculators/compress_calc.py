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

class CompressCalculator(Calculator):
    """
    Calculator for compress potential.
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, calc: Calculator, L0: float, Lend: float):
        self._calc = calc
        self._L0 = L0
        self._Lend = Lend
        self.enable_compress = True
        super().__init__()

    @property
    def calc(self):
        return self._calc

    @debug_helper(enable=True, print_args=False, print_return=False)
    def calculate(self, atoms, properties, system_changes):
        with Timing("compress calculate"):
            self._calc.calculate(atoms=atoms, properties=['energy', 'forces'], system_changes=['positions'])
            energy = self._calc.results['energy']
            forces = self._calc.results['forces']
            stress = self._calc.results['stress']
            self.results['energy'] = energy
            self.results['forces'] = forces
            self.results['stress'] = stress
            
    @debug_helper(enable=True, print_args=False, print_return=False)
    def compress(self, atoms, iterator, custom_loggor, total_steps: int):
        with Timing("setup compress interaction"):
            if not self.enable_compress:
                return
                
            Lt = self._L0 + (self._Lend - self._L0) * (iterator.nsteps / total_steps)
            print(f"Lt: {Lt:.4f}")
            atoms.set_cell([Lt, Lt, Lt], scale_atoms=True)
