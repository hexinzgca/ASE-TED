import os, sys
import numpy as np
from ase import units, Atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from loguru import logger
import argparse
import toml
import copy
import pandas as pd

current_script_dir = os.getcwd()
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from ted.calculators.ReaxFFCalculator import ReaxFFCalculator_LAMMPS
from ted.calculators.OPLSAACalculator import OPLSAACalculator_LAMMPS
from ted.calculators.partitioned_calc import PartitionedCalculator
from ted.calculators.compress_calc import CompressCalculator
from ted.integrators.langevin_nvt import LangevinBAOAB
from ted.calculators.lammps_utils import parse_lammps_data_to_ase_atoms, load_lammps_data_0, update_lammps_data
from ted.calculators.decorator_utils import Timing

parser = argparse.ArgumentParser(description="Compress a system using ReaxFF Simulation")
parser.add_argument("--solver", "-s", type=str, nargs="+", default=["ReaxFF", "OPLSAA"], 
                    help="List of solver names [inner -> outer partitions]")
parser.add_argument("--flag", "-f", type=str, default='small1', help="system flags")
parser.add_argument("--reaxff", "-rf", type=str, default="data/reaxff/CHON_reaxff.ffield", 
                    help="Path to ReaxFF force-field file (lammps format)")
parser.add_argument("--oplsaa", "-op", type=str, default="data/oplsaa/CHON_oplsaa.ffield", 
                    help="Path to OPLSAA force-field file (lammps format)")
parser.add_argument("--restart", '-rt', type=str, default="", help="Restart from a previous trajectory file")
parser.add_argument("--uniqname", "-un", type=str, default="",   help="Unique name for the system")
parser.add_argument("--partition", "-p", type=str, default="",   help="Default partition file name: uniqname.part")
parser.add_argument("--neff", "-n", type=str, default="", help="Default non-equilibrium force-field file name: uniqname.neff")
parser.add_argument("--constraint", "-ct", type=str, default="", help="Default constraint definition file name: uniqname.const")
parser.add_argument("--thermo", "-th", type=str, default="", help="Default thermostat definition file name: uniqname.thermo")
parser.add_argument("--coord", "-c", type=str, default="", help="Default coordinate file path: uniqname.xyz")
parser.add_argument("--input", "-i", type=str, default="", help="Default input configuration file path: uniqname.toml")
parser.add_argument("--dump", "-d", type=str, default="", help="Default dump configuration file path: uniqname.dump")
parser.add_argument("--log", "-l", type=str, default="", help="Default log file path: uniqname.log")
parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
args = parser.parse_args()

class CustomLogger:
    def __init__(self, filename: str):
        self.fileio = open(filename, 'a')
    def print(self, msg):
        self.fileio.write(msg + '\n')
    def __del__(self):
        self.fileio.close()

if __name__ == "__main__":
    config = {
        "global": {
            "timestep": 0.5,      # (ase time unit fs?)
            "temperature": 360.0, # in Kelvin
            "steps": 4000,
            "comp_steps": 20,
            "interval": 40,
        },
    }
    if os.path.exists(args.input): config.update(toml.load(args.input))

    flag = 'interact_system1'
    if os.path.exists(f"{flag}/run.log"): os.remove(f"{flag}/run.log")        
    logger.add(f"{flag}/run.log", rotation="10 MB", level="INFO")
    logger = logger.bind(name="Interaction Analysis (for ReaxFF)")

    # EMA: 6322x4
    # PEG: 143x160
    # total: 48168 atoms
    idx_list = []
    for i in range(4):
        idx_list += [list(range(i*6322, (i+1)*6322))]
    for i in range(160):
        idx_list += [list(range(6322*4 + i*143, 6322*4 + (i+1)*143))]
    # print(idx_list)

    # step 1: built ASE atoms
    with open(f'{flag}/pack_mol.data', 'r') as f:
        data = load_lammps_data_0(f.read())
        data = update_lammps_data(data, update_atom_index=True)
    atoms = parse_lammps_data_to_ase_atoms(data)

    

    if os.path.exists(f'{flag}/ana.xyz'):
        from ase.io import read
        from ase import Atoms
        traj = read(f'{flag}/ana.xyz', index=':')
    
    nframe = len(traj)
    data = np.zeros((nframe, 6))

    reax_calc0 = ReaxFFCalculator_LAMMPS(ff_file=f'{flag}/reaxff.ff', tmp_dir=f'{flag}/tmp_reax1')

    for i in range(nframe):
        print('='*10 + f'{i}')
        atoms.set_positions(traj[i].get_positions())
        atoms.set_cell(traj[i].get_cell())
    
        cell = atoms.get_cell()
        atoms.set_calculator(reax_calc0)
        total_energy = atoms.get_potential_energy()
        print(f"total_energy: {total_energy:.4f}")

        def copy_atom(atoms: Atoms, rlist: list):
            # print(rlist)
            new_atom = Atoms(symbols=np.array(atoms.get_chemical_symbols())[rlist], 
                    positions=atoms.get_positions()[rlist,:], 
                    cell=atoms.get_cell())
            return new_atom
        
        part_energy = []
        for mpart in idx_list:
            m_atoms = copy_atom(atoms, mpart)
            m_atoms.calc = reax_calc0
            m_energy = m_atoms.get_potential_energy()
            part_energy.append(m_energy)
            # print(f"m_energy: {m_energy:.4f}")
        
        c1_list = []
        for mpart in idx_list[:4]:
            c1_list += list(mpart)
        c1_atoms = copy_atom(atoms, c1_list)
        c1_atoms.calc = reax_calc0
        c1_energy = c1_atoms.get_potential_energy()
        print(f"c1_energy: {c1_energy:.4f}")
        print(f"c1 bind energy: {c1_energy - sum(part_energy[:4]):.4f}")

        c2_list = []
        for mpart in idx_list[4:]:
            c2_list += list(mpart)
        c2_atoms = copy_atom(atoms, c2_list)
        c2_atoms.calc = reax_calc0
        c2_energy = c2_atoms.get_potential_energy()
        print(f"c2_energy: {c2_energy:.4f}")
        print(f"c2 bind energy: {c2_energy - sum(part_energy[4:]):.4f}")
        print(f"c1-c2 bind diff: {total_energy - c1_energy - c2_energy:.4f}")

        data[i, 0] = total_energy
        data[i, 1] = c1_energy
        data[i, 2] = c1_energy - sum(part_energy[:4])
        data[i, 3] = c2_energy
        data[i, 4] = c2_energy - sum(part_energy[4:])
        data[i, 5] = total_energy - c1_energy - c2_energy

    pd.DataFrame(data).to_csv('binding_energy.csv')
