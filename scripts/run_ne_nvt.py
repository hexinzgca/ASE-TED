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
from ase.io import read, write

def lammps_input(filename_in='initial.data', filename_out='final.dump', ensemble='npt', step=100, nsamp = 1):
    return f"""
log log.lammps

atom_style full 
units real
boundary p p p 
box tilt large
atom_modify sort 0 0.0
timestep         0.5

# 先定义临时区域（region）, 尺寸随便填（会被覆盖）
region temp_box block 0 1 0 1 0 1 units box
create_box 3 temp_box

# 读取dump文件（包含初始坐标和速度）
read_dump  {filename_in} 0 x y z vx vy vz box yes add yes

mass 1 12.011000
mass 2 4.032000
mass 3 15.999000

pair_style reaxff NULL
pair_coeff * * reaxff.ff C H O
fix charge all qeq/reax 1 0.0 10.0 1.0e-6 reaxff
newton on

# 快速计算优化：邻居列表设置
neighbor        0.5 bin
neigh_modify    every 10 delay 0 check no one 10000 page 100000

fix             1 all {ensemble} temp 360.0 360.0 100.0 {'' if ensemble == 'nvt' else 'iso 1.0 1.0 250.0'}
thermo_style    custom step temp press pe ke etotal vol density # 实时监控密度变化
thermo          1
dump            1 all custom {step//nsamp} {filename_out} id type x y z vx vy vz
dump_modify     1 sort id flush yes {'' if nsamp == 1 else 'append yes'}
run             {step}

write_restart   final.restart
    """

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from ted.calculators.ReaxFFCalculator import ReaxFFCalculator_LAMMPS
from ted.calculators.OPLSAACalculator import OPLSAACalculator_LAMMPS
from ted.calculators.partitioned_calc import PartitionedCalculator
from ted.calculators.neffonly_calc import NeFFOnlyCalculator
from ted.integrators.langevin_nvt import LangevinBAOAB
from ted.calculators.lammps_utils import parse_lammps_data_to_ase_atoms, load_lammps_data_0, update_lammps_data
from ted.calculators.decorator_utils import Timing
from ted.calculators.lammps_utils import ase2lammps_dump
from ted.calculators.lammps_utils import lammps2ase_dump

parser = argparse.ArgumentParser(description="Non Equilibrium - Partitioned Region Dynamics (ReaxFF/MatterSim):MM Simulation")
parser.add_argument("--solver", "-s", type=str, nargs="+", default=["ReaxFF", "OPLSAA"], 
                    help="List of solver names [inner -> outer partitions]")
parser.add_argument("--flag", "-f", type=str, default='ne_system1', help="system flags")
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

def write_frame(filename: str, atoms: Atoms, append: bool = True):
    assert filename.endswith('.xyz'), 'filename must end with .xyz'
    write(filename, atoms, append=append)
    with Trajectory(filename.replace('.xyz', '.traj'), mode='a') as traj:
        traj.write(atoms)

logger.info(f"FLAG       Temperature(K)        Ekin(eV)        Epot(eV)     Volume(A^3)     Rho(g/cm^3)")
def log_atoms_information(atoms: Atoms, flag: str):
    masses_true = atoms.get_masses().copy()
    for i in range(len(atoms)):
        if atoms[i].symbol == 'H': masses_true[i] = 1.0080 # reset H-atoms masses to 1.0080 amu
    density = masses_true.sum() / atoms.get_volume() / (0.001*units.kg) * (0.01*units.m)**3
    logger.info(f"{flag} {atoms.get_temperature():>15.2f} {atoms.get_kinetic_energy():>15.4f} {atoms.get_potential_energy():>15.4f} {atoms.get_volume():>15.2f} {density:>15.4f}")

cmd = f'zsh -c "cd {args.flag} && source ~/.zshrc && lmp < run.in > log.lammps"'

if __name__ == "__main__":
    config = {
        "global": {
            "timestep": 0.5,      # (ase time unit fs?)
            "temperature": 360.0, # in Kelvin
            "steps": 2000000,
            "interval": 400,
        },
    }
    if os.path.exists(args.input): config.update(toml.load(args.input))

    flag = args.flag
    if os.path.exists(f"{flag}/run.log"): os.remove(f"{flag}/run.log")        
    logger.add(f"{flag}/run.log", rotation="10 MB", level="INFO")
    logger = logger.bind(name="Topo Enhenced Dynamics (for ReaxFF)")

    # step 1: built ASE atoms
    with open(f'{flag}/oplsaa2_react.data', 'r') as f:
        data = load_lammps_data_0(f.read())
        data = update_lammps_data(data, update_atom_index=True)
    atoms = parse_lammps_data_to_ase_atoms(data)
    logger.info(f"\nProcessing Number of atoms: {len(atoms)}")
    masses = atoms.get_masses()
    logger.info(f"\nProcessing masses: {masses}")
    print('for statistics, here brute force reset H-atoms masses to a larger one! x 6.0')
    for i in range(len(atoms)):
        if atoms[i].symbol == 'H': masses[i] *= 4.0
    atoms.set_masses(masses)
    logger.info(f'\nProcessing masses after reset H-atoms: {masses}')

    neffonly_calc = NeFFOnlyCalculator(
        NeFFOnly_file=f'{flag}/neff.neff',
        bond_topo_file=f'{flag}/oplsaa2_react.data',
        work_record_file=f'{flag}/neff.work',
        bond_record_file=f'{flag}/neff.bond'
    )
    atoms.calc = neffonly_calc

    class CustomLogger:
        def __init__(self, filename: str):
            self.fileio = open(filename, 'a')
        def print(self, msg):
            self.fileio.write(msg + '\n')
        def __del__(self):
            self.fileio.close()
    if os.path.exists(f'{flag}/neff.log'): os.remove(f'{flag}/neff.log')
    if os.path.exists(f'{flag}/part.log'): os.remove(f'{flag}/part.log')
    neff_logger = CustomLogger(filename=f'{flag}/neff.log')
    part_logger = CustomLogger(filename=f'{flag}/part.log')

    T_tau = 100.0 * units.fs        # 控温弛豫时间：50 fs（经验值，控温越紧值越小）
    P_tau = 250.0 * units.fs        # 控压弛豫时间：1000 fs（经验值，压浴弛豫通常比热浴慢）
    integrator = LangevinBAOAB(
        atoms=atoms,
        timestep=config["global"]["timestep"] * units.fs,
        temperature_K=config["global"]["temperature"],  # K
        # externalstress=externalstress,  # NPT 控压（1 atm）
        # hydrostatic=True,   # 仅体积变化，保持晶胞形状
        P_mass_factor=1.0,  # 压浴质量系数（默认即可）
        disable_cell_langevin=True, # 关闭晶胞的 Langevin 控温
        rng=np.random.default_rng(), # no seed!!!
    )
    integrator.attach(neffonly_calc.analysis, interval=1, atoms=atoms, iterator=integrator, 
            custom_loggor=neff_logger, noneq=True)

    # step 2: run NPT simulation
    interval = config["global"]["interval"]
    ncycle = config["global"]["steps"] // interval

    for fn in [f'{flag}/traj_ne_npt.xyz', f'{flag}/traj_ne_npt.traj', f'{flag}/log_merge.lammps']:
        if os.path.exists(fn): os.remove(fn)

    for icycle in range(ncycle):
        at_mid = True if icycle == ncycle//2 else False
        after_mid = True if icycle >= ncycle//2 else False
        neffonly_calc.at_mid = at_mid
        neffonly_calc.after_mid = after_mid

        # load atoms
        if os.path.exists(f'{flag}/final.dump'):
            initial_atoms = lammps2ase_dump(f'{flag}/final.dump')
            atoms.set_positions(initial_atoms.get_positions())
            atoms.set_velocities(initial_atoms.get_velocities())
            atoms.set_cell(initial_atoms.get_cell())
        
        if icycle%5 == 0: write_frame(f'{flag}/traj_ne_npt.xyz', atoms)
        log_atoms_information(atoms, f'N0-{icycle*config["global"]["interval"]}')
        
        # NeFF STEER RUN
        integrator.run(interval//40)
        log_atoms_information(atoms, f'N1-{icycle*config["global"]["interval"]}')

        # LAMMPS RUN
        ase2lammps_dump(atoms, f'{flag}/initial.dump')
        with open(f'{flag}/run.in', 'w') as f:
            f.write(lammps_input(filename_in='initial.dump', filename_out='final_lmp.dump', ensemble='nvt',
                nsamp=1, step=interval))
            f.close()
        try:
            ret_code = os.system(cmd)
            os.system(f'cd {flag} && cat log.lammps | grep -B1 Loop | head -n1 >> log_merge.lammps')
        except:
            logger.error(f'\nProcessing NPT simulation failed at cycle {icycle}')
            exit(-1)
        
        if os.path.exists(f'{flag}/final_lmp.dump'):
            initial_atoms = lammps2ase_dump(f'{flag}/final_lmp.dump')
            atoms.set_positions(initial_atoms.get_positions())
            atoms.set_velocities(initial_atoms.get_velocities())
            atoms.set_cell(initial_atoms.get_cell())
        log_atoms_information(atoms, f'N2-{icycle*config["global"]["interval"]}')

        # NeFF STEER RUN
        integrator.run(interval//40)
        log_atoms_information(atoms, f'N3-{icycle*config["global"]["interval"]}')
        ase2lammps_dump(atoms, f'{flag}/final.dump')
    write_frame(f'{flag}/traj_ne_npt.xyz', atoms)
    ase2lammps_dump(atoms, f'{flag}/initial_stage2.dump')

    # final run npt:
    with open(f'{flag}/run.in', 'w') as f:
        f.write(lammps_input_final(filename_in='initial_stage2.dump', filename_out='final_stage2.dump', ensemble='nvt',
            nsamp=100, step=config["global"]["steps"]))
        f.close()
    try:
        ret_code = os.system(cmd)
    except:
        logger.error(f'\nProcessing NPT simulation failed at cycle {icycle}')
        exit(-1)

    Timing.report()
    total_steps = config["global"]["steps"]
    timestep_in_fs = config["global"]["timestep"]
    speed = total_steps / Timing.timers["NeFF Molecular Dynamics"][1] # use wall time
    speed *= timestep_in_fs * 1e-6 * 86400.0  # convert step/s to ns/day
    logger.info(f'NeFF Molecular Dynamics Speed: {speed:.6f} ns / day')
