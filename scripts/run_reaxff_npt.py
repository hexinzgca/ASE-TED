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

def read_lammps_dump(filename, index=':'):
    # 方式1：读取整个dump文件（多帧），返回轨迹列表
    traj = read(filename, format='lammps-dump-text', index=index)  # index=':' 读取所有帧
    atoms = traj[-1]  # 取最后一帧
    # 验证读取结果
    # print("原子数量：", len(atoms))
    # print("晶格盒子：", atoms.get_cell())
    # print("原子坐标：", atoms.get_positions()[:3])  # 打印前3个原子的坐标
    # print("原子质量：", atoms.get_masses()[:3])  # 打印前3个原子的质量
    # 若dump包含速度，可读取速度
    # if atoms.get_velocities() is not None:
    #     print("原子速度：", atoms.get_velocities()[:3])  # 打印前3个原子的速度
    # else:
    #     print("⚠️  dump文件中未检测到速度信息，请检查dump文件是否包含vx/vy/vz字段")
    return atoms

def write_lammps_dump(filename, atoms):
    write(
        filename,                # 输出文件名
        atoms,                   # ASE Atoms对象
        format='lammps-data',    # 格式指定为LAMMPS data
        velocities=True,         # 保存速度（核心参数）
        units='real',            # 单位体系（需与LAMMPS脚本匹配，如metal/real）
        atom_style='full'
    )

def lammps_input(filename_in='initial.data', filename_out='final.dump', step=100):
    return f"""
atom_style full 
units real
boundary p p p 
box tilt large
atom_modify sort 0 0.0
timestep         0.5

# -------------------- 读取UFF格式data文件 --------------------
read_data       {filename_in}
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

fix             1 all npt temp 360.0 360.0 100.0 iso 1.0 1.0 250.0
thermo_style    custom step temp press pe ke etotal vol density # 实时监控密度变化
thermo          10
dump            1 all custom {step} {filename_out} id type x y z vx vy vz  # 只输出最后一步
dump_modify     1 sort id 
run             {step}

write_data      final.data 
    """

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from ted.calculators.ReaxFFCalculator import ReaxFFCalculator_LAMMPS
from ted.calculators.OPLSAACalculator import OPLSAACalculator_LAMMPS
from ted.calculators.partitioned_calc import PartitionedCalculator
from ted.calculators.neff_calc import NeFFCalculator
from ted.integrators.langevin_nvt import LangevinBAOAB
from ted.calculators.lammps_utils import parse_lammps_data_to_ase_atoms, load_lammps_data_0, update_lammps_data
from ted.calculators.decorator_utils import Timing

parser = argparse.ArgumentParser(description="Non Equilibrium - Partitioned Region Dynamics (ReaxFF/MatterSim):MM Simulation")
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

if __name__ == "__main__":
    config = {
        "global": {
            "timestep": 0.5,      # (ase time unit fs?)
            "temperature": 360.0, # in Kelvin
            "steps": 4000000,
            "interval": 100,
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

    # step 2: run NPT simulation
    ncycle = config["global"]["steps"] // config["global"]["interval"]
    for icycle in range(ncycle):
        with open(f'{flag}/run.in', 'w') as f:
            f.write(lammps_input(filename_in='initial.data', filename_out='final.dump', step=config["global"]["interval"]))
            f.close()
            
        try:
            os.system(f'cd {flag} && source env/bin/activate && lmp .... -in run.in')
        except:
            logger.error(f'\nProcessing NPT simulation failed at cycle {icycle}')
            exit(1)

        atoms = read_lammps_dump(f'{flag}/final.dump')
        neffplus_calc = NeFFPlusCalculator(
            neff_file=f'{flag}/neff.neff',
            bond_topo_file=f'{flag}/oplsaa2_react.data',
            work_record_file=f'{flag}/neff.work', 
            bond_record_file=f'{flag}/neff.bond'
        )
        atoms.calc = neffplus_calc
        
        # xyz = atoms.get_positions()
        # vel = atoms.get_velocities()
        # mass = atoms.get_masses()
        # neffplus_calc.calculate(atoms)
        # xyz = atoms.get_positions()

        T_tau = 100.0 * units.fs        # 控温弛豫时间：50 fs（经验值，控温越紧值越小）
        P_tau = 250.0 * units.fs        # 控压弛豫时间：1000 fs（经验值，压浴弛豫通常比热浴慢）
        integrator = LangevinBAOAB(
            atoms=atoms,
            timestep=config["global"]["timestep"] * units.fs,
            temperature_K=config["global"]["temperature"],  # K
            externalstress=externalstress,  # NPT 控压（1 atm）
            hydrostatic=True,   # 仅体积变化，保持晶胞形状
            P_mass_factor=1.0,  # 压浴质量系数（默认即可）
            disable_cell_langevin=False,  # 开启晶胞的 Langevin 控温
            rng=np.random.default_rng(), # no seed!!!
        )
        integrator.run(1)

    def init_atom_from_last_frame(atom, fn_traj):
        with Trajectory(fn_traj, mode='r') as traj:
            atom.set_positions(traj[-1].get_positions())
            atom.set_velocities(traj[-1].get_velocities())

    if args.restart:
        init_atom_from_last_frame(atoms, args.restart)
        logger.info(f'\nProcessing initial velocity from restart file!!!')
    else:
        vel = atoms.get_velocities()
        logger.info(f'\nProcessing initial velocity after reset H-atoms: {vel}')
        MaxwellBoltzmannDistribution(atoms, temperature_K=360.0)

    logger.info(f'\nProcessing initial velocity after reset H-atoms: {atoms.get_velocities()}')
    logger.info(f'Test atom periodic boundary condition: {atoms.get_pbc()}')

    
    # part_calc = PartitionedCalculator(partCalcs=subCalcs, partFile=f'{flag}/part.part')
    
    neff_calc = NeFFCalculator(calc=reax_calc0,
        neff_file=f'{flag}/neff.neff',
        bond_topo_file=f'{flag}/oplsaa2_react.data',
        work_record_file=f'{flag}/neff.work', 
        bond_record_file=f'{flag}/neff.bond'
    )

    # bind calculator to atoms
    atoms.calc = neff_calc

    def write_frame(filename: str, atoms: Atoms, append: bool = True):
        assert filename.endswith('.xyz'), 'filename must end with .xyz'
        write(filename, atoms, append=append)
        with Trajectory(filename.replace('.xyz', '.traj'), mode='a') as traj:
            traj.write(atoms)

    def log_atoms_information(atoms: Atoms, flag: str, iterator):
        if iterator.nsteps == 0:
            #                          ===============!===============!===============!===============!===============!
            logger.info(f"{flag}   Step  Temperature(K)        Ekin(eV)        Epot(eV)     Volume(A^3)     Rho(g/cm^3)")
        
        masses_true = atoms.get_masses().copy()
        for i in range(len(atoms)):
            if atoms[i].symbol == 'H': masses_true[i] = 1.0080 # reset H-atoms masses to 1.0080 amu
        density = masses_true.sum() / atoms.get_volume() / (0.001*units.kg) * (0.01*units.m)**3
        logger.info(f"{flag} {iterator.nsteps:>6d} {atoms.get_temperature():>15.2f} {atoms.get_kinetic_energy():>15.4f} {atoms.get_potential_energy():>15.4f} {atoms.get_volume():>15.2f} {density:>15.4f}")

    # run minimization here
    if not args.restart:
        with Timing("Minimization"):
            total_min_steps = config["global"]["min_steps"]
            logger.info(f"Starting FIRE minimization for {total_min_steps} steps...")
            dyn = FIRE(atoms, logfile=None, trajectory=None)

            if os.path.exists(f"{flag}/trajectory_min.xyz"): os.remove(f"{flag}/trajectory_min.xyz")
            dyn.attach(log_atoms_information, interval=1, atoms=atoms, flag="MIN", iterator=dyn)
            dyn.attach(write_frame, interval=1, filename=f"{flag}/trajectory_min.xyz", atoms=atoms)
            dyn.run(steps=total_min_steps)
            logger.info("* FIRE MINIMIZATION FINISHED!")
    else:
        logger.info(f'\nProcessing initial velocity from restart file so skip minimization!')

    # run NeFF molecular dynamics here
    with Timing("NeFF Molecular Dynamics"):
        # Initialize integrator
        integrator = LangevinBAOAB(
            atoms=atoms,
            timestep=config["global"]["timestep"] * units.fs,  # fs
            T_tau = 50 * config["global"]["timestep"] * units.fs,  # fs
            temperature_K=config["global"]["temperature"],  # K
            disable_cell_langevin=True,  # 仅NVT系综，禁用盒子 Langevin
            rng=np.random.default_rng(), # no seed!!!
        )
        logger.info(f"test random number: {integrator.rng.random()}")

        traj_path = f"{flag}/trajectory_sample.xyz"
        if os.path.exists(traj_path): os.remove(traj_path)
        if os.path.exists(traj_path.replace('.xyz', '.traj')): os.remove(traj_path.replace('.xyz', '.traj'))
        if os.path.exists(neff_calc._work_record_file): os.remove(neff_calc._work_record_file)
        if os.path.exists(neff_calc._bond_record_file): os.remove(neff_calc._bond_record_file)

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

        sample_interval = config["global"]["interval"]
        integrator.attach(write_frame, interval=sample_interval, filename=traj_path, atoms=atoms)
        # integrator.attach(part_calc.analysis, interval=1, atoms=atoms, iterator=integrator, 
        #     custom_loggor=part_logger)
        integrator.attach(neff_calc.analysis, interval=1, atoms=atoms, iterator=integrator, 
            custom_loggor=neff_logger, noneq=True) 
        integrator.attach(log_atoms_information, interval=10, atoms=atoms, flag="NVT", iterator=integrator)

        neff_calc.analysis(atoms=atoms, iterator=integrator,
            custom_loggor=neff_logger, noneq=True, only_initialize=True) # initialize q & k1 & k2

        total_steps = config["global"]["steps"]
        integrator.run(total_steps)
        logger.info("* NeFF MD FINISHED!")

    Timing.report()
    total_steps = config["global"]["steps"]
    timestep_in_fs = config["global"]["timestep"]
    speed = total_steps / Timing.timers["NeFF Molecular Dynamics"][1] # use wall time
    speed *= timestep_in_fs * 1e-6 * 86400.0  # convert step/s to ns/day
    logger.info(f'NeFF Molecular Dynamics Speed: {speed:.6f} ns / day')
