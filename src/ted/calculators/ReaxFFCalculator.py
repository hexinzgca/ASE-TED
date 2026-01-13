from ase.calculators.lammpsrun import LAMMPS

from .lammps_utils import load_lammps_data_0, parse_lammps_data_to_ase_atoms
from .decorator_utils import Timing
import numpy as np
import os

os.environ['ASE_LAMMPSRUN_COMMAND'] = 'lmp'  # 确保lmp在系统PATH中

class ReaxFFCalculator_LAMMPS(LAMMPS):
    def __init__(self, ff_file, tmp_dir='tmp_reax', restrain_bond_topo=''):
        self.n_atoms = -1 # Any number of atoms
        self.tmp_dir = tmp_dir
        files = [ff_file]
        base_parameters = {
            'keep_tmp_files': False,
            'tmp_dir': tmp_dir,
            'units': 'real',
            'boundary': 'p p p',
            'atom_style': 'full',  # 支持拓扑，与你的ff文件匹配
            'pair_style': 'reaxff NULL',  # 直接将LAMMPS的pair_style值作为字符串传入
            'pair_coeff': [f'* * {ff_file} C H O\nfix charge all qeq/reax 1 0.0 10.0 1.0e-6 reaxff'],  # 若ff文件在当前目录，直接用文件名；否则用绝对路径
            # 'fix': ['charge all qeq/reax 1 0.0 10.0 1.0e-6 reaxff'],
            # 读取data文件（拓扑+原子坐标+原子类型）
            # 'read_data': data_file,
            # ===================== 核心修改：dump once 命令（无步数依赖，单次触发） =====================
            # 格式：dump <ID> <原子组> once <文件名> custom <输出属性>
            # 关键：once 表示「单次输出」，无需动力学步数，执行后直接生成 dump 文件
            'dump': ['charge_dump all once reaxff_charge.dump custom id type x y z charge'],
            # dump_modify 配置不变（排序原子ID，确保与ASE一致）
            'dump_modify': ['charge_dump sort id filetype atom'],
            # 基础 thermo 配置（确保能量/力计算正常）
            'thermo_style': 'custom step pe ke etotal temp press',
            'thermo': 1,
            # 'thermo_args': ['step', 'pe', 'ke', 'etotal', 'temp', 'press'],
        }

        self._result = {
            "energy": None,
            "force": None,
        }

        super().__init__(
            files = files,
            **base_parameters
        )

    def calculate(self, atoms, properties, system_changes):
        with Timing(f"reax calculate"):
            super().calculate(atoms, properties, system_changes)

            self._result['energy'] = self.results['energy']
            self._result['force'] = self.results['forces']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ReaxFFCalculator_LAMMPS')
    parser.add_argument('--data_file', '-d', type=str, default='oplsaa.data', help='lammps data file (only read atoms)')
    parser.add_argument('--ff_file', '-f', type=str, default='reaxff.ff', help='reaxff force field file')

    args = parser.parse_args()

    with open(args.data_file, 'r') as f:
        data = load_lammps_data_0(f.read())
        atoms = parse_lammps_data_to_ase_atoms(data)

    lammps = ReaxFFCalculator_LAMMPS(ff_file = args.ff_file, restrain_bond_topo=args.data_file)
    lammps.set_kscale(0.1)
    atoms.calc = lammps

    energy1 = atoms.get_potential_energy()
    force1 = atoms.get_forces()
    print("Energy ", energy1)
    print(f'max force norm: {np.max(np.linalg.norm(force1, axis=1))}')

    xyz = atoms.get_positions()
    dxyz = xyz*0 + 0.01
    dxyz[90:] = 0.0
    xyz += dxyz
    atoms.set_positions(xyz)
    energy2 = atoms.get_potential_energy()
    force2 = atoms.get_forces()
    print("Energy ", energy2)
    print(f'max force norm: {np.max(np.linalg.norm(force2, axis=1))}')
    print("Energy_Diff ", energy2 - energy1)
    print("Forces*dR ", np.sum(0.5*(force2+force1)*dxyz))


