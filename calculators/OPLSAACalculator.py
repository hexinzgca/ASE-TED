import os
from .lammpsrun_alter import LAMMPS
from .lammps_utils import load_lammps_data_0, update_lammps_data, parse_lammps_data_to_ase_atoms
from .decorator_utils import Timing
import numpy as np

os.environ['ASE_LAMMPSRUN_COMMAND'] = 'lmp'  # 确保lmp在系统PATH中

OPLSAACalculator_IMPL = 'LAMMPS'

# # OPLSAACalculator_IMPL = 'OPENMM'
# class OPLSAACalculator_OPENMM(OPENMM):
#     def __init__(self) -> None:
#         super().__init__()

class OPLSAACalculator_LAMMPS(LAMMPS):
    def __init__(self, data_file, tmp_dir='tmp_opls'):
        with open(data_file, 'r') as f:
            self.ref_data = load_lammps_data_0(f.read())
            self.ref_data = update_lammps_data(self.ref_data, update_atom_index=True)

        self.n_atoms = len(self.ref_data['Atoms'])
        self.tmp_dir = tmp_dir

        files = [data_file]
        base_parameters = {
            'keep_tmp_files': False,
            'tmp_dir': tmp_dir,
            'units': 'real',
            'boundary': 'p p p',
            'atom_style': 'full',  # 支持拓扑，与你的ff文件匹配
            #======================
            # 非键相互作用（pair_style与ff文件Pair Coeffs匹配）
            'pair_style': 'lj/cut/coul/long 12.0',
            'pair_modify': 'mix geometric',  # OPLS专用混合规则，与你的ff文件匹配
            'special_bonds': 'lj/coul 0.0 0.0 0.5',  # OPLS分子内作用修正
            'kspace_style': 'pppm 1.0e-4',
            # 拓扑相互作用（style与ff文件对应区块匹配）
            'bond_style': 'harmonic',       # 与ff文件Bond Coeffs匹配
            'angle_style': 'harmonic',       # 与ff文件Angle Coeffs匹配
            'dihedral_style': 'opls',        # 与ff文件Dihedral Coeffs匹配（OPLS格式）
            'improper_style': 'cvff',        # 与ff文件Improper Coeffs匹配（CVFF格式）
            # 读取data文件（拓扑+原子坐标+原子类型）
            'read_data': data_file,
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
        with Timing(f"opls calc (dir={self.tmp_dir})"):
            super().calculate(atoms, properties, system_changes)

            self._result['energy'] = self.results['energy']
            self._result['force'] = self.results['forces']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OPLSAACalculator_LAMMPS')
    parser.add_argument('--data_file', '-d', type=str, default='oplsaa.data', help='lammps data file')

    args = parser.parse_args()

    with open(args.data_file, 'r') as f:
        data = load_lammps_data_0(f.read())
        data = update_lammps_data(data, update_atom_index=True)
        atoms = parse_lammps_data_to_ase_atoms(data)

    lammps = OPLSAACalculator_LAMMPS(data_file = args.data_file)
    atoms.calc = lammps

    energy1 = atoms.get_potential_energy()
    force1 = atoms.get_forces()
    print("Energy ", energy1)
    print(f'max force norm: {np.max(np.linalg.norm(force1, axis=1))}')
    # print("Forces ", force1)

    exit(0)

    xyz = atoms.get_positions()
    dxyz = xyz*0 + 0.01
    dxyz[1000:] = 0.0
    xyz += dxyz
    atoms.set_positions(xyz)
    energy2 = atoms.get_potential_energy()
    force2 = atoms.get_forces()
    print("Energy ", energy2)
    # print("Forces ", force2)
    print("Energy_Diff ", energy2 - energy1)
    print("Forces*dR ", np.sum(0.5*(force2+force1)*dxyz))

    

