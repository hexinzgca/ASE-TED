from ase.calculators.lammpsrun import LAMMPS
from utils.unit_conversion import lammps_real_to_ase


class ReaxFFQMCalculator(LAMMPS):
    """基于LAMMPS ReaxFF的QM计算器"""
    def __init__(self, ffield_path, elements, executable, **kwargs):
        """
        参数：
            ffield_path: ReaxFF力场文件路径（如CHO_reaxff.ffield）
            elements: QM区域元素列表（如['C','H','O']）
            executable: LAMMPS可执行文件路径
            **kwargs: 传递给LAMMPS计算器的参数
        """
        # ReaxFF核心参数
        parameters = {
            'pair_style': 'reaxff NULL',
            'pair_coeff': [f'* * {ffield_path} {" ".join(elements)}']
        }
        # 合并用户自定义参数
        if 'parameters' in kwargs:
            parameters.update(kwargs.pop('parameters'))
        
        super().__init__(
            executable=executable,
            parameters=parameters,
            units='real',
            keep_tmp_files=False,
            **kwargs
        )

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """重写能量获取，自动转换单位"""
        energy_lmp = super().get_potential_energy(atoms, force_consistent)
        energy_ase, _ = lammps_real_to_ase(energy_lmp, 0)
        self.results['energy'] = energy_ase
        return energy_ase

    def get_forces(self, atoms=None):
        """重写力获取，自动转换单位"""
        forces_lmp = super().get_forces(atoms)
        _, forces_ase = lammps_real_to_ase(0, forces_lmp)
        self.results['forces'] = forces_ase
        return forces_ase