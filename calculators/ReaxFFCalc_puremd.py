import os
import subprocess
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
from typing import Optional, Dict, List


class PureMDReaxFFCalculator(Calculator):
    """
    基于 PuReMD 原生语法的 ReaxFF 力场 ASE 计算器
    完全兼容 PuReMD 输入格式，支持能量、受力计算，对标 LAMMPS ReaxFF ASE 接口
    依赖：PuReMD v3.0+、ASE、numpy
    """
    implemented_properties = ['energy', 'forces']
    default_parameters = {
        'fffile': None,          # 必需：ReaxFF 力场文件（*.ff，PuReMD 兼容格式）
        'puremd_exec': 'puremd', # PuReMD 可执行文件路径
        'tmp_dir': './puremd_tmp',# 临时文件目录
        'cleanup': True,         # 计算后清理临时文件
        'logfile': 'puremd_reaxff.log', # 运行日志
        'units': 'real',         # 单位体系：real(kcal/mol, Å) / metal(eV, Å)
        'bc': 'f f f'            # 边界条件：f(固定)/p(周期)，三方向空格分隔
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 校验核心参数
        self.fffile = self.parameters['fffile']
        if self.fffile is None or not os.path.exists(self.fffile):
            raise FileNotFoundError(f"ReaxFF 力场文件不存在: {self.fffile}")
        # 初始化临时目录
        self.tmp_dir = self.parameters['tmp_dir']
        os.makedirs(self.tmp_dir, exist_ok=True)
        # 缓存文件路径
        self.puremd_in = os.path.join(self.tmp_dir, 'puremd_reaxff.in')
        self.puremd_out = os.path.join(self.tmp_dir, 'puremd_reaxff.out')
        self.puremd_dump = os.path.join(self.tmp_dir, 'puremd_reaxff.dump')

    def _write_puremd_input(self, atoms) -> None:
        """
        生成 PuReMD 原生语法的输入文件（ReaxFF 单点计算专用）
        PuReMD 输入文件结构：控制参数 → 力场设置 → 原子坐标
        """
        natoms = len(atoms)
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        # 获取唯一元素（用于力场映射）
        unique_syms = list(set(symbols))

        # ===================== PuReMD 原生输入内容 =====================
        puremd_input = f"""# PuReMD 输入文件 (ASE 自动生成) - ReaxFF 单点能量/受力计算
# -------------------------- 控制参数段 --------------------------
simulation       qeq              # 启用 ReaxFF 必需的 QEq 电荷平衡
nsteps           0                # 0步动力学 = 单点计算（仅计算能量/力）
units            {self.parameters['units']}  # 单位体系
bc               {self.parameters['bc']}      # 边界条件
print_freq       1                # 输出频率
dump_freq        1                # Dump 文件输出频率
dump_file        {self.puremd_dump}  # 受力输出文件
dump_style       custom          # 自定义 Dump 格式
dump_properties  id species x y z fx fy fz  # Dump 输出内容

# -------------------------- 盒子参数段 --------------------------
lattice          orthogonal       # 正交盒子（ASE 原子结构默认）
lx               {cell[0,0]:.6f} # 盒子x方向长度
ly               {cell[1,1]:.6f} # 盒子y方向长度
lz               {cell[2,2]:.6f} # 盒子z方向长度
xy               0.0             # 非正交性参数（正交盒子为0）
xz               0.0
yz               0.0

# -------------------------- 力场参数段 --------------------------
forcefield       reaxff           # 力场类型：ReaxFF
ffield          {self.fffile}   # ReaxFF 力场文件路径
elements         {" ".join(unique_syms)}  # 体系包含的元素

# -------------------------- 原子坐标段 --------------------------
natoms           {natoms}         # 原子总数
coords_type      cartesian        # 笛卡尔坐标
"""
        # 写入原子坐标（格式：id species x y z）
        for idx, (sym, pos) in enumerate(zip(symbols, positions), 1):
            puremd_input += f"{idx} {sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
        # ===================== 输入文件结束 =====================

        # 写入文件
        with open(self.puremd_in, 'w', encoding='utf-8') as f:
            f.write(puremd_input)

    def _parse_puremd_output(self) -> Dict[str, np.ndarray]:
        """
        解析 PuReMD 输出：总能量（log） + 原子受力（dump）
        适配 PuReMD 原生输出格式
        """
        # 1. 解析总势能（从 PuReMD 输出日志）
        total_energy = None
        unit_conv = 1.0  # 单位转换系数：real(kcal/mol)→eV 需要×0.0433641
        if self.parameters['units'] == 'real':
            unit_conv = 0.0433641

        with open(self.puremd_out, 'r', encoding='utf-8') as f:
            for line in f:
                # PuReMD 输出的势能行格式："Potential Energy =  xxx.xxxx"
                if 'Potential Energy' in line:
                    try:
                        total_energy = float(line.strip().split('=')[-1]) * unit_conv
                        break
                    except (ValueError, IndexError):
                        continue
        if total_energy is None:
            raise RuntimeError("未找到势能数据，请检查 PuReMD 日志")

        # 2. 解析原子受力（从 Dump 文件）
        forces = []
        with open(self.puremd_dump, 'r', encoding='utf-8') as f:
            # 跳过 Dump 文件头部（PuReMD Dump 头部含版本/时间信息）
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            # 提取原子行（每行对应一个原子的 id species x y z fx fy fz）
            atom_lines = lines[1:] if len(lines) > 1 else lines  # 跳过首行标题
            for line in atom_lines[:natoms]:
                parts = line.split()
                fx, fy, fz = map(float, parts[-3:])
                forces.append(np.array([fx, fy, fz]) * unit_conv)  # 单位转换

        forces = np.array(forces)
        if forces.shape != (len(atoms), 3):
            raise RuntimeError(f"受力维度不匹配：预期 {len(atoms)}×3，实际 {forces.shape}")

        return {'energy': total_energy, 'forces': forces}

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.atoms = atoms.copy()
        natoms = len(atoms)

        # 1. 生成 PuReMD 输入文件
        self._write_puremd_input(atoms)

        # 2. 运行 PuReMD
        puremd_exec = self.parameters['puremd_exec']
        logfile = self.parameters['logfile']
        try:
            cmd = [puremd_exec, '-i', self.puremd_in, '-o', self.puremd_out]
            with open(logfile, 'w', encoding='utf-8') as log_f:
                subprocess.run(cmd, check=True, stdout=log_f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"PuReMD 运行失败，请查看日志 {logfile}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"PuReMD 可执行文件未找到: {puremd_exec}")

        # 3. 解析结果
        results = self._parse_puremd_output()
        self.results['energy'] = results['energy']
        self.results['forces'] = results['forces']

        # 4. 清理临时文件
        if self.parameters['cleanup']:
            for f in [self.puremd_in, self.puremd_out, self.puremd_dump]:
                if os.path.exists(f):
                    os.remove(f)
            if not os.listdir(self.tmp_dir):
                os.rmdir(self.tmp_dir)
        