import numpy as np

# 在integrators/nhc_nvt.py开头添加
import importlib
import ase
import ase.md

ase_version = ase.__version__
major, minor = map(int, ase_version.split('.')[:2])

from ase.md import NoseHooverChain  # ASE 3.26+ 正确导入路径
from ase.units import fs, ps
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet
# from ase.md.nosehoover import NoseHoover
# from ase.md.nosehoover_chain import NoseHooverChain

class TemperatureRampNHC:
    """
    带升温/降温的Nose-Hoover Chain NVT积分器
    支持线性升温/降温、恒温保持
    """
    def __init__(self, atoms, timestep, temp_profile, nchain=4, taut=0.5):
        """
        参数：
            atoms: Atoms对象（需附加NEQMMMCalculator）
            timestep: 时间步长（fs）
            temp_profile: 温度剖面 [(t1, T1), (t2, T2), ...]，按时间升序排列
            nchain: NHC链长度（默认4）
            taut: NHC弛豫时间（ps）
        """
        self.atoms = atoms
        self.timestep = timestep  # fs
        self.temp_profile = temp_profile
        self.nchain = nchain
        self.taut = taut  # ps → 转换为fs
        self.time = 0.0  # 累计模拟时间（fs）
        
        # 初始化NHC积分器
        self.nhc = NoseHooverChain(
            atoms,
            timestep * 1e-3,  # fs → ps
            temperature_K=temp_profile[0][1],
            nchain=nchain,
            taut=taut
        )

    def get_target_temp(self):
        """根据当前时间获取目标温度"""
        # 线性插值温度剖面
        times = [p[0] for p in self.temp_profile]
        temps = [p[1] for p in self.temp_profile]
        return np.interp(self.time, times, temps)

    def step(self):
        """单步积分，更新温度和非平衡QMMM的时间"""
        # 1. 更新NHC目标温度
        target_temp = self.get_target_temp()
        self.nhc.set_temperature(target_temp)
        
        # 2. 更新非平衡QMMM的模拟时间（fs → ps）
        if hasattr(self.atoms.calc, 'set_time'):
            self.atoms.calc.set_time(self.time * 1e-3)
        
        # 3. 执行NHC积分步
        self.nhc.step()
        
        # 4. 更新累计时间
        self.time += self.timestep

    def run(self, n_steps):
        """运行n_steps步，输出温度和能量"""
        for i in range(n_steps):
            self.step()
            # 每100步输出状态
            if i % 100 == 0:
                temp = self.atoms.get_temperature()
                energy = self.atoms.get_potential_energy()
                print(f"Step {i:5d} | Time {self.time:.2f} fs | Temp {temp:.2f} K | Energy {energy:.4f} eV")


if False:
    import numpy as np
    from ase.units import fs, ps, kB, eV, Angstrom, amu

    class TemperatureRampNHC:
        """
        纯Python实现的Nose-Hoover Chain NVT积分器
        不依赖任何ASE md子模块，仅使用基础ASE功能
        """
        def __init__(self, atoms, timestep, temp_profile, nchain=4, taut=0.5):
            # 基础参数
            self.atoms = atoms
            self.dt_fs = timestep  # 时间步长（fs）
            self.dt = timestep * fs  # 转换为ASE内部单位（秒）
            self.temp_profile = temp_profile
            self.nchain = nchain
            self.taut = taut * ps  # 弛豫时间（秒）
            self.time = 0.0  # 累计模拟时间（fs）
            
            # 原子质量（kg）：amu → kg
            self.masses = self.atoms.get_masses()[:, np.newaxis] * amu
            self.n_atoms = len(self.atoms)
            
            # NHC链初始化
            self.Q = self._init_nhc_Q()  # 链的质量参数
            self.eta = np.zeros(self.nchain)  # 链位置
            self.eta_dot = np.zeros(self.nchain)  # 链速度

        def _init_nhc_Q(self):
            """初始化NHC链的质量参数（Martyna 1992标准）"""
            T0 = self.temp_profile[0][1]
            Q = np.zeros(self.nchain)
            # 第一条链：耦合原子动能
            Q[0] = self.n_atoms * 3 * kB * T0 * (self.taut)**2
            # 后续链：耦合前一条链
            for i in range(1, self.nchain):
                Q[i] = kB * T0 * (self.taut)**2
            return Q

        def _get_target_temp(self):
            """线性插值获取当前目标温度"""
            times = [p[0] for p in self.temp_profile]
            temps = [p[1] for p in self.temp_profile]
            return np.interp(self.time, times, temps)

        def _update_nhc_chain(self, vel, T_target):
            """更新NHC链并修正原子速度"""
            # 计算原子动能（J）
            kinetic = 0.5 * np.sum(self.masses * vel**2)
            
            # 计算链的受力
            force = np.zeros(self.nchain)
            force[0] = (2 * kinetic / (3 * self.n_atoms) - kB * T_target) / self.taut
            for i in range(1, self.nchain):
                force[i] = (self.Q[i-1] * self.eta_dot[i-1]**2 - kB * T_target) / self.taut
            
            # Verlet积分更新链
            self.eta_dot += 0.5 * self.dt * force / self.Q
            self.eta += self.dt * self.eta_dot
            
            # 热浴耦合：修正原子速度
            vel *= np.exp(-0.5 * self.dt * self.eta_dot[0])
            return vel

        def step(self):
            """单步积分（速度Verlet + NHC控温）"""
            T_target = self._get_target_temp()
            
            # 1. 更新非平衡QMMM的模拟时间（供V(t)使用）
            if hasattr(self.atoms.calc, 'set_time'):
                self.atoms.calc.set_time(self.time / 1000.0)  # fs → ps
            
            # 2. 速度Verlet第一步：速度半更新 + 位置全更新
            # 坐标转换：Å → m
            pos = self.atoms.get_positions() * Angstrom
            # 速度转换：Å/fs → m/s
            vel = self.atoms.get_velocities() * Angstrom / fs
            # 受力转换：eV/Å → N
            forces = self.atoms.get_forces() * eV / Angstrom
            
            # 速度半更新
            vel += 0.5 * self.dt * forces / self.masses
            # NHC热浴耦合
            vel = self._update_nhc_chain(vel, T_target)
            # 位置全更新
            pos += self.dt * vel
            
            # 3. 更新原子坐标（m → Å）
            self.atoms.set_positions(pos / Angstrom)
            # 更新速度（m/s → Å/fs）
            self.atoms.set_velocities(vel * fs / Angstrom)
            
            # 4. 速度Verlet第二步：速度半更新
            forces = self.atoms.get_forces() * eV / Angstrom
            vel = self.atoms.get_velocities() * Angstrom / fs
            vel += 0.5 * self.dt * forces / self.masses
            # 再次NHC热浴耦合
            vel = self._update_nhc_chain(vel, T_target)
            self.atoms.set_velocities(vel * fs / Angstrom)
            
            # 5. 更新累计时间
            self.time += self.dt_fs

        def run(self, n_steps):
            """运行n_steps步，输出关键状态"""
            # 初始化原子速度（玻尔兹曼分布）
            self.atoms.set_velocities(
                self.atoms.get_velocities_from_temp(self._get_target_temp())
            )
            
            # 打印表头
            print("="*65)
            print(f"{'步数':^10} | {'累计时间(fs)':^15} | {'当前温度(K)':^15} | {'势能(eV)':^15}")
            print("-"*65)
            
            # 主积分循环
            for i in range(n_steps):
                self.step()
                
                # 每100步输出一次
                if i % 100 == 0:
                    current_temp = self.atoms.get_temperature()
                    current_energy = self.atoms.get_potential_energy()
                    print(f"{i:^10d} | {self.time:^15.2f} | {current_temp:^15.2f} | {current_energy:^15.4f}")
            
            print("="*65)
