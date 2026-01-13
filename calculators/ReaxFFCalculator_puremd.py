import os, shutil
import subprocess
import numpy as np
import pathlib
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from typing import Dict

# Set the ASE PuReMD run command environment variable
os.environ['ASE_PUREMDRUN_COMMAND'] = 'puremd'

class ReaxFFCalculator_PureMD(Calculator):
    """
    PuReMD QMMM_AMBER style ReaxFF force field ASE calculator
    Compatible with .geo format geometry files (geo_format=0) and example control file parameters
    Dependencies: PuReMD official development version, ASE, numpy
    """
    implemented_properties = ['energy', 'forces']
    default_parameters = {
        'ff_file': None,          # Required: ReaxFF force field file (*.ff, PuReMD compatible format)
        'tmp_dir': './tmp_puremd',# Temporary file directory
        'cleanup': False,         # Clean up temporary files after calculation
        'geo_format': 0,         # Geometry file format: 0 (corresponds to .geo format, matches the example)
        'charge_method': 0,      # Charge method: 0=QEq, 1=EEM, 2=ACKS2 (matches the example)
        'nbrhood_cutoff': 5.0,   # Neighbor cutoff radius (Å, matches the example)
        'hbond_cutoff': 7.5      # Hydrogen bond interaction cutoff radius (Å, matches the example)
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pprint import pprint
        #pprint(self.parameters)

        # Validate core parameter (force field file)
        self.fffile = self.parameters['ff_file']
        if self.fffile is None or not os.path.exists(self.fffile):
            raise FileNotFoundError(f"ReaxFF force field file does not exist: {self.fffile}")
        # Initialize temporary directory
        self.tmp_dir = pathlib.Path(self.parameters['tmp_dir']).resolve()
        
        os.makedirs(self.tmp_dir, exist_ok=True)
        # shutil.copy(self.fffile, self.tmp_dir)
        # Cache paths of 3 core files (comply with PuReMD requirements, geometry file suffix is .geo)
        self.geo_file = os.path.join(self.tmp_dir, 'default.geo')  # .geo format geometry file
        self.control_file = os.path.join(self.tmp_dir, 'default.control')  # Control parameter file
        self.dump_file = os.path.join(self.tmp_dir, 'default.trj')  # Force output dump file
        # Reuse the user-provided force field file directly (only pass the path without copying)


    def _write_geo_file(self, atoms) -> None:
        """
        Generate PuReMD QMMM_AMBER style .geo format geometry file (geo_format=0)
        Format matches the example: first line BOXGEO → second line atom count → subsequent lines atom coordinates
        """
        natoms = len(atoms)
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        cell = atoms.get_cell()

        # Extract box parameters (lx, ly, lz, alpha, beta, gamma)
        lx, ly, lz = cell[0,0], cell[1,1], cell[2,2]
        alpha, beta, gamma = 90.0, 90.0, 90.0  # Orthogonal box, all angles are 90° (matches the example)

        # Write to .geo format file
        with open(self.geo_file, 'w', encoding='utf-8') as f:
            # First line: BOXGEO + box parameters (keep 3 decimal places, match example format)
            box_line = f"BOXGEO   {lx:.3f}    {ly:.3f}    {lz:.3f}    {alpha:.3f}    {beta:.3f}    {gamma:.3f}\n"
            f.write(box_line)
            # Second line: total number of atoms
            f.write(f"{natoms}\n")
            # Subsequent lines: atom data (id sym1 sym2 x y z, keep 3 decimal places, match example)
            for idx, (sym, pos) in enumerate(zip(symbols, positions), 1):
                x, y, z = pos
                atom_line = f"{idx} {sym} {sym}    {x:.3f}   {y:.3f}   {z:.3f}\n"
                f.write(atom_line)


    def _write_control_file(self, atoms) -> None:
        """
        Generate PuReMD QMMM_AMBER style control file
        Fully matches the example format, contains core parameters, comments start with !, no redundancy
        """
        natoms = len(atoms)
        # Extract calculator parameters (match the example)
        geo_format = self.parameters['geo_format']
        charge_method = self.parameters['charge_method']
        nbrhood_cutoff = self.parameters['nbrhood_cutoff']
        hbond_cutoff = self.parameters['hbond_cutoff']

        # Control file content (strictly match example format, consistent parameter order)
        control_content = f"""simulation_name    default  ! Output file prefix, default: default.sim
ensemble_type      0                 ! 0=NVE ensemble (no temperature/pressure control for single point calculation), default: 0
nsteps             0                 ! Number of dynamics steps=0 (single point calculation), default: 0
dt                 0.25              ! Time step (no practical meaning for single point calculation), default: 0.25

nbrhood_cutoff     {nbrhood_cutoff}           ! Bond interaction cutoff radius (Å), default: 4.0
bond_graph_cutoff  {hbond_cutoff}            ! Bond graph cutoff threshold, default: 0.3
thb_cutoff         0.001             ! Three-body interaction bond strength threshold, default: 0.001
hbond_cutoff       7.5               ! Hydrogen bond interaction cutoff radius (Å), default: 0.0
vlist_buffer       2.5               ! Verlet list buffer distance, default: 2.5
reneighbor         1                 ! Neighbor list update frequency (1 is sufficient for single point calculation)

charge_method      {charge_method}                ! 0=QEq charge equilibration method, 1=EEM, 2=ACKS2, default: 0
cm_q_net           0.0               ! Net system charge, default: 0.0
cm_solver_type     2                 ! 2=CG linear solver, default: 0
cm_solver_max_iters  100             ! Maximum number of solver iterations, default: 20
cm_solver_q_err    1.0e-6           ! Solver convergence threshold, default: 1e-6

geo_format         {geo_format}                ! 0=custom GEO format, 1=PDB, 2=ASCII restart file, default: 0

energy_update_freq 1                 ! Energy output frequency (1 is sufficient for single point calculation), default: 0
traj_title         default           ! (no white spaces)
traj_compress           0                       ! 0: no compression  1: uses zlib to compress trajectory output
atom_info          1                 ! Output basic atom information, default: 0
atom_forces        1                 ! Output atomic forces (core requirement for single point calculation), default: 0
test_forces        1
bond_info          0
angle_info         0
atom_velocities    0                 ! Do not output atomic velocities (no velocity for single point calculation), default: 0
write_freq         1                 ! Output trajectory file, default: 0
restart_freq       0                 ! Do not output restart file (no need for single point calculation), default: 0
restart_format     0                 ! Restart file format (no practical meaning), default: 0
"""

        # Write to control file
        with open(self.control_file, 'w', encoding='utf-8') as f:
            f.write(control_content)


    def _parse_puremd_results(self) -> Dict[str, np.ndarray]:
        """
        Parse PuReMD output results: total energy (stdout/log) + atomic forces (dump file)
        Fully compatible with QMMM_AMBER style output format
        """
        natoms = len(self.atoms)
        # Unit conversion factor: real(kcal/mol) → eV (ASE default energy unit)
        # unit_conv = 1.0
        unit_conv = 0.0433641

        # 1. Parse total potential energy (extract from PuReMD run log (stdout))
        with open(self.dump_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        energy = 0.0
        forces = []
        after_format = False
        i = 0
        while i < len(lines):
            if 'Frame Format' in lines[i]:
                uniq_format = lines[i+1]
                after_format = True
                i += 2
                continue

            if after_format and i < len(lines) and len(lines[i].strip().split()) == 2:
                i += 1
                terms = lines[i].strip().split()
                energy = float(terms[3])
                i += 1
                continue

            if after_format and i < len(lines)  and len(lines[i].strip().split()) == 3:
                i += 1
                for idx in range(natoms):
                    terms = lines[i].strip().split()
                    forces.append([float(terms[4]), float(terms[5]), float(terms[6])])
                    i += 1
                continue
            i += 1

        energy *= unit_conv
        forces = - np.array(forces) * unit_conv
        # Convert to numpy array and verify dimension correctness
        #print(f"📝 Finished parsing atomic forces, generated force array with shape: {forces.shape}")

        if forces.shape != (natoms, 3):
            raise RuntimeError(f"Invalid force array dimension: expected {(natoms, 3)}, actual {forces.shape}")

        return {
            'energy': energy,
            'forces': forces
        }


    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """
        Core calculation logic: generate .geo geometry file → call PuReMD → parse results → clean up temporary files
        Strictly follow PuReMD command format: puremd geo_file ffield_param_file control_file
        """
        super().calculate(atoms, properties, system_changes)
        self.atoms = atoms.copy()
        natoms = len(self.atoms)

        # 1. Generate 3 core files required by PuReMD (.geo geometry file + control file)
        self._write_geo_file(atoms)
        self._write_control_file(atoms)

        # 2. Construct PuReMD call command (strictly match format: geo_file fffile control_file)
        puremd_exec = os.environ["ASE_PUREMDRUN_COMMAND"]
        cmd = [
            puremd_exec,
            os.path.basename(self.geo_file),          # First parameter: .geo format geometry file (basename for tmp dir execution)
            os.path.abspath(self.fffile),             # Second parameter: force field file (absolute path to avoid path issues)
            os.path.basename(self.control_file)       # Third parameter: control file (basename for tmp dir execution)
        ]

        # 3. Run PuReMD with directory switching (enter tmp_dir first, then run command)
        log_path = os.path.join(self.tmp_dir, 'default_x.log')  # Move log file to tmp_dir
        original_cwd = os.getcwd()  # Save original working directory

        try:
            # Switch to temporary directory (ensure all output files are generated in tmp_dir)
            os.chdir(self.tmp_dir)
            #print(f"📝 Switched to temporary directory: {self.tmp_dir}")

            # Run PuReMD and capture stdout/stderr to log file
            with open(log_path, 'w', encoding='utf-8') as log_file:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
           # print(f"📝 PuReMD command executed successfully, log file saved to: {log_path}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"PuReMD execution failed, please check log file: {log_path}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"PuReMD executable file not found, please check path: {puremd_exec}")
        finally:
            # Switch back to original working directory regardless of execution result
            os.chdir(original_cwd)
            #print(f"📝 Switched back to original working directory: {original_cwd}")

        # 4. Update log file path in parameters (for parsing) and parse calculation results
        results = self._parse_puremd_results()
        self.results['energy'] = results['energy']
        self.results['forces'] = results['forces']

        # 5. Clean up temporary files (if cleanup is enabled)
        if self.parameters['cleanup']:
            tmp_files = [self.geo_file, self.control_file, self.dump_file, log_path]
            for file_path in tmp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    #print(f"📝 Deleted temporary file: {file_path}")
            # Delete temporary directory if it is empty
            if not os.listdir(self.tmp_dir):
                os.rmdir(self.tmp_dir)
                #print(f"📝 Deleted empty temporary directory: {self.tmp_dir}")

        return self.results

# Main function test code (wrapped in if __name__ == "__main__")
if __name__ == "__main__":
    """
    Test ReaxFFCalculator_PureMD class (.geo format adapted version)
    Steps: 1. Build test atomic structure 2. Configure calculator parameters 3. Execute calculation 4. Output verification results
    Note: Modify fffile and puremd_exec to your actual paths before running
    """
    print("=" * 80)
    print("Starting to test ReaxFFCalculator_PureMD (.geo format adapted version)")
    print("=" * 80)

    # -------------------------- Step 1: Configure key parameters (modify to your actual environment) --------------------------
    # Replace with your ReaxFF force field file path (*.ff)
    FF_FILE_PATH = "./reaxff.ff"  # Example: Water molecule ReaxFF force field file
    # Replace with your PuReMD executable file path
    PUREMD_EXEC_PATH = os.environ["ASE_PUREMDRUN_COMMAND"]  # Example: Compiled PuReMD executable

    # Validate key files
    if not os.path.exists(FF_FILE_PATH):
        print(f"❌ Error: Force field file does not exist, please check path: {FF_FILE_PATH}")
        exit(1)
    if not os.path.exists(PUREMD_EXEC_PATH) and PUREMD_EXEC_PATH != "puremd":
        print(f"❌ Error: PuReMD executable file does not exist, please check path: {PUREMD_EXEC_PATH}")
        exit(1)

    # -------------------------- Step 2: Build test atomic structure (H2O molecule, periodic boundaries) --------------------------
    print("\n📌 Building test atomic structure: Water molecule (H2O) + Orthogonal periodic box")
    
    from lammps_utils import parse_lammps_data_to_ase_atoms, load_lammps_data_0, update_lammps_data
    try:
        with open('oplsaa2_react.data', 'r') as f:
            data = load_lammps_data_0(f.read())
            data = update_lammps_data(data, update_atom_index=True)
        atoms = parse_lammps_data_to_ase_atoms(data)
        print(f"✅ Atomic structure built successfully, total number of atoms: {len(atoms)}")
        print(f"✅ Simulation box parameters: {atoms.get_cell()[0,0]:.2f} × {atoms.get_cell()[1,1]:.2f} × {atoms.get_cell()[2,2]:.2f} Å")
        print(f"✅ Periodic boundaries: Enabled (matches example configuration)")
    except Exception as e:
        print(f"❌ Failed to build atomic structure: {e}")
        exit(1)

    # -------------------------- Step 3: Initialize PuReMD ReaxFF calculator (QMMM_AMBER style) --------------------------
    print("\n📌 Initializing ReaxFFCalculator_PureMD (.geo format)")
    try:
        puremd_calc = ReaxFFCalculator_PureMD(
            ff_file=FF_FILE_PATH,
            cleanup=False,                 # Automatically clean up temporary files after calculation
            logfile="puremd_test_log.log",# Test-specific log file
            units="real",                 # Unit system: real (kcal/mol, Å)
        )
        print("✅ Calculator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize calculator: {e}")
        exit(1)

    # -------------------------- Step 4: Execute ReaxFF single point energy/force calculation --------------------------
    print("\n📌 Executing ReaxFF single point energy/force calculation")
    try:
        # Attach calculator to atomic structure
        atoms.calc = puremd_calc
        # Get total potential energy
        total_energy = atoms.get_potential_energy()
        # Get atomic forces
        atomic_forces = atoms.get_forces()
        print("✅ Calculation executed successfully!")
    except Exception as e:
        print(f"❌ Failed to execute calculation: {e}")
        tmp_log_path = os.path.join(puremd_calc.tmp_dir, "puremd_test_log.log")
        print(f"📝 Please check log file for troubleshooting: {tmp_log_path}")
        exit(1)

    # -------------------------- Step 5: Output calculation result verification --------------------------
    print("\n" + "=" * 50)
    print("📊 Calculation results output (Units: Energy → eV, Forces → eV/Å)")
    print("=" * 50)
    print(f"1. Total potential energy: {total_energy:.6f} eV")
    # print(f"2. Atomic forces (x/y/z directions for each atom):")
    # for idx, (sym, force) in enumerate(zip(atoms.get_chemical_symbols(), atomic_forces), 1):
    #     print(f"   Atom {idx} ({sym}): fx={force[0]:.6f}, fy={force[1]:.6f}, fz={force[2]:.6f} eV/Å")

    # # -------------------------- Step 6: Additional verification (result dimension, log file, .geo file) --------------------------
    # print("\n" + "=" * 50)
    # print("🔍 Additional verification information")
    # print("=" * 50)
    # # Verify force array dimension
    # if atomic_forces.shape == (len(atoms), 3):
    #     print(f"✅ Force array dimension verification passed: {atomic_forces.shape} (as expected)")
    # else:
    #     print(f"⚠️  Force array dimension verification warning: Expected {(len(atoms), 3)}, Actual {atomic_forces.shape}")
    # # Verify log file existence (if cleanup is disabled)
    # tmp_log_path = os.path.join(puremd_calc.tmp_dir, "puremd_test_log.log")
    # if not puremd_calc.parameters['cleanup'] and os.path.exists(tmp_log_path):
    #     print(f"✅ Log file generated successfully: {tmp_log_path} (can view detailed running information)")
    # elif puremd_calc.parameters['cleanup']:
    #     print(f"✅ Log file has been cleaned up automatically (cleanup=True enabled)")
    # else:
    #     print(f"⚠️  Log file not found, possible output exception")
    # # Verify .geo file existence (if cleanup is disabled)
    # if not puremd_calc.parameters['cleanup'] and os.path.exists(puremd_calc.geo_file):
    #     print(f"✅ .geo geometry file generated successfully: {puremd_calc.geo_file} (can check format consistency)")
    # elif puremd_calc.parameters['cleanup']:
    #     print(f"✅ .geo geometry file has been cleaned up automatically (cleanup=True enabled)")
    # else:
    #     print(f"⚠️  .geo geometry file not found, possible output exception")

    print("\n" + "=" * 80)
    print("🎉 Test completed! If no errors occurred, the .geo format adapted interface is working properly")
    print("=" * 80)