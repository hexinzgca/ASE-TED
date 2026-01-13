"""This module defines an ASE interface to OpenMM."""

import ase.units as units
import numpy as np
from ase.calculators.calculator import Calculator
try:
    import openmm as mm
except ImportError:
    raise ImportError("OpenMM is not installed.")

class OPENMM(Calculator):
    """
    Interface to OpenMM using an ASE calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to calculate.
    label : str
        The label of the calculator.
    context : openmm.Context
        The OpenMM context to use for the calculation.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, atoms=None, label=None, context=None):
        # if not have_openmm:
        #     raise RuntimeError("OpenMM is not installed.")
        Calculator.__init__(self, label, atoms)
        self._context = context

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            if "energy" in self.results:
                del self.results["energy"]
            if "forces" in self.results:
                del self.results["forces"]

        if "energy" not in self.results:
            # Set the positions of the atoms in the OpenMM context
            self._context.setPositions(atoms.get_positions() * mm.unit.angstrom)

            # Compute the potential energy
            energy = self._context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)

            # Store the energy in the results dictionary
            self.results["energy"] = energy * units.kJ / units.mol
        if "forces" not in self.results:
            # Set the positions of the atoms in the OpenMM context
            self._context.setPositions(atoms.get_positions() * mm.unit.angstrom)

            # Compute the forces
            forces = self._context.getState(getForces=True, getEnergy=True).getForces(asNumpy=True).value_in_unit(mm.unit.kilojoule_per_mole / mm.unit.angstrom)
            energy = self._context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)

            # Store the forces and energy in the results dictionary
            self.results["forces"] = forces * units.kJ / units.mol / units.Angstrom
            #forces * units.kJ / units.mol / units.nm
            self.results["energy"] = energy * units.kJ / units.mol


"""Module containing utility functions for OpenMM and ASE interoperability."""
def openmm_topology_to_ase_atoms(openmm_topology, positions=None):
    """
    Convert an OpenMM topology to an ASE Atoms object.

    Parameters
    ----------
    openmm_topology : openmm.Topology
        The OpenMM topology object.
    positions : np.ndarray, optional
        The positions of the atoms. If not provided, the positions are set to None.
    """
    from ase import Atoms     
    import ase.units as units
    import numpy as np
    try:
        import openmm as mm 
    except ImportError:
        raise ImportError("OpenMM is not installed.")

    # Extract information from the OpenMM topology
    symbols = [atom.element.symbol for atom in openmm_topology.atoms()]
    atomic_numbers = [atom.element.atomic_number for atom in openmm_topology.atoms()]
    masses = [
        atom.element.mass.value_in_unit(mm.unit.dalton)
        for atom in openmm_topology.atoms()
    ]
    cell = (
        openmm_topology.getUnitCellDimensions().value_in_unit(mm.unit.nanometer)
        * units.nm
    )
    pbc = cell is not None

    # Create and set the information in the ASE Atoms object
    atoms = Atoms(symbols)
    atoms.set_chemical_symbols(symbols)
    atoms.set_atomic_numbers(atomic_numbers)
    if positions is not None:
        positions = np.array(positions.value_in_unit(mm.unit.angstrom)).tolist()
        atoms.set_positions(positions)
    atoms.set_masses(masses)
    atoms.set_cell(cell)
    atoms.set_pbc(pbc)

    return atoms