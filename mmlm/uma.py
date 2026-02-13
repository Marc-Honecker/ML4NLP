import ase
from fairchem.core.datasets import AseDBDataset
from fairchem.core import pretrained_mlip, FAIRChemCalculator

import numpy as np

from ase.md.verlet import VelocityVerlet
from ase import units


def main():
    dataset_path = "../data/Omol/val"
    dataset = AseDBDataset({"src": dataset_path})
    idx = 2_000_000

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    calculator = FAIRChemCalculator(predictor, task_name="omol")

    atoms = dataset.get_atoms(idx)
    print(atoms.get_total_energy())
    print(atoms.get_potential_energy())

    atoms.calc = calculator

    dyn = VelocityVerlet(
        atoms,
        timestep=0.1 * units.fs
    )

    dyn.attach(log_properties, interval=1, atoms=atoms)
    dyn.run(steps=1_000)


def log_properties(atoms: ase.Atoms):
    com = atoms.get_center_of_mass()
    rg = radius_of_gyration(atoms)
    epot = atoms.get_potential_energy()

    print(f"COM: {com}, Rg: {rg:.4f} Å, Epot: {epot:.4f} eV")


def radius_of_gyration(atoms):
    masses = atoms.get_masses()
    positions = atoms.get_positions()
    com = atoms.get_center_of_mass()

    diff = positions - com
    squared_dist = np.sum(diff ** 2, axis=1)

    rg2 = np.sum(masses * squared_dist) / np.sum(masses)
    return np.sqrt(rg2)


if __name__ == "__main__":
    main()
