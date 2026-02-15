import os
import json
from datetime import datetime

import ase
from fairchem.core.datasets import AseDBDataset
from fairchem.core import pretrained_mlip, FAIRChemCalculator

import numpy as np

from ase.md import MDLogger
from ase.md.verlet import VelocityVerlet
from ase import units


def log_metadata(dt: float, n_steps: int, molecule_idx: int, reference_energy: float, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata = {
        "dt": dt,
        "n_steps": n_steps,
        "molecule_idx": molecule_idx,
        "reference_energy": reference_energy,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    dataset_path = "../data/Omol/val"
    dataset = AseDBDataset({"src": dataset_path})

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"md_outputs/{timestamp}/"
    molecule_idx = 2_000_000
    dt = 0.1
    n_steps = 10_000

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    calculator = FAIRChemCalculator(predictor, task_name="omol")

    atoms = dataset.get_atoms(molecule_idx)
    reference_energy = atoms.get_potential_energy()
    atoms.calc = calculator

    log_metadata(dt, n_steps, molecule_idx, reference_energy, output_dir)

    dyn = VelocityVerlet(
        atoms,
        timestep=dt * units.fs
    )

    dyn.attach(MDLogger(dyn, atoms, output_dir + "md.log", header=True, stress=False, peratom=False), interval=10)
    dyn.attach(log_properties, interval=1, atoms=atoms, output_dir=output_dir)
    dyn.run(steps=n_steps)


def log_properties(atoms: ase.Atoms, output_dir: str):
    com = atoms.get_center_of_mass()
    rg = radius_of_gyration(atoms)
    epot = atoms.get_potential_energy()

    print(f"COM: {com}, Rg: {rg:.4f} Å, Epot: {epot:.4f} eV")

    with open(os.path.join(output_dir, "com.npy"), "a") as f:
        f.write(f"{com[0]:.6f},{com[1]:.6f},{com[2]:.6f}\n")

    with open(os.path.join(output_dir, "rg.npy"), "a") as f:
        f.write(f"{rg:.6f}\n")


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
