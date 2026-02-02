import torch
import os

from mmlm.md.AtomStore import AtomStore


class CenterOfMass:
    def __init__(self, dirname: str):
        self._com_positions = []
        self._radius_of_gyration = []
        self._timesteps = []
        self._dirname = dirname

        if not os.path.exists(os.path.dirname(dirname)):
            os.makedirs(os.path.dirname(dirname))

    def compute(self, atom_store: AtomStore, time: float):
        """
        Compute the center of mass and radius of gyration for a set of positions and masses.

        Parameters:
        positions (torch.Tensor): Tensor of shape (N, 3) representing the positions of N atoms.
        masses (torch.Tensor): Tensor of shape (N,) representing the masses of N atoms.
        """
        positions = atom_store.x
        masses = atom_store.masses

        total_mass = torch.sum(masses)
        com = torch.sum(positions * masses, dim=0) / total_mass
        r2_g = torch.mean((positions - com) ** 2)

        self._com_positions.append(com)
        self._radius_of_gyration.append(r2_g.sqrt().item())
        self._timesteps.append(time)

    def write_measurements_to_file(self):
        """
        Write the computed center of mass positions and radius of gyration to a file.
        """
        com_string = "# Center of Mass Positions (time, x, y, z)\n"
        rog_string = "# Radius of Gyration (time, rog)\n"
        for timestep, com, rog in zip(self._timesteps, self._com_positions, self._radius_of_gyration):
            com_string += f"{timestep:.4f},{com[0].item():.6f},{com[1].item():.6f},{com[2].item():.6f}\n"
            rog_string += f"{timestep:.4f},{rog:.6f}\n"

        with open(self._dirname + 'com.npy', "w") as file:
            file.write(com_string)

        with open(self._dirname + "rog.npy", "w") as file:
            file.write(rog_string)
