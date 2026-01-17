from abc import ABC, abstractmethod

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from mmlm.md.AtomStore import AtomStore
from mmlm.models.pos_readout_model import PositionReadoutModel


class Potential(ABC):
    @abstractmethod
    def compute_total_potential_energy_and_forces(self, atom_store: AtomStore) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the total potential energy and forces acting on the atoms.

        Args:
            atom_store (AtomStore): The AtomStore containing atom positions and other relevant data.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - total_potential_energy (torch.Tensor): The total potential energy of the system.
                - forces (torch.Tensor): The forces acting on each atom.
        """
        pass

    def compute_total_potential_energy(self, atom_store: AtomStore) -> torch.Tensor:
        """
        Compute the total potential energy of the system.
        """
        total_potential_energy, _ = self.compute_total_potential_energy_and_forces(atom_store)
        return total_potential_energy

    def compute_forces(self, atom_store: AtomStore) -> torch.Tensor:
        """
        Compute the forces acting on the atoms.
        """
        _, forces = self.compute_total_potential_energy_and_forces(atom_store)
        return forces


class GraphFreeMLIP(Potential):
    def __init__(self, model: PositionReadoutModel):
        self.model = model

    def compute_total_potential_energy_and_forces(self, atom_store: AtomStore) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=atom_store.batch_data['input_ids'],
                labels=atom_store.batch_data['labels'],
            )

        force_pred = outputs.logits[1][: atom_store.batch_info.num_atoms]  # (1, n_atoms, 3)
        energy_pred = outputs.logits[0]  # (1, 1)

        return energy_pred, force_pred.squeeze(0)  # (n_atoms, 3)
