from dataclasses import dataclass

import torch

atom_id_to_mass_map = {
    1: 1.008,  # Hydrogen
    2: 4.0026,  # Helium
    3: 6.94,  # Lithium
    4: 9.0122,  # Beryllium
    5: 10.81,  # Boron
    6: 12.011,  # Carbon
    7: 14.007,  # Nitrogen
    8: 15.999,  # Oxygen
    9: 18.998,  # Fluorine
    10: 20.180,  # Neon
    11: 22.990,  # Sodium
    12: 24.305,  # Magnesium
    13: 26.982,  # Aluminum
    14: 28.085,  # Silicon
    15: 30.974,  # Phosphorus
    16: 32.06,  # Sulfur
    17: 35.45,  # Chlorine
    18: 39.948,  # Argon
    19: 39.098,  # Potassium
    20: 40.078,  # Calcium
}


def compute_masses(atom_types: torch.Tensor) -> torch.Tensor:
    masses = torch.tensor([atom_id_to_mass_map.get(int(atom_id.item())) for atom_id in atom_types],
                          device=atom_types.device)

    return masses.unsqueeze(-1)


@dataclass(frozen=True, eq=True, kw_only=True)
class BatchInfo:
    start_idx: int
    end_idx: int
    num_atoms: int


def get_batch_info(batch: dict) -> BatchInfo:
    atoms = batch['input_ids']['atoms']
    non_zero_elements = torch.nonzero(atoms)

    num_atoms = len(non_zero_elements)
    start_idx = non_zero_elements[0][1].item()
    end_idx = non_zero_elements[-1][1].item()

    return BatchInfo(
        start_idx=start_idx,
        end_idx=end_idx + 1,
        num_atoms=num_atoms
    )
