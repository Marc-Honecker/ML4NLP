from dataclasses import dataclass

import torch
import hydra
from omegaconf import DictConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from mmlm.custom_tokenizer import get_tokenizer
from mmlm.datasets_v2 import TextDataset
from mmlm.models.pos_readout_model import PositionReadoutModel
from mmlm.train import get_model
from mmlm.utils.utils import get_n_actual_bins, get_start_end_indices_by_token_type, get_collator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def leap_frog_verlet(positions: torch.Tensor, velocities: torch.Tensor, forces: torch.Tensor, mass: torch.Tensor,
                     dt: float):
    velocities += dt * forces / mass
    positions += dt * velocities

    return positions, velocities


def compute_masses(atom_types: torch.Tensor) -> torch.Tensor:
    atom_id_to_mass = {
        1: 1.008,  # Hydrogen
        2: 4.0026,  # Helium
        5: 10.81,  # Boron
        6: 12.011,  # Carbon
        7: 14.007,  # Nitrogen
        8: 15.999,  # Oxygen
        9: 18.998,  # Fluorine
        15: 30.974,  # Phosphorus
        16: 32.06,  # Sulfur
    }

    masses = torch.tensor([atom_id_to_mass.get(int(atom_id.item())) for atom_id in atom_types],
                          device=atom_types.device)

    return masses.unsqueeze(-1)


def compute_forces_and_energy(model: PositionReadoutModel, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        outputs: CausalLMOutputWithPast = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )

    force_pred = outputs.logits[1]  # (1, n_atoms, 3)
    energy_pred = outputs.logits[0]  # (1, 1)

    return energy_pred, force_pred.squeeze(0)  # (n_atoms, 3)


def run_md(batch: dict, model: PositionReadoutModel, n_steps: int = 100, dt: float = 0.5):
    batch_info = get_batch_info(batch)

    velocities = torch.zeros(size=(batch_info.num_atoms, 3), device=DEVICE)
    positions = batch['input_ids']['numbers'][:, batch_info.start_idx:batch_info.end_idx, :]  # (1, n_atoms, 3)
    masses = compute_masses(batch['input_ids']['atoms'][:, batch_info.start_idx:batch_info.end_idx].squeeze(0))

    running_energy = 0.0

    for step in range(n_steps):
        energy, forces = compute_forces_and_energy(model, batch)  # (n_atoms, 3)
        forces = forces[:batch_info.num_atoms]

        leap_frog_verlet(
            positions=positions.squeeze(0),
            velocities=velocities,
            forces=forces,
            mass=masses,
            dt=dt
        )

        if step % 100 == 0:
            print(
                f"Step {step}, Energy: {energy.item():.4f}, Total Force Magnitude: {forces.norm().item():.4f}")
            print(
                f"R[0].norm(): {positions[0, 0, :].norm().item()}, V[0].norm(): {velocities[0, :].norm().item()}, F[0].norm(): {forces[0, :].norm().item()}")
            print()

        running_energy += energy.item()

    print(running_energy / n_steps)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    n_actual_bins = get_n_actual_bins(args)
    tokenizer = get_tokenizer(
        n_bins=n_actual_bins,
        n_atom_types=args.dataset.n_atom_types,
        spin_min=args.dataset.spin_min,
        spin_max=args.dataset.spin_max,
        charge_min=args.dataset.charge_min,
        charge_max=args.dataset.charge_max,
        finetune=args.training.finetune,
        joint_embed_atoms=args.dataset.joint_embed_atoms,
    )

    start_end_indices_by_token_type = get_start_end_indices_by_token_type(args, tokenizer, n_actual_bins)

    model = get_model(args, tokenizer, start_end_indices_by_token_type).to(DEVICE)
    model.eval()

    assert isinstance(model, PositionReadoutModel), "Model is not of type PositionReadoutModel"

    val_dataset = hydra.utils.instantiate(args.dataset.val_dataset)
    assert isinstance(val_dataset, TextDataset), "Dataset is not of type TextDataset"
    sample = val_dataset[2_000_000]

    collator = get_collator(args, tokenizer)
    batch = collator([sample])

    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else
                 {kk: vv.to(DEVICE) for kk, vv in v.items()})
             for k, v in batch.items()}

    run_md(batch, model, n_steps=10_000, dt=0.5)


if __name__ == "__main__":
    main()
