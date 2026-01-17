import logging
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from mmlm.custom_tokenizer import get_tokenizer
from mmlm.utils.utils import get_n_actual_bins

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


def initialize_tokenizer(args: DictConfig) -> tuple[PreTrainedTokenizer, dict[Any, Any]]:
    n_actual_bins = get_n_actual_bins(args)

    if args.training.finetune:
        assert (
            not args.dataset.per_atom_target
        ), "Per-atom target is already done in finetuning!"
        if args.dataset.get("norm_stats_path", None) is None:
            logging.warning(
                "No norm stats path provided for finetuning! Using default values (0, 1) for energy and force mean and std."
            )

    if args.dataset.joint_embedding:
        if n_actual_bins["pos"] > 15:
            logging.warning(
                f"The number of bins for position is greater than 15."
            )
        n_actual_bins["pos"] = n_actual_bins["pos"] ** 3

    if args.dataset.joint_embedding_force and not args.training.finetune:
        if n_actual_bins["force"] > 15:
            logging.warning(
                f"The number of bins for force is greater than 15. This may lead to a large number of tokens with joint embedding ({n_actual_bins['force'] ** 3})."
            )
        n_actual_bins["force"] = n_actual_bins["force"] ** 3

    if args.dataset.n_bins is not None:
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
        tokenizer.add_special_tokens(
            {
                "pad_token": "[PAD]",
                "mask_token": "[MASK]",
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
            }
        )
    else:
        raise NotImplementedError("Need to discretize numbers from now on!")

    return tokenizer, n_actual_bins
