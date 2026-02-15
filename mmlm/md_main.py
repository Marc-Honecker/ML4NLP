import os
from pathlib import Path
from datetime import datetime
import json

import torch
import hydra
from omegaconf import DictConfig

from mmlm.datasets_v2 import TextDataset
from mmlm.md.AtomStore import AtomStore
from mmlm.md.Potential import GraphFreeMLIP
from mmlm.md.measurements import CenterOfMass, PotentialEnergyRegister
from mmlm.md.propagator import LeapFrogVerletPropagator
from mmlm.md.util import initialize_tokenizer
from mmlm.models.pos_readout_model import PositionReadoutModel
from mmlm.train import get_model
from mmlm.utils.utils import get_start_end_indices_by_token_type, get_collator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_out_metadata(outdir: str, args: DictConfig, batch: dict):
    if not os.path.exists(os.path.dirname(outdir)):
        os.makedirs(os.path.dirname(outdir))

    metadata_path = Path(os.path.join(outdir, "metadata.json"))
    metadata = {
        "checkpoint": args.training.checkpoint,
        "molecule_idx": args.dataset.molecule_idx,
        "n_steps": args.md.n_steps,
        "dt": args.md.dt,
        "log_interval": args.md.log_interval,
        "reference_energy": batch['labels']['target_labels'].item(),
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=4)


def run_md(args: DictConfig, batch: dict, model: PositionReadoutModel):
    atom_store = AtomStore(batch)
    propagator = LeapFrogVerletPropagator()
    potential = GraphFreeMLIP(model=model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"md_outputs/{timestamp}/"
    write_out_metadata(output_dir, args, batch)

    com_measurer = CenterOfMass(dirname=output_dir)
    potential_energy_register = PotentialEnergyRegister(dirname=output_dir)

    n_steps = args.md.n_steps
    dt = args.md.dt
    log_interval = args.md.log_interval

    running_energy = 0.0

    for step in range(n_steps):
        energy, forces = potential.compute_total_potential_energy_and_forces(atom_store)
        atom_store.f = forces

        propagator.propagate(atom_store=atom_store, dt=dt)

        if step % log_interval == 0:
            print(f"Step {step}/{n_steps}, Potential Energy: {energy.item():.4f}")
            com_measurer.compute(atom_store, time=step * dt)
            potential_energy_register.record(energy.item(), time=step * dt)

        running_energy += energy.item()

    print(running_energy / n_steps)

    com_measurer.write_measurements_to_file()
    potential_energy_register.write_measurements_to_file()


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    tokenizer, n_actual_bins = initialize_tokenizer(args)
    start_end_indices_by_token_type = get_start_end_indices_by_token_type(args, tokenizer, n_actual_bins)

    model = get_model(args, tokenizer, start_end_indices_by_token_type).to(DEVICE)
    model.eval()

    assert isinstance(model, PositionReadoutModel), "Model is not of type PositionReadoutModel"

    val_dataset = hydra.utils.instantiate(args.dataset.val_dataset)
    assert isinstance(val_dataset, TextDataset), "Dataset is not of type TextDataset"
    sample = val_dataset[args.dataset.molecule_idx]

    collator = get_collator(args, tokenizer)
    batch = collator([sample])

    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else
                 {kk: vv.to(DEVICE) for kk, vv in v.items()})
             for k, v in batch.items()}

    run_md(args, batch, model)


if __name__ == "__main__":
    main()
