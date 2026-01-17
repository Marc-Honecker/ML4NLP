import torch
import hydra
from omegaconf import DictConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from mmlm.custom_tokenizer import get_tokenizer
from mmlm.datasets_v2 import TextDataset
from mmlm.md.AtomStore import AtomStore
from mmlm.md.Potential import GraphFreeMLIP
from mmlm.md.propagator import LeapFrogVerletPropagator
from mmlm.models.pos_readout_model import PositionReadoutModel
from mmlm.train import get_model
from mmlm.utils.utils import get_n_actual_bins, get_start_end_indices_by_token_type, get_collator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_md(batch: dict, model: PositionReadoutModel, n_steps: int = 100, dt: float = 0.5):
    atom_store = AtomStore(batch)
    propagator = LeapFrogVerletPropagator()
    potential = GraphFreeMLIP(model=model)

    running_energy = 0.0

    for step in range(n_steps):
        energy, forces = potential.compute_total_potential_energy_and_forces(atom_store)
        atom_store.f = forces

        propagator.propagate(atom_store=atom_store, dt=dt)

        if step % 100 == 0:
            print(
                f"Step {step}, Energy: {energy.item():.4f}, Total Force Magnitude: {atom_store.f.norm().item():.4f}")
            print(
                f"R[0].norm(): {atom_store.x[0, :].norm().item()}, V[0].norm(): {atom_store.v[0, :].norm().item()}, F[0].norm(): {atom_store.f[0, :].norm().item()}")
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

    run_md(batch, model, n_steps=1_000, dt=0.1)


if __name__ == "__main__":
    main()
