import argparse
import yaml
import torch

from mmlm.models.pos_readout_model import PositionReadoutModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model_from_config(cfg: dict) -> PositionReadoutModel:
    from mmlm.models.continuous_model import ContinuousConfig  # Beispiel

    model_cfg = ContinuousConfig(**cfg["model"])
    model = PositionReadoutModel(model_cfg)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model


def encode_positions_to_tokens(positions: torch.Tensor):
    """
    positions: Tensor [n_atoms, 3] oder [B, n_atoms, 3]
    Rückgabe: Dict kompatibel zu `input_ids` im Model:
              {"tokens": LongTensor[B, L], ...evtl. weitere Keys}
    """
    # TODO: an Ihren Tokenizer / Ihr Datenset anpassen.
    # Hier nur ein Dummy, der `positions` NICHT wirklich kodiert.
    # Sie müssen hier die gleiche Kodierung verwenden,
    # mit der Sie das Modell trainiert haben.
    raise NotImplementedError("encode_positions_to_tokens() an Projekt anpassen")


def simple_velocity_verlet(positions, velocities, forces, masses, dt):
    """
    Ein einfacher Velocity-Verlet-Integrator als Demo.
    positions, velocities, forces: [B, n_atoms, 3]
    masses: [B, n_atoms, 1] oder [n_atoms, 1]
    """
    acc = forces / masses
    new_positions = positions + velocities * dt + 0.5 * acc * dt * dt
    new_forces = None  # wird nach neuem Forward-Pass gesetzt
    new_velocities = velocities + 0.5 * (acc + acc) * dt  # hier Placeholder
    return new_positions, new_velocities, new_forces


def run_md(
    model: PositionReadoutModel,
    init_positions: torch.Tensor,
    n_steps: int = 10,
    dt: float = 0.5,
):
    model.eval().to(DEVICE)

    # Batching
    positions = init_positions.unsqueeze(0).to(DEVICE)  # [1, n_atoms, 3]
    n_atoms = positions.shape[1]
    velocities = torch.zeros_like(positions)  # Start mit v=0
    masses = torch.ones(1, n_atoms, 1, device=DEVICE)  # Dummy: alle 1 u

    energies = []
    traj = [positions.detach().cpu().clone()]

    with torch.no_grad():
        for step in range(n_steps):
            # --- 1) Positionen → Tokens kodieren ---
            input_ids = encode_positions_to_tokens(positions)  # Dict
            # Achtung: Model erwartet `input_ids["tokens"]` als LongTensor

            # --- 2) Forward-Pass ---
            out = model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                labels=None,
                return_dict=True,
                num_logits_to_keep=positions.shape[1],
            )

            # `process_continuous_output` gibt (energy_pred, padded_forces) zurück
            energy_pred, padded_forces = out.logits
            # energy_pred: [B, 1] oder [B, ...] je nach Modell
            energies.append(energy_pred.detach().cpu())

            # Für Kräfte hängt es davon ab, ob `regress_forces=True` war.
            # Falls Kräfte NICHT trainiert wurden, ist `padded_forces` nur Dummy.
            # → In diesem Fall können Sie keine MD machen, nur Energiescans.
            # Falls Ihr Modell Kräfte liefert, hier entsprechend extrahieren.
            # Beispiel (anzupassen!):
            # forces: [B, n_atoms, 3]
            # TODO: Kräfte korrekt aus `out.logits` oder `force_pred` ziehen.
            raise NotImplementedError("Kraft-Auslese aus dem Modell anpassen")

            # --- 3) Integrationsschritt ---
            positions, velocities, forces = simple_velocity_verlet(
                positions, velocities, forces, masses, dt
            )

            traj.append(positions.detach().cpu().clone())

    traj = torch.stack(traj, dim=1)  # [B, T, n_atoms, 3]
    energies = torch.stack(energies, dim=1)  # [B, T, ...]
    return traj, energies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="5M_final.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Pfad zum vortrainierten Checkpoint, z.B. `checkpoints/step_100000.pt`")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model_from_config(cfg)
    model = load_checkpoint(model, args.ckpt)

    # TODO: Startkonfiguration laden (z.B. aus XYZ, PDB, o.ä.)
    # Beispiel-Dummy:
    n_atoms = 5
    init_positions = torch.zeros(n_atoms, 3)  # [n_atoms, 3]

    traj, energies = run_md(
        model=model,
        init_positions=init_positions,
        n_steps=args.steps,
        dt=args.dt,
    )

    # Ergebnisse speichern
    torch.save(
        {"traj": traj, "energies": energies},
        "md_result.pt",
    )


if __name__ == "__main__":
    main()