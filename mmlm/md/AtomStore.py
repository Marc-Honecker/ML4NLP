import torch

from mmlm.md.util import get_batch_info, compute_masses


class AtomStore:
    def __init__(self, batch_data: dict[str, dict]):
        self._validate_batch_data(batch_data)

        self.batch_data = batch_data
        self.batch_info = get_batch_info(batch_data)

        self.x: torch.Tensor = self.batch_data['input_ids']['numbers'][
            :, self.batch_info.start_idx:self.batch_info.end_idx, :].squeeze(0)
        self.v: torch.Tensor = torch.zeros(size=(self.batch_info.num_atoms, 3))
        self.f: torch.Tensor = torch.zeros(size=(self.batch_info.num_atoms, 3))
        self.masses: torch.Tensor = compute_masses(
            self.batch_data['input_ids']['atoms'][:, self.batch_info.start_idx:self.batch_info.end_idx].squeeze(
                0))

    # @property
    # def x(self) -> torch.Tensor:
    #     return self.batch_data['input_ids']['numbers'][
    #         :, self.batch_info.start_idx:self.batch_info.end_idx, :].squeeze(0)
    #
    # @x.setter
    # def x(self, new_positions: torch.Tensor):
    #     self.batch_data['input_ids']['numbers'][
    #         :, self.batch_info.start_idx:self.batch_info.end_idx, :] = new_positions.unsqueeze(0)

    @staticmethod
    def _validate_batch_data(batch_data: dict):
        if not isinstance(batch_data, dict):
            raise TypeError("batch_data must be a dictionary.")

        required_keys = ['input_ids', 'labels']
        for key in required_keys:
            if key not in batch_data:
                raise ValueError(f"Missing required key '{key}' in batch data.")

            if not isinstance(batch_data[key], dict):
                raise TypeError(f"batch_data['{key}'] must be a dictionary.")

        input_ids_keys = batch_data['input_ids'].keys()
        required_input_ids_keys = ['tokens', 'numbers', 'atoms']
        for key in required_input_ids_keys:
            if key not in input_ids_keys:
                raise ValueError(f"Missing required key '{key}' in batch_data['input_ids'].")

        labels_keys = batch_data['labels'].keys()
        required_labels_keys = ['labels', 'num_labels', 'target_labels', 'force_labels']
        for key in required_labels_keys:
            if key not in labels_keys:
                raise ValueError(f"Missing required key '{key}' in batch_data['labels'].")
