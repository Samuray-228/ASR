import torch
import torch_audiomentations
from torch import nn


class HighPassFilter(nn.Module):
    def __init__(self, prob, min_cutoff_freq, max_cutoff_freq, sample_rate):
        super().__init__()
        self.augmentation = torch_audiomentations.HighPassFilter(
            p=prob,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            sample_rate=sample_rate,
        )

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.augmentation(spectrogram.unsqueeze(1)).squeeze(1)
