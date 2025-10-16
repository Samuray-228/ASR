import torch
from torch import nn
from torchaudio import transforms


class TimeMasking(nn.Module):
    def __init__(self, param, prob):
        super().__init__()
        self.augmentation = transforms.TimeMasking(time_mask_param=param)
        self.prob = prob

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.prob:
            return self.augmentation(spectrogram)

        return spectrogram
