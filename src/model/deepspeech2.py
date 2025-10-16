import torch
from torch import Tensor, nn
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=True,
            batch_first=False,
        )

    def forward(self, x, lengths, hidden_state=None):
        x = x.permute(1, 2, 0)
        x = self.batch_norm(x)
        x = x.permute(2, 0, 1)
        packed_x = pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        packed_out, hidden_state = self.gru(packed_x, hidden_state)
        x, _ = pad_packed_sequence(packed_out)
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)
        return x, hidden_state


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(41, 11),
            stride=(2, 2),
            padding=(20, 5),
        )
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.activation1 = nn.Hardtanh(0, 20)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(21, 11),
            stride=(2, 1),
            padding=(10, 5),
        )
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.activation2 = nn.Hardtanh(0, 20)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=96,
            kernel_size=(21, 11),
            stride=(2, 1),
            padding=(10, 5),
        )
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.activation3 = nn.Hardtanh(0, 20)

    def forward(self, spectrogram: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        spectrogram = self.conv1(spectrogram)
        lengths = self.update_lengths(lengths, self.conv1)
        spectrogram = self.batch_norm1(spectrogram)
        spectrogram = self.activation1(spectrogram)
        spectrogram = self.apply_mask(spectrogram, lengths)

        spectrogram = self.conv2(spectrogram)
        lengths = self.update_lengths(lengths, self.conv2)
        spectrogram = self.batch_norm2(spectrogram)
        spectrogram = self.activation2(spectrogram)
        spectrogram = self.apply_mask(spectrogram, lengths)

        spectrogram = self.conv3(spectrogram)
        lengths = self.update_lengths(lengths, self.conv3)
        spectrogram = self.batch_norm3(spectrogram)
        spectrogram = self.activation3(spectrogram)
        spectrogram = self.apply_mask(spectrogram, lengths)
        return spectrogram, lengths

    def update_lengths(self, lengths: Tensor, conv: nn.Conv2d) -> Tensor:
        kernel_size = conv.kernel_size[1]
        stride = conv.stride[1]
        padding = conv.padding[1]
        dilation = conv.dilation[1]
        return (lengths + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def apply_mask(self, x: Tensor, lengths: Tensor) -> Tensor:
        batch_size, _, _, max_len = x.size()
        mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) >= lengths.cuda().unsqueeze(1)
        mask = mask[:, None, None, :]
        return x.masked_fill(mask, 0)


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, rnn_layers, hidden_size, dropout, n_tokens):
        super().__init__()

        self.conv_module = Conv()
        rnn_input_size = self.calculate_rnn_input_size(n_feats)
        deep_speech_block_1 = Block(input_size=rnn_input_size, hidden_size=hidden_size, dropout=dropout)
        deep_speech_block_2 = Block(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
        self.deep_speech_blocks = nn.ModuleList([deep_speech_block_1, *[deep_speech_block_2 for _ in range(rnn_layers - 1)]])
        self.head = nn.Linear(in_features=hidden_size, out_features=n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        spectrogram = spectrogram.unsqueeze(1)
        spectrogram, length = self.conv_module(spectrogram, spectrogram_length)

        batch_size, num_channels, n_feats, spec_len = spectrogram.shape
        spectrogram = spectrogram.reshape(batch_size, num_channels * n_feats, spec_len)
        spectrogram = spectrogram.permute(2, 0, 1).contiguous()

        for block in self.deep_speech_blocks:
            spectrogram, _ = block(spectrogram, length)

        spec_len, batch_size, hidden_size = spectrogram.shape
        spectrogram = self.head(spectrogram.view(spec_len * batch_size, hidden_size))
        output = (spectrogram.view(spec_len, batch_size, -1).permute(1, 0, 2).contiguous())
        return {"log_probs": nn.functional.log_softmax(output, dim=2), "log_probs_length": length}

    def calculate_rnn_input_size(self, n_feats):
        for conv in [self.conv_module.conv1, self.conv_module.conv2, self.conv_module.conv3]:
            n_feats = (n_feats + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
        rnn_input_size = n_feats * self.conv_module.conv3.out_channels
        return rnn_input_size
