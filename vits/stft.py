import torch
from torch import nn


class TorchSTFT(nn.Module):
    def __init__(
        self, filter_length: int = 800, hop_length: int = 200, win_length: int = 800
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length, periodic=True)

    def transform(self, input_data: torch.Tensor):
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(input_data.device),
            return_complex=True,
        )

        magnitude = torch.abs(forward_transform)
        phase = torch.angle(forward_transform)
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )

        # unsqueeze to stay consistent with conv_transpose1d implementation
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data: torch.Tensor):
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction
