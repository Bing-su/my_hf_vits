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
        window = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("window", window, persistent=False)

    def transform(self, input_data: torch.Tensor):
        """
        istft VITS doesn't use transform

        Args:
            input_data (FloatTensor): (batch, samples)
        Returns:
            magnitude (FloatTensor): (batch, frequencies, frames)
            phase (FloatTensor): (batch, frequencies, frames)
        """
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
        )

        magnitude = torch.abs(forward_transform)
        phase = torch.angle(forward_transform)
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor):
        """
        Args:
            magnitude (FloatTensor): (batch, frequencies, frames)
            phase (FloatTensor): (batch, frequencies, frames)
        Returns:
            reconstructed audio (FloatTensor): (batch, 1, samples')
        """
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
        )

        # unsqueeze to stay consistent with conv_transpose1d implementation
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data: torch.Tensor):
        """
        istft VITS doesn't use forward

        So the dimensions of the input and output are different,
        but I leave it unmodified.

        Args:
            input_data (FloatTensor): (batch, samples)
        Returns:
            reconstructed audio (FloatTensor): (batch, 1, samples')
        """
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction

    def extra_repr(self) -> str:
        return "filter_length={}, hop_length={}, win_length={}".format(
            self.filter_length, self.hop_length, self.win_length
        )
