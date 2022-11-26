from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import weight_norm
from transformers.utils import ModelOutput

from .stft import TorchSTFT

if TYPE_CHECKING:
    from .configuration_vits import VITSConfig


@dataclass
class VITSDecoderOutput(ModelOutput):
    """
    Output type of `Decoder`

    Args:
        wav_fake (`torch.FloatTensor` of shape `(B, 1, T)`):
            The predicted waveform.
        wav_fake_mb (*optional* `torch.FloatTensor` of shape
                     `(B, subbands, T // subbands)`):
            The predicted waveform with multi-band.
            only for MultibandIstftDecoder, MultistreamIstftDecoder.
    """

    wav_fake: torch.FloatTensor = None
    wav_fake_mb: Optional[torch.FloatTensor] = None


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return ((kernel_size - 1) * dilation) // 2


class ResNetBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3, 5)
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):

            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)

            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class Decoder(nn.Module):
    def __init__(self, config: "VITSConfig"):
        super().__init__()

        self.config = config
        self.speaker_id_embedding_dim = config.speaker_id_embedding_dim
        self.in_z_channel = config.z_channels
        self.upsample_initial_channel = config.upsample_initial_channel
        self.deconv_strides = config.deconv_strides
        self.deconv_kernel_sizes = config.deconv_kernel_sizes
        self.resblock_kernel_sizes = config.resblock_kernel_sizes
        self.resblock_dilation_sizes = config.resblock_dilation_sizes

        self.num_deconvs = len(self.deconv_strides)
        # generate self.num_resnet_blocks resnetblocks per Deconv1d layer
        self.num_resnet_blocks = len(self.resblock_kernel_sizes)

        # network to apply first to input z
        self.conv1d_pre = nn.Conv1d(
            self.in_z_channel,
            self.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        # network to apply first to input embedded speaker id
        self.cond = nn.Conv1d(
            self.speaker_id_embedding_dim, self.upsample_initial_channel, 1
        )

        # create each Deconv1d layer
        self.ups = nn.ModuleList()
        for i, (stride, kernel) in enumerate(
            zip(self.deconv_strides, self.deconv_kernel_sizes)
        ):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        self.upsample_initial_channel // (2**i),
                        self.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=kernel,
                        stride=stride,
                        padding=(kernel - stride) // 2,
                    )
                )
            )

        # generate self.num_resnet_blocks ResnetBlocks per Deconv1d layer
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resnet_blocks_channels = self.upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(
                self.resblock_kernel_sizes, self.resblock_dilation_sizes
            ):
                self.resblocks.append(
                    ResNetBlock(
                        channels=resnet_blocks_channels,
                        kernel_size=kernel,
                        dilation=dilation,
                    )
                )

        # resnet_blocks_channels = 32
        self.conv1d_post = nn.Conv1d(
            resnet_blocks_channels, 1, 7, 1, padding=3, bias=False
        )
        self.ups.apply(init_weights)

    def forward(
        self, z: Tensor, speaker_id_embedded: Tensor, return_dict: Optional[bool] = None
    ):
        """
        Args:
            z (FloatTensor): (batch_size, config.z_channels, T)
            speaker_id_embedded (FloatTensor):
                (batch_size, config.speaker_id_embedding_dim, 1)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # align the number of channels of z, speaker_id_embedded by conv1d
        x = self.conv1d_pre(z) + self.cond(speaker_id_embedded)
        # apply each Deconv1d layer
        for i in range(self.num_deconvs):
            x = F.leaky_relu(x, 0.1)
            # apply each Deconv1d layer
            x = self.ups[i](x)
            # Apply ResnetBlock by self.num_resnet_blocks,
            # let xs be (sum of outputs/self.num_resnet_blocks)
            xs = None
            for j in range(self.num_resnet_blocks):
                if xs is None:
                    xs = self.resblocks[i * self.num_resnet_blocks + j](x)
                else:
                    xs += self.resblocks[i * self.num_resnet_blocks + j](x)
            x = xs / self.num_resnet_blocks
        x = F.leaky_relu(x)
        # Output audio has 1 channel, [B, 1, T]
        x = self.conv1d_post(x)
        wav_fake = torch.tanh(x)

        if not return_dict:
            return wav_fake, None
        return VITSDecoderOutput(wav_fake=wav_fake)


class IstftDecoder(nn.Module):
    def __init__(self, config: "VITSConfig"):
        super().__init__()
        self.config = config
        self.speaker_id_embedding_dim = config.speaker_id_embedding_dim
        self.in_z_channel = config.z_channels
        self.upsample_rates = config.upsample_rates
        self.upsample_kernel_sizes = config.upsample_kernel_sizes
        self.upsample_initial_channel = config.upsample_initial_channel
        self.resblock_kernel_sizes = config.resblock_kernel_sizes
        self.resblock_dilation_sizes = config.resblock_dilation_sizes

        self.gen_istft_n_fft = config.gen_istft_n_fft
        self.gen_istft_hop_size = config.gen_istft_hop_size
        self.post_n_fft = self.gen_istft_n_fft

        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)
        self.conv1d_pre = weight_norm(
            nn.Conv1d(
                self.in_z_channel,
                self.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )

        self.cond = nn.Conv1d(
            self.speaker_id_embedding_dim, self.upsample_initial_channel, 1
        )

        self.ups = nn.ModuleList()
        for i, (stride, kernel) in enumerate(
            zip(self.upsample_rates, self.upsample_kernel_sizes)
        ):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        self.upsample_initial_channel // (2**i),
                        self.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=kernel,
                        stride=stride,
                        padding=(kernel - stride) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resnet_blocks_channels = self.upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(
                self.resblock_kernel_sizes, self.resblock_dilation_sizes
            ):
                self.resblocks.append(
                    ResNetBlock(
                        channels=resnet_blocks_channels,
                        kernel_size=kernel,
                        dilation=dilation,
                    )
                )

        self.conv1d_post = weight_norm(
            nn.Conv1d(resnet_blocks_channels, self.post_n_fft + 2, 7, 1, padding=3)
        )
        self.ups.apply(init_weights)
        self.conv1d_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft,
        )

    def forward(
        self, z: Tensor, speaker_id_embedded: Tensor, return_dict: Optional[bool] = None
    ):
        """
        Args:
            z (FloatTensor): (batch_size, config.z_channels, T)
            speaker_id_embedded (FloatTensor):
                (batch_size, config.speaker_id_embedding_dim, 1)
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        x = self.conv1d_pre(z) + self.cond(speaker_id_embedded)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv1d_post(x)

        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.pi * torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        wav_fake = self.stft.inverse(spec, phase)

        if not return_dict:
            return wav_fake, None
        return VITSDecoderOutput(wav_fake=wav_fake)
