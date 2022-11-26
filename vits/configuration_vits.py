from typing import List, Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class VITSConfig(PretrainedConfig):
    model_type = "vits"

    def __init__(
        self,
        vocab_size: int = 392,
        phoneme_embedding_dim: int = 192,
        speaker_id_embedding_dim: int = 256,
        n_speakers: int = 1,
        speaker_to_id: Optional[Dict[str, int]] = None,
        z_channels: int = 192,
        # text encoder
        n_heads: int = 2,
        n_layers: int = 6,
        text_encoder_kernel_size: int = 3,
        text_encoder_p_dropout: float = 0.1,
        # posterior encoder
        spec_channels: int = 513,
        pe_wavenet_kernel_size: int = 5,
        pe_wavenet_dilation_rate: int = 1,
        pe_wavenet_n_resblocks: int = 16,
        # decoder
        upsample_initial_channel: int = 512,
        deconv_strides: List[int] = [8, 8, 2, 2],
        deconv_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        ## istft decoder
        decoder_type: str = "mb_istft",
        upsample_rates: List[int] = [4, 4],
        upsample_kernel_sizes: List[int] = [16, 16],
        subbands: int = 4,
        gen_istft_n_fft: int = 16,
        gen_istft_hop_size: int = 4,
        # flow
        flow_n_flows: int = 4,
        flow_wavenet_kernel_size: int = 5,
        flow_wavenet_dilation_rate: int = 1,
        flow_wavenet_n_resblocks: int = 4,
        # stochastic duration predictor
        filter_channels: int = 192,
        sdp_kernel_size: int = 3,
        sdp_p_dropout: float = 0.5,
        sdp_n_flows: int = 4,
        # generator
        segment_size: int = 32,
        # etc
        **kwargs,
    ):
        decoder_type = decoder_type.lower()
        if decoder_type not in [
            "ms_istft",
            "mb_istft",
            "istft",
            "original",
        ]:
            raise ValueError(
                "decoder_type must be one of "
                "['ms_istft', 'mb_istft', 'istft', 'original'], "
                f"but got {decoder_type}"
            )

        if speaker_to_id is not None and n_speakers != len(speaker_to_id):
            raise ValueError(
                "n_speakers must be equal to the length of speaker_to_id, "
                f"but got n_speakers={n_speakers} and "
                f"len(speaker_to_id)={len(speaker_to_id)}"
            )
        elif speaker_to_id is None:
            speaker_to_id = {str(i): i for i in range(n_speakers)}

        self.vocab_size = vocab_size
        self.phoneme_embedding_dim = phoneme_embedding_dim
        self.speaker_id_embedding_dim = speaker_id_embedding_dim
        self.n_speakers = n_speakers
        self.speaker_to_id = speaker_to_id
        self.z_channels = z_channels
        self.spec_channels = spec_channels
        # text encoder
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.text_encoder_kernel_size = text_encoder_kernel_size
        self.text_encoder_p_dropout = text_encoder_p_dropout
        # posterior encoder
        self.pe_wavenet_kernel_size = pe_wavenet_kernel_size
        self.pe_wavenet_dilation_rate = pe_wavenet_dilation_rate
        self.pe_wavenet_n_resblocks = pe_wavenet_n_resblocks
        # decoder
        self.upsample_initial_channel = upsample_initial_channel
        self.deconv_strides = deconv_strides
        self.deconv_kernel_sizes = deconv_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        ## istft decoder
        self.decoder_type = decoder_type
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.subbands = subbands
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        # flow
        self.flow_n_flows = flow_n_flows
        self.flow_wavenet_kernel_size = flow_wavenet_kernel_size
        self.flow_wavenet_dilation_rate = flow_wavenet_dilation_rate
        self.flow_wavenet_n_resblocks = flow_wavenet_n_resblocks
        # stochastic duration predictor
        self.filter_channels = filter_channels
        self.sdp_kernel_size = sdp_kernel_size
        self.sdp_p_dropout = sdp_p_dropout
        self.sdp_n_flows = sdp_n_flows
        # generator
        self.segment_size = segment_size

        super().__init__(**kwargs)
