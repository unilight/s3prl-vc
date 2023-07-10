# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Diffusion-based acoustic model implementation for voice conversion
References:
    - https://github.com/MoonInTheRiver/DiffSinger
    - https://github.com/nnsvs/nnsvs
"""

from typing import Sequence
from collections import OrderedDict

import torch
import torch.nn as nn
import logging

from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from s3prl_vc.models.diffsinger import GaussianDiffusion, DiffNet


class Diffusion(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,  # input dimension of conditioning features
        out_dim: int,  #
        denoiser_residual_channels: int,
        use_spemb=False,
        resample_ratio=1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.denoiser_residual_channels = denoiser_residual_channels
        self.resample_ratio = resample_ratio

        # ppgel -> melspec
        self.mel_model = GaussianDiffusion(
            in_dim=in_dim,
            out_dim=80,
            denoise_fn=DiffNet(
                encoder_hidden_dim=in_dim,
                residual_channels=denoiser_residual_channels,
                use_spk_emb=use_spemb,
            ),
            # TODO: an encoder can also be specified here
            # encoder=Conv1dResnet(
            #    in_dim=in_dim,
            #    hidden_dim=256,
            #    num_layers=2,
            #    out_dim=256
            # ),
        )
        """Initialize Diffusion Module.

        Args:
            in_dim (int): Dimension of the inputs.
            out_dim (int): Dimension of the outputs.
            denoiser_residual_channels (int): Dimension of diffusion model hidden units.
            use_spemb (bool): Whether or not to use speaker embeddings.
            resample_ratio (float): Ratio to align the input and output features.
        """

    def forward(
        self,
        x,
        lengths,
        y_mel,
        spk=None,
    ):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of padded input conditioning features (B, Lmax, in_dim).
            lengths (LongTensor): Batch of lengths of each input batch (B,).
            y_mel (Tensor): Batch of padded target features (B, Lmax, out_dim).
            spk (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Ground truth noise.
            Tensor: Predicted noise.
            LongTensor: Resampled lengths based on upstream feature.

        """

        # resample the input features according to resample_ratio
        x = x.permute(0, 2, 1)
        resampled_features = F.interpolate(x, scale_factor=self.resample_ratio)
        x = resampled_features.permute(0, 2, 1)
        lengths = lengths * self.resample_ratio

        # cut if necessary
        if x.size(1) > y_mel.size(1):
            x = x[:, : y_mel.size(1), :]
        elif x.size(1) < y_mel.size(1):
            y_mel = y_mel[:, : x.size(1), :]

        if spk is not None:
            spk = spk.squeeze(-1)

        mel_inp = x
        mel_ = self.mel_model(mel_inp, lengths, y_mel, spk)
        return (mel_[0], mel_[1], lengths)

    def inference(self, x, spk=None):
        """Calculate during inference.

        Args:
            x (Tensor): Batch of padded input conditioning features (B, Lmax, in_dim).
            spk (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Predicted mel-spectrogram.

        """

        # resample the input features according to resample_ratio
        x = x.permute(0, 2, 1)
        resampled_features = F.interpolate(x, scale_factor=self.resample_ratio)
        x = resampled_features.permute(0, 2, 1)

        if spk is not None:
            spk = spk.transpose(1, 0)

        mel_ = self.mel_model.inference(x, spk_emb=spk)
        return mel_
