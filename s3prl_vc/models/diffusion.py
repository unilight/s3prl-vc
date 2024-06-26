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
        input_dim,
        output_dim,
        resample_ratio,
        stats,
        # model params below
        use_spemb=False,
        denoiser_residual_channels=256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.denoiser_residual_channels = denoiser_residual_channels
        self.resample_ratio = resample_ratio
        self.stats = stats

        # ppgel -> melspec
        self.mel_model = GaussianDiffusion(
            in_dim=input_dim,
            out_dim=output_dim,
            denoise_fn=DiffNet(
                encoder_hidden_dim=input_dim,
                residual_channels=denoiser_residual_channels,
                use_spk_emb=use_spemb,
            ),
        )
        """Initialize Diffusion Module.

        Args:
            input_dim (int): Dimension of the inputs.
            output_dim (int): Dimension of the outputs.
            denoiser_residual_channels (int): Dimension of diffusion model hidden units.
            use_spemb (bool): Whether or not to use speaker embeddings.
            resample_ratio (float): Ratio to align the input and output features.
        """

    def forward(
        self,
        x,
        lengths,
        targets=None,
        spk_embs=None,
        f0s=None,  # not used, but just to unify with taco2
    ):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of padded input conditioning features (B, Lmax, input_dim).
            lengths (LongTensor): Batch of lengths of each input batch (B,).
            targets (Tensor): Batch of padded target features (B, Lmax, output_dim).
            spk (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).

        Returns (training):
            Tensor: Ground truth noise.
            Tensor: Predicted noise.
            LongTensor: Resampled lengths based on upstream feature.
        Returns (inference):
            Tensor: Predicted mel spectrogram.
        """

        # resample the input features according to resample_ratio
        x = x.permute(0, 2, 1)
        resampled_features = F.interpolate(x, scale_factor=self.resample_ratio)
        x = resampled_features.permute(0, 2, 1)
        lengths = lengths * self.resample_ratio

        if spk_embs is not None and type(spk_embs) != list:
            spk_embs = spk_embs.squeeze(-1)

        if targets is not None:
            # normalize
            targets = (targets - self.stats["mean"]) / self.stats["scale"]

            # cut if necessary
            if x.size(1) > targets.size(1):
                x = x[:, : targets.size(1), :]
            elif x.size(1) < targets.size(1):
                targets = targets[:, : x.size(1), :]

            # training
            mel_ = self.mel_model(x, lengths, targets, spk_embs)
            return mel_[0], mel_[1], lengths

        elif targets is None:
            # inference
            mel_ = self.mel_model.inference(x, spk_emb=spk_embs)

            # normalize
            mel_ = self.stats["mean"] + (mel_ * self.stats["scale"])
            return mel_.squeeze(0), None, lengths
