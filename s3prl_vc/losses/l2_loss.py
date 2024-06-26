# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import torch
import torch.nn.functional as F

from s3prl_vc.layers.utils import make_non_pad_mask


class L2Loss(torch.nn.Module):
    """
    L2 loss module supporting (1) loss calculation in the normalized target feature space
                              (2) masked loss calculation
    """

    def __init__(self):
        super(L2Loss, self).__init__()
        self.objective = torch.nn.MSELoss(reduction="mean")

    def forward(self, predicted, predicted_lens, target, target_lens, device):
        # match the upstream feature length to acoustic feature length to calculate the loss

        # NOTE:
        # diffusion model needs the inputs and outputs to be
        # of the same length, so there are cases where it is
        # cut to match with the upsampled upstream features
        # thus, target_lens is not used

        if predicted.shape[1] > target.shape[1]:
            predicted = predicted[:, : target.shape[1]]
            masks = make_non_pad_mask(predicted_lens).unsqueeze(-1).to(device)
        if predicted.shape[1] <= target.shape[1]:
            target = target[:, : predicted.shape[1]]
            masks = make_non_pad_mask(predicted_lens).unsqueeze(-1).to(device)

        # calculate masked loss
        predicted_masked = predicted.masked_select(masks)
        target_masked = target.masked_select(masks)
        loss = self.objective(predicted_masked, target_masked)
        return loss
