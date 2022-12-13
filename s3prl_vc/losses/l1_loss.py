# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import logging

import numpy as np
import torch
import torch.nn.functional as F

from s3prl_vc.layers.utils import make_non_pad_mask


class L1Loss(torch.nn.Module):
    """
    L1 loss module supporting (1) loss calculation in the normalized target feature space
                              (2) masked loss calculation
    """

    def __init__(self):
        super(L1Loss, self).__init__()
        self.objective = torch.nn.L1Loss(reduction="mean")

    def forward(self, predicted, predicted_lens, target, target_lens, device):
        # match the upstream feature length to acoustic feature length to calculate the loss
        if predicted.shape[1] > target.shape[1]:
            predicted = predicted[:, : target.shape[1]]
            masks = make_non_pad_mask(target_lens).unsqueeze(-1).to(device)
        if predicted.shape[1] <= target.shape[1]:
            target = target[:, : predicted.shape[1]]
            masks = make_non_pad_mask(predicted_lens).unsqueeze(-1).to(device)

        # calculate masked loss
        predicted_masked = predicted.masked_select(masks)
        target_masked = target.masked_select(masks)
        loss = self.objective(predicted_masked, target_masked)
        return loss
