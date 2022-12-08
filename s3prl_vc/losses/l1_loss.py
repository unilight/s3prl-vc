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
    def __init__(self, stats):
        super(L1Loss, self).__init__()
        self.objective = torch.nn.L1Loss(reduction="mean")

    def forward(self, y, _y, y_lens, _y_lens, device):
        # match the upstream feature length to acoustic feature length to calculate the loss
        if y.shape[1] > _y.shape[1]:
            y = y[:, :_y.shape[1]]
            masks = make_non_pad_mask(_y_lens).unsqueeze(-1).to(device)
        if y.shape[1] <= _y.shape[1]:
            _y = _y[:, :y.shape[1]]
            masks = make_non_pad_mask(y_lens).unsqueeze(-1).to(device)
        
        # calculate masked loss
        y_masked = y.masked_select(masks)
        _y_masked = _y.masked_select(masks)
        loss = self.objective(y_masked, _y_masked)
        return loss