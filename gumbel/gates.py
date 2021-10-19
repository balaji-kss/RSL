# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F

from gumbel_module import GumbelSigmoid
from sigmoid_module import Sigmoid

class GumbelGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super(GumbelGate, self).__init__()
        # self.gs = GumbelSigmoid()
        self.gs = Sigmoid()
        self.in_channels = in_channels

        # self.gating_layers = [
        #     nn.Linear(in_channels, 2*in_channels),
        #     nn.Linear(2*in_channels, 2*in_channels),
        #     nn.Linear(2*in_channels, in_channels),
        # ]

        self.gating_layers = [
            nn.Linear(in_channels, in_channels),
        ]
        
        self.gate_network = nn.Sequential(*self.gating_layers)
        self.init_weights()

    def init_weights(self, gate_bias_init: float = 0.0) -> None:

        for i in range(len(self.gating_layers)):

            fc = self.gate_network[i]
            torch.nn.init.xavier_uniform_(fc.weight)
            fc.bias.data.fill_(gate_bias_init)

    def forward(self, gate_inp: torch.Tensor) -> torch.Tensor:
        """Gumbel gates, Eq (8)"""

        #gate_inp_abs = torch.abs(gate_inp)
        gate_inp_abs = gate_inp**2
        pi_log = self.gate_network(gate_inp_abs)

        return self.gs(pi_log, force_hard=True)
