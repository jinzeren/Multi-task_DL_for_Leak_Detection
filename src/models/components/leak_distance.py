from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakClsDis(nn.Module):
    def __init__(self,
                 input_n: int = 256,
                 base_n: int = 32,
                 neuron_scaling_factor: List = [4, 2, 1]):
        super(LeakClsDis, self).__init__()
        self.ln_1 = nn.Linear(
            input_n, neuron_scaling_factor[0] * base_n, bias=False)
        self.ln_2 = nn.Linear(
            neuron_scaling_factor[0] * base_n, neuron_scaling_factor[1] * base_n, bias=False)
        self.bn_1 = nn.BatchNorm1d(neuron_scaling_factor[0] * base_n)
        self.bn_2 = nn.BatchNorm1d(neuron_scaling_factor[1] * base_n)
        self.ln_3_c = nn.Linear(
            neuron_scaling_factor[1] * base_n, neuron_scaling_factor[2] * base_n)
        self.ln_3_r = nn.Linear(
            neuron_scaling_factor[1] * base_n, neuron_scaling_factor[2] * base_n)
        self.out_c = nn.Linear(neuron_scaling_factor[2] * base_n, 1)
        self.out_r = nn.Linear(neuron_scaling_factor[2] * base_n, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_1(self.ln_1(x)))
        x = self.relu(self.bn_2(self.ln_2(x)))
        cls_out = self.out_c(self.relu(self.ln_3_c(x)))
        dis_out = self.out_r(self.relu(self.ln_3_r(x)))

        return cls_out, dis_out
