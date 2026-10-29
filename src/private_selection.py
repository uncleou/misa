"""Feature-wise private readout conditioned on detached shared context."""
import math
import torch
from torch import nn


class PrivateSelection(nn.Module):
    modalities = ('t', 'a', 'v')

    def __init__(self, dim, hidden_dim=64, init_gate=0.9, mode='shared_context'):
        super().__init__()
        if mode not in ('shared_context', 'private_only'):
            raise ValueError('Unknown private selection mode')
        self.mode = mode
        self.context = (nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU())
                        if mode == 'shared_context' else None)
        self.norms = nn.ModuleDict({m: nn.LayerNorm(dim) for m in self.modalities})
        width = 2 * dim if self.context is not None else dim
        self.gates = nn.ModuleDict({m: nn.Sequential(
            nn.Linear(width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
            for m in self.modalities})
        for gate in self.gates.values():
            # Near-pass-through, but nonzero weights allow context gradients on step 1.
            nn.init.normal_(gate[-1].weight, mean=0., std=1e-3)
            nn.init.constant_(gate[-1].bias, math.log(init_gate / (1 - init_gate)))

    def forward(self, shared, private):
        context = None
        if self.context is not None:
            context = self.context(torch.cat([shared[m] for m in self.modalities], -1).detach())
        selected, gates = {}, {}
        for m in self.modalities:
            inp = self.norms[m](private[m])
            if context is not None:
                inp = torch.cat((inp, context), -1)
            gates[m] = torch.sigmoid(self.gates[m](inp))
            selected[m] = gates[m] * private[m]
        return selected, gates
