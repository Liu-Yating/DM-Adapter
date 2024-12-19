import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], input_gate: nn.Module, task_gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.input_gate = input_gate
        self.task_gate = task_gate
        self.args = moe_args
        self.alpha = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, inputs: torch.Tensor, task_param) -> torch.Tensor:
        input_gate_logits = self.input_gate(inputs)
        task_gate_logits = self.task_gate(task_param)
        
        gate_logits = (1-self.alpha) * input_gate_logits + self.alpha *  task_gate_logits
        
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )

        # calculate aux_loss
        weights_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(inputs.dtype)
        average_weight = torch.mean(weights_softmax, dim=[0,1])

        # use top 2 to cal
        indices_top2  = F.one_hot(selected_experts, num_classes=self.args.num_experts).sum(dim=2)
        average_count = torch.mean(indices_top2.float(), dim=[0,1]).to(inputs.dtype)

        # cal aux loss, Load-Balancing Loss
        l_aux = torch.mean(average_weight * average_count) * self.args.num_experts 

        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)

        
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            idx_1, idx_2, nth_expert = torch.where(selected_experts == i)
            results[idx_1, idx_2] += weights[idx_1, idx_2, nth_expert, None] * expert(inputs[idx_1,idx_2])

        return results, l_aux.float()
