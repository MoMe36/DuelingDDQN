import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 


class DuelingNetwork(nn.Module): 

    def __init__(self, obs, ac): 

        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(), 
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)

    def forward(self, x): 

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1,1)
        return q_val
