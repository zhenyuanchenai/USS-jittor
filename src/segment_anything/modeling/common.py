import jittor as jt
from jittor import nn
from typing import Type

class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module]=nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def execute(self, x): 
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = jt.ones(num_channels)
        self.bias = jt.zeros(num_channels)
        self.eps = eps

    def execute(self, x):
        u = x.mean(1, keepdims=True)
        s = (x - u).pow(2).mean(1, keepdims=True)
        x = ((x - u) / jt.sqrt((s + self.eps)))
        x = ((self.weight[:, None, None] * x) + self.bias[:, None, None])
        
        return x