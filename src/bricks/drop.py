from jittor.nn import Dropout
import jittor.nn as nn
import jittor as jt
from ..utils_van.registry import DROPOUT_LAYERS, build_from_cfg

class DropPath(nn.Module):
    '''Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    '''
    def __init__(self, p=0.5, is_train=False):
        '''
            :param p: Specifies the probability of each batch retention. Defaults to 0.5.
            :type p: float dtype
            :param is_train: Specify whether it is a training model. Defaults to False.
            :type is_train: bool
        '''
        self.p = p
        self.is_train = is_train
        #TODO: test model.train() to change self.is_train
    def execute(self, x):
        if self.p == 0. or not self.is_train:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
        output = x.divide(keep_prob) * random_tensor.floor()
        return output


def droppath(x,p=0.5,is_train=False):
    return DropPath(p=p,is_train=is_train)(x)

DROPOUT_LAYERS.register_module(name='Dropout', module=Dropout)
DROPOUT_LAYERS.register_module(name='DropPath', module=DropPath)


def build_dropout(cfg, **default_args):
    """Builder for drop out layers."""
    return build_from_cfg(cfg, DROPOUT_LAYERS, **default_args)
