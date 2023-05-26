from __future__ import annotations

import torch.nn.functional as F

from enum import Enum

class NNActivationFn(Enum):
    ELU         = 'elu'
    SELU        = 'selu'
    TANH        = 'tanh'
    RELU        = 'relu'
    SOFTMAX     = 'softmax'
    SIGMOID     = 'sigmoid'
    SOFTPLUS    = 'softplus'
    LEAKY_RELU  = 'leaky_relu'
    
    _map = {
        ELU         : F.elu
        , SELU      : F.selu
        , TANH      : F.tanh
        , RELU      : F.relu
        , SOFTMAX   : F.softmax
        , SIGMOID   : F.sigmoid
        , SOFTPLUS  : F.softplus
        , LEAKY_RELU: F.leaky_relu
    }
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @staticmethod
    def to_activation_fn(value: NNActivationFn):
        return NNActivationFn._map.get(value, None)
