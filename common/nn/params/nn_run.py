from __future__ import annotations

import os
import yaml
import torch
import hashlib

from typing import List, Optional
from dataclasses import dataclass, field, replace

from ..params.nn_params import NNParams
from ..params.nn_train_params import NNTrainParams
from ..params.nn_model_params import NNModelParams
from ..params.nn_iteration_data_point import NNIterationDataPoint

@dataclass(frozen=True, kw_only=True, slots=True)
class NNRun:
    net_params  : NNParams
    train_params: NNTrainParams
    model_params: NNModelParams 
    
    _id         : Optional[str]                         = field(repr=False, default=None)
    _rep        : Optional[dict]                        = field(repr=False, default=None)
    idps        : Optional[List[NNIterationDataPoint]]  = field(repr=False, default=None)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def rep(self) -> str:
        return self._rep
    
    def __post_init__(self):
        rep = dict(
            sorted(
                {
                    **self.net_params.to_dict()
                    , **self.train_params.to_dict()
                    , **self.model_params.to_dict()
                }.items()
            )
        )
        
        id = hashlib.md5(
            str(rep).encode('utf-8')
        ).hexdigest()
        
        object.__setattr__(self, '_id', id)
        object.__setattr__(self, '_rep', rep)
    
    def with_idps(self, value: List[NNIterationDataPoint]):
        return replace(self, idps=value)
        
    def save(self) -> None:
        yaml_path = f"./runs/{self.id}/run.yaml"
        
        dir_path = os.path.dirname(yaml_path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.rep, f)