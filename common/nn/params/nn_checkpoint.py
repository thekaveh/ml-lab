from __future__ import annotations

import os
import torch

from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict

from ..params.nn_iteration_data_point import NNIterationDataPoint
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpoint:
    idp         : NNIterationDataPoint
    model_state : OrderedDict
    optim_state : OrderedDict
    
    def to_checkpoint_file(self, path: str) -> None:
        dir_path = os.path.dirname(path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        torch.save(self, path)
        
    def to_best_checkpoint(self) -> None:
        self.to_checkpoint_file(path="./checkpoint/best.pt")
        
    def to_first_checkpoint(self) -> None:
        self.to_checkpoint_file(path="./checkpoint/first.pt")
        
    def to_1st_quartile_checkpoint(self) -> None:
        self.to_checkpoint_file(path="./checkpoint/q1.pt")
        
    def to_2nd_quartile_checkpoint(self) -> None:
        self.to_checkpoint_file(path="./checkpoint/q2.pt")
        
    def to_3rd_quartile_checkpoint(self) -> None:
        self.to_checkpoint_file(path="./checkpoint/q3.pt")
        
    def to_last_checkpoint(self) -> None:
        self.to_checkpoint_file(path="./checkpoint/last.pt")
        
    @staticmethod
    def from_checkpoint_file(path: str) -> Optional[NNCheckpoint]:
        if not os.path.exists(path):
            return None
        
        ret = torch.load(path)
        
        if not isinstance(ret, NNCheckpoint):
            return None
        
        return ret
    
    @staticmethod
    def from_best_checkpoint() -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_checkpoint_file(path="./checkpoint/best.pt")
    
    @staticmethod
    def from_first_checkpoint() -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_checkpoint_file(path="./checkpoint/first.pt")
    
    @staticmethod
    def from_1st_quartile_checkpoint() -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_checkpoint_file(path="./checkpoint/q1.pt")
    
    @staticmethod
    def from_2n_quartile_checkpoint() -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_checkpoint_file(path="./checkpoint/q2.pt")
    
    @staticmethod
    def from_3rd_quartile_checkpoint() -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_checkpoint_file(path="./checkpoint/q3.pt")
    
    @staticmethod
    def from_last_checkpoint() -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_checkpoint_file(path="./checkpoint/last.pt")