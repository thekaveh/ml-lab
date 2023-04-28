import torch
import torch_geometric as pyg
import torch.nn.functional as F

from torch import nn
from typing import List, Union
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True, slots=True)
class GraphSageNNParams:
    input_dim   : int
    output_dim  : int
    dropout_prob: float
    hidden_dims : Union[List[int], None] = None

class GraphSageNN(nn.Module):
    def __init__(self, params: GraphSageNNParams):
        super(GraphSageNN, self).__init__()
        
        self.params = params
        
        self.dims = [params.input_dim]
        self.dims += params.hidden_dims if params.hidden_dims is not None else []
        self.dims += [params.output_dim]
        
        self.layers = nn.ModuleList(
            [
                pyg.nn.SAGEConv(
                    in_channels=self.dims[idx]
                    , out_channels=self.dims[idx+1]
                )
                    for idx in range(len(self.dims) - 1)
            ]
        )
        
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(p=params.dropout_prob)
                    for _ in range(len(params.hidden_dims) if params.hidden_dims is not None else 0)
            ]
        )

    def forward(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # X = X.view(X.size(0), -1)
        
        for layer_idx, layer in enumerate(self.layers[:-1]):
            X = layer(X, E)
            X = F.relu(X)
            
            if layer_idx < len(self.dropouts):
                X = self.dropouts[layer_idx](X)
                
        X = self.layers[-1](X, E)
        
        return X
    
    def unpack_batch(self, batch):
        X, E, Y = batch.x, batch.edge_index, batch.y
        
        return (X, E), Y
    
    def __str__(self):
        return f"GraphSageNN{{dims={self.dims}, dropout={self.params.dropout_prob}}}"