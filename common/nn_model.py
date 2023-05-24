import torch
import numpy as np
import torch_geometric as pyg

from tqdm import tqdm
from torch import nn, optim
from sklearn import metrics
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
from dataclasses import dataclass, field, replace

@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    optim               : str
    n_epochs            : int
    weight_decay        : float
    learning_rate       : float
    train_loader        : DataLoader
    snapshot_interval   : Optional[int]                 = None
    val_loader          : Optional[DataLoader]          = None
    snapshot_x          : Optional[Tuple[np.ndarray]]   = None
    
    def __str__(self):
        return f"Train=[epochs={self.n_epochs}, optim={self.optim}, lr={self.learning_rate:1.0e}, weight_decay={self.weight_decay:1.0e}]"
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNEvaluationDataPoint:
    f1          : float
    recall      : float
    accuracy    : float
    precision   : float
    loss        : Optional[float] = None
    error       : Optional[float] = None
    
    def with_loss(self, value: float):
        return replace(self, loss=value)
    
    def with_error(self, value: float):
        return replace(self, error=value)
    
    @staticmethod
    def of(Y: np.ndarray, Y_hat: np.ndarray):
        return NNEvaluationDataPoint(
            accuracy=metrics.accuracy_score(y_true=Y, y_pred=Y_hat)
            , f1=metrics.f1_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
            , recall=metrics.recall_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
            , precision=metrics.precision_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
        )
    
    @staticmethod
    def mean_of(edps: List['NNEvaluationDataPoint']):
        ret = NNEvaluationDataPoint(
            f1=np.mean([edp.f1 for edp in edps])
            , recall=np.mean([edp.recall for edp in edps])
            , accuracy=np.mean([edp.accuracy for edp in edps])
            , precision=np.mean([edp.precision for edp in edps])
        )

        if len([edp.loss for edp in edps if edp.loss is not None]) > 0:
            ret = ret.with_loss(np.mean([edp.loss for edp in edps if edp.loss is not None]))
            
        if len([edp.error for edp in edps if edp.error is not None]) > 0:
            ret = ret.with_error(np.mean([edp.error for edp in edps if edp.error is not None]))

        return ret
        
@dataclass(frozen=True, kw_only=True, slots=True)
class NNSnapshotDataPoint:
    y_hat   : np.ndarray
    y_log   : np.ndarray

@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
    iter_idx    : int
    epoch_idx   : int
    batch_idx   : int  
    train_edp   : NNEvaluationDataPoint
    val_edp     : Optional[NNEvaluationDataPoint]   = None
    snapshot    : Optional[NNSnapshotDataPoint]     = None

class NNModel():
    def __init__(self, net: nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        
        self.net = net.to(self.device)
        self.loss = nn.CrossEntropyLoss().to(self.device)

    def train(self, params: NNTrainParams):
        train_str = f"{self.net} x {params}"
        validate = params.val_loader is not None
        snapshot = params.snapshot_x is not None

        if params.optim == "sgd":
            optimizer = optim.SGD(
                lr=params.learning_rate
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )
        elif params.optim == "adam":
            optimizer = optim.Adam(
                lr=params.learning_rate
                , params=self.net.parameters()
                , weight_decay=params.weight_decay
            )
        
        idx_iter: int                       = 0
        idps    : List[IterationDataPoint]  = []
        n_iter  : int                       = int(params.n_epochs * len(params.train_loader))

        tqdm_bar = tqdm(
            desc=train_str
            , total=n_iter
        )

        with torch.set_grad_enabled(True):
            for idx_epoch in range(params.n_epochs):
                for batch_idx, batch in enumerate(params.train_loader):
                    self.net.train()
                    self.net.zero_grad()
                        
                    X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                    
                    train_loss = self.loss(Y_hat_log, Y)
                    
                    train_edp = (
                        NNEvaluationDataPoint.of(Y=Y.numpy(), Y_hat=Y_hat.numpy())
                            .with_loss(value=float(train_loss))
                            .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
                    )
                        
                    val_edp = self.evaluate(loader=params.val_loader) if validate else None
                        
                    if snapshot and (
                        params.snapshot_interval is None or (
                            (params.snapshot_interval is not None)
                            and (idx_iter % params.snapshot_interval == 0)
                        )
                    ):
                        snapshot_y_log, snapshot_y_hat = self.predict(X=params.snapshot_x)
                        
                        snapshot = NNSnapshotDataPoint(y_hat=snapshot_y_hat, y_log=snapshot_y_log)
                    else:
                        snapshot = None

                    idps.append(
                        NNIterationDataPoint(
                            val_edp=val_edp
                            , snapshot=snapshot
                            , iter_idx=idx_iter
                            , epoch_idx=idx_epoch
                            , batch_idx=batch_idx
                            , train_edp=train_edp
                        )
                    )

                    train_loss.backward()
                    optimizer.step()

                    idx_iter += 1
                    tqdm_bar.update(1)
                    tqdm_bar.set_postfix_str(f"error: {val_edp.error if val_edp is not None else train_edp.error:.4f}")
        
        return train_str, np.array(idps)

    def evaluate(self, loader: DataLoader):
        self.net.eval()
        edps = []

        with torch.no_grad():
            for batch in loader:
                _, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                
                edps.append(
                    NNEvaluationDataPoint.of(Y=Y.numpy(), Y_hat=Y_hat.numpy())
                        .with_loss(value=float(self.loss(Y_hat_log, Y)))
                        .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
                )

        return NNEvaluationDataPoint.mean_of(edps)

    def __fwd_pass(self, batch):
        X, Y = self.net.unpack_batch(batch)
        
        X = tuple(x.to(self.device) for x in X)
        Y = Y.to(self.device)
        
        Y_hat_log = self.net(*X)
        Y_hat = Y_hat_log.argmax(dim=1)
        
        return X, Y, Y_hat_log, Y_hat
    
    def predict(self, X: np.ndarray):
        if not isinstance(X, tuple):
            X = (X,)
        
        X = tuple(torch.from_numpy(x).to(self.device) for x in X)
        
        self.net.eval()
        with torch.no_grad():
            Y_hat_log = self.net(*X).detach().numpy()
            Y_hat = Y_hat_log.argmax(axis=1)
            
            return Y_hat_log, Y_hat