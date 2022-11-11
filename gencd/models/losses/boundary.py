import numpy as np
from typing import Callable, List, Optional, Sequence, Union, Tuple
from scipy.ndimage import distance_transform_edt

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

class BoundaryLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.

            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.
        """
        super().__init__(reduction=reduction)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # compute
        target_dist = torch.from_numpy(batch_onehot2dist(target.cpu().numpy())).type(torch.float32).to(input.device)
        f = torch.einsum("bkwh,bkwh->bkwh", input, target_dist)
            
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class BoundaryBCEWithLogitsLoss(_Loss):
    def __init__(
        self,
        lambda_bd: float = 1.0,
        lambda_bce: float = 1.0,
        bce_pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight, reduction=reduction,)
        self.bd = BoundaryLoss(sigmoid=True, reduction=reduction,)
        self.lambda_bce = lambda_bce 
        self.lambda_bd = lambda_bd
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(input, target)
        bd_loss = self.bd(input, target)
        total_loss: torch.Tensor = self.lambda_bce * bce_loss + self.lambda_bd * bd_loss

        return total_loss        
        
def onehot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    """
    seg: (ch, x, y)
    """
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = distance_transform_edt(negmask, sampling=resolution) * negmask \
                - (distance_transform_edt(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

def batch_onehot2dist(x):
    return np.array(list(map(onehot2dist, x)))