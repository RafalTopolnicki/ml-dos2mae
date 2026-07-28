import torch.nn as nn

import torch


def pair_attention_loss(A, preferred_matrix, weight=1.0, normalize=True):
    """
    Encourage attention mass on preferred channel pairs.

    Parameters
    ----------
    A : torch.Tensor
        Learned attention, shape [B, C, C]
    preferred_matrix : torch.Tensor
        Preference weights, shape [C, C]
        0 means not preferred, non-zero means preferred with strength.
    weight : float
        Overall scaling of this regularization term.
    normalize : bool
        If True, divide by number (or sum) of preferred entries so scale is stable.

    Returns
    -------
    torch.Tensor scalar loss
    """
    # Ensure tensor on same device/dtype
    P = preferred_matrix.to(device=A.device, dtype=A.dtype)  # [C, C]

    mask = (P != 0).to(dtype=A.dtype)  # [C, C]
    if mask.sum() == 0:
        return A.new_tensor(0.0)

    # Broadcast P to [B, C, C]
    P_b = P.unsqueeze(0)  # [1, C, C]
    mask_b = mask.unsqueeze(0)

    # Reward putting attention mass where preferred (minimize negative reward)
    reward = (A * P_b * mask_b).sum()

    if normalize:
        # normalize by total preference weight (or count) to keep scale stable
        denom = (P.abs() * mask).sum()
        denom = denom if denom > 0 else mask.sum()
        reward = reward / denom

    loss = -reward
    return weight * loss

class MSEWithPairAttentionLoss(nn.Module):
    def __init__(self, pair_weight=0.0, preferred_matrix=None):
        super().__init__()
        self.pair_weight = pair_weight
        if preferred_matrix is not None:
            self.preferred_matrix = torch.tensor(preferred_matrix, dtype=torch.float32)
        else:
            self.preferred_matrix = None
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, A=None):
        loss = self.mse(y_pred, y_true)

        if A is not None and self.preferred_matrix is not None and self.pair_weight > 0:
            pair_loss = pair_attention_loss(
                A,
                self.preferred_matrix,
                weight=self.pair_weight
            )
            loss = loss + pair_loss

        return loss