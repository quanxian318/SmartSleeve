"""
Loss functions for EMG ratio prediction.

组合损失: L = λ₁·MSE + λ₂·(1-Pearson) + λ₃·MAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PearsonLoss(nn.Module):
    """1 - Pearson correlation coefficient，用于保证预测趋势与真实值一致。"""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, 2] or [N, 2]
            target: [B, 2] or [N, 2]
        Returns:
            scalar: mean(1 - r) over 2 output channels
        """
        pc = pred - pred.mean(dim=0, keepdim=True)
        tc = target - target.mean(dim=0, keepdim=True)

        cov = (pc * tc).sum(dim=0)
        p_std = torch.sqrt((pc ** 2).sum(dim=0) + self.eps)
        t_std = torch.sqrt((tc ** 2).sum(dim=0) + self.eps)

        r = cov / (p_std * t_std + self.eps)
        return (1.0 - r).mean()


class CombinedLoss(nn.Module):
    """
    组合损失，支持逐通道记录 Pearson r 供日志使用。

    L = λ_mse * MSE + λ_pearson * (1 - r) + λ_mae * MAE
    """

    def __init__(self, lambda_mse=1.0, lambda_pearson=0.5, lambda_mae=0.3):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_pearson = lambda_pearson
        self.lambda_mae = lambda_mae
        self.pearson = PearsonLoss()

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        corr_loss = self.pearson(pred, target)
        mae = F.l1_loss(pred, target)

        total = (
            self.lambda_mse * mse
            + self.lambda_pearson * corr_loss
            + self.lambda_mae * mae
        )

        with torch.no_grad():
            pc = pred - pred.mean(dim=0, keepdim=True)
            tc = target - target.mean(dim=0, keepdim=True)
            cov = (pc * tc).sum(dim=0)
            ps = torch.sqrt((pc ** 2).sum(dim=0) + 1e-8)
            ts = torch.sqrt((tc ** 2).sum(dim=0) + 1e-8)
            r_ch = cov / (ps * ts + 1e-8)

        return total, {
            'mse': mse.item(),
            'pearson_loss': corr_loss.item(),
            'mae': mae.item(),
            'total': total.item(),
            'r_biceps': r_ch[0].item(),
            'r_triceps': r_ch[1].item(),
        }
