"""
model_bpu.py — BPU 原生 4D 版 AnchorCalibTCN
=============================================
关键改动: Conv1d → Conv2d(kernel=(1, k)), 全网络保持 4D tensor 流动。
权重可从原版 Conv1d 直接迁移 (unsqueeze(2)), 无需重新训练。

输入: merged_input [B, 26, 1, 64]  (C=26, H=1, W=64)
       通道0-9:   motion特征 (10通道)
       通道10-25: calib向量 (16通道, 时间轴广播)
输出: [B, 2]  biceps_ratio, triceps_ratio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CausalConv2d — 因果卷积 (只在时间维度W上做因果填充)
# ============================================================
class CausalConv2d(nn.Module):
    """2D causal convolution: 只在时间维度(W)左填充, H=1 不作卷积."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # padding 只在 W 维左侧: (kernel_size-1) * dilation
        self.padding_w = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size),       # H=1, W=kernel_size
            padding=(0, 0),                      # 手动 pad
            dilation=(1, dilation),              # H 不膨胀, W 膨胀
        )

    def forward(self, x):
        # x: [B, C, 1, T]
        # F.pad 4D 格式: (padW_left, padW_right, padH_top, padH_bottom)
        return self.conv(F.pad(x, (self.padding_w, 0, 0, 0)))


# ============================================================
# TemporalBlock2D — 残差时序卷积块 (2D版)
# ============================================================
class TemporalBlock2D(nn.Module):
    """Residual block: CausalConv2d -> ReLU -> Dropout -> CausalConv2d -> + Residual."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, (1, 1))  # 1x1 Conv2d 做通道对齐
            if in_channels != out_channels
            else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.conv2(out)
        out = self.relu(out + self.downsample(x))
        return self.dropout(out)


# ============================================================
# TCNEncoder2D — 时序卷积编码器 (4D原生)
# ============================================================
class TCNEncoder2D(nn.Module):
    """
    堆叠多层 TemporalBlock2D, dilation 指数增长 (1, 2, 4, 8, ...)。
    全流程在 4D tensor [B, C, 1, T] 上进行, 无需 transpose/squeeze。
    """

    def __init__(self, input_dim, hidden_channels, kernel_size=5, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(hidden_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock2D(in_ch, ch, kernel_size, dilation, dropout))
            in_ch = ch
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_channels[-1]

    def forward(self, x):
        """x: [B, C, 1, T] -> [B, C_out, 1, T]"""
        return self.network(x)


# ============================================================
# CalibEncoder — 校准向量编码器 (MLP, 不变)
# ============================================================
class CalibEncoder(nn.Module):
    """MLP 编码个人校准向量 -> 个体化嵌入."""

    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, calib):
        """calib: [B, calib_dim] -> [B, output_dim]"""
        return self.network(calib)


# ============================================================
# AnchorCalibTCN_BPU — BPU 原生 4D 完整模型
# ============================================================
class AnchorCalibTCN_BPU(nn.Module):
    """
    BPU 原生 4D 版 AnchorCalibTCN。
    全程保持 4D tensor，Conv2d 替代 Conv1d，无需 squeeze/transpose。

    输入:
        merged [B, 26, 1, 64]
          - 通道 0-9:  motion 特征 (10通道)
          - 通道 10-25: calib 向量 (16通道, 空间广播到 1x64)

    输出:
        [B, 2]  [biceps_ratio, triceps_ratio]
    """

    def __init__(
        self,
        motion_dim=10,
        calib_dim=16,
        window_size=64,
        tcn_channels=(128, 256, 256, 256),
        tcn_kernel=5,
        tcn_dropout=0.2,
        calib_hidden=(96, 192, 96),
        calib_dropout=0.1,
        fusion_hidden=(384, 192),
        fusion_dropout=0.1,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.calib_dim = calib_dim
        self.window_size = window_size

        # ── 运动编码器 (TCN, 全程4D) ──
        self.tcn = TCNEncoder2D(motion_dim, list(tcn_channels), tcn_kernel, tcn_dropout)
        tcn_out = tcn_channels[-1]

        # 时序池化: 拼接 [最后时刻, 全局平均] → 投影
        self.motion_proj = nn.Sequential(
            nn.Linear(tcn_out * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
        )

        # ── 校准编码器 (MLP) ──
        self.calib_encoder = CalibEncoder(calib_dim, list(calib_hidden), calib_dropout)

        # ── 融合层 ──
        fusion_in = 128 + calib_hidden[-1]
        fusion_layers = []
        in_dim = fusion_in
        for h_dim in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
            ])
            in_dim = h_dim
        self.fusion = nn.Sequential(*fusion_layers)

        # ── 回归头（双通道独立输出）──
        self.head_biceps = nn.Linear(fusion_hidden[-1], 1)
        self.head_triceps = nn.Linear(fusion_hidden[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 26, 1, 64]  合并输入
                前 motion_dim 通道 = 运动序列
                后 calib_dim  通道 = 校准向量(取空间位置 (0,0))
        Returns:
            [B, 2]
        """
        # 1. 分离 motion 和 calib
        motion = x[:, :self.motion_dim, :, :]       # [B, 10, 1, 64]
        calib  = x[:, self.motion_dim:, 0, 0]        # [B, 16]  取 (h=0, w=0)

        # 2. TCN 编码 (全程 4D)
        tcn_out = self.tcn(motion)                    # [B, C_tcn, 1, 64]

        # 3. 时序池化: 压缩 H=1 -> 取最后时刻 + 全局平均
        tcn_3d = tcn_out.squeeze(2)                  # [B, C_tcn, 64]  临时压 H 维
        last = tcn_3d[:, :, -1]                      # [B, C_tcn]
        avg  = tcn_3d.mean(dim=2)                    # [B, C_tcn]
        h_motion = self.motion_proj(torch.cat([last, avg], dim=-1))  # [B, 128]

        # 4. MLP 编码校准向量
        h_calib = self.calib_encoder(calib)           # [B, calib_hidden[-1]]

        # 5. 融合 + 回归
        fused = self.fusion(torch.cat([h_motion, h_calib], dim=-1))
        biceps  = self.head_biceps(fused)
        triceps = self.head_triceps(fused)

        return torch.cat([biceps, triceps], dim=-1)
