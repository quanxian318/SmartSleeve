"""
AnchorCalib-TCN: 基于标准动作校准的个体化肌电预测网络
========================================================
Architecture:
  Motion Encoder (TCN):    angle_trajectory[t-D:t] → h_motion
  Calibration Encoder (MLP): personal_calib_vector → h_calib
  Fusion + Regression Head:  [h_motion, h_calib] → [biceps_ratio, triceps_ratio]

输入:
  - motion: [B, T, motion_dim] — 运动学时间窗口
  - calib:  [B, calib_dim]    — 个人校准向量

输出:
  - ratios: [B, 2] — [肱二头肌激活比例, 肱三头肌激活比例]

还原公式:
  V_emg = ratio × (V_90 - V_rest) + V_rest
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CausalConv1d — 因果卷积（只看过去帧，无未来信息泄漏）
# ============================================================
class CausalConv1d(nn.Module):
    """1D convolution with causal padding (左填充，右不填)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation,
        )

    def forward(self, x):
        # x: [B, C, T] → pad left only → conv
        return self.conv(F.pad(x, (self.padding, 0)))


# ============================================================
# TemporalBlock — 残差时序卷积块
# ============================================================
class TemporalBlock(nn.Module):
    """Residual block: CausalConv → ReLU → Dropout → CausalConv → ReLU → Dropout + Residual."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.conv2(out)
        out = self.relu(out + self.downsample(x))
        return self.dropout(out)


# ============================================================
# TCNEncoder — 时序卷积编码器
# ============================================================
class TCNEncoder(nn.Module):
    """
    堆叠多层 TemporalBlock，dilation 指数增长 (1, 2, 4, 8, ...)。
    输入:  [B, T, C_in]
    输出:  [B, T, C_out]
    """

    def __init__(self, input_dim, hidden_channels, kernel_size=5, dropout=0.2):
        """
        Args:
            input_dim:       每帧运动特征数
            hidden_channels: 每层通道数列表，如 [128, 256, 256, 256]
            kernel_size:     卷积核大小
            dropout:         Dropout 比例
        """
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(hidden_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation, dropout))
            in_ch = ch
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_channels[-1]

    def forward(self, x):
        """
        Args:
            x: [B, T, C_in] 运动序列
        Returns:
            [B, T, C_out] 编码后的时序特征
        """
        x = x.transpose(1, 2)          # [B, C, T] for Conv1d
        x = self.network(x)
        return x.transpose(1, 2)       # [B, T, C]


# ============================================================
# CalibEncoder — 校准向量编码器
# ============================================================
class CalibEncoder(nn.Module):
    """MLP 编码个人校准向量 → 个体化嵌入."""

    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        """
        Args:
            input_dim:   校准向量维度 (默认 16)
            hidden_dims: 隐藏层维度列表，如 [96, 192, 96]
        """
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
        """calib: [B, calib_dim] → [B, output_dim]"""
        return self.network(calib)


# ============================================================
# AnchorCalibTCN — 完整模型
# ============================================================
class AnchorCalibTCN(nn.Module):
    """
    AnchorCalib-TCN: 校准引导的个性化肌电预测网络。

    输入:
        motion : [B, T, motion_dim]  运动学窗口 (角度/角速度/角加速度/相位)
        calib  : [B, calib_dim]      个人校准向量

    输出:
        ratios : [B, 2]              [biceps_ratio, triceps_ratio]
    """

    def __init__(
        self,
        motion_dim=6,
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
        """
        Args:
            motion_dim:     运动特征维度
            calib_dim:      校准向量维度
            window_size:    时间窗口长度
            tcn_channels:   TCN 各层通道数
            tcn_kernel:     TCN 卷积核大小
            tcn_dropout:    TCN dropout
            calib_hidden:   校准编码器隐藏层维度
            calib_dropout:  校准编码器 dropout
            fusion_hidden:  融合层隐藏维度
            fusion_dropout: 融合层 dropout
        """
        super().__init__()
        self.motion_dim = motion_dim
        self.calib_dim = calib_dim
        self.window_size = window_size

        # ── 运动编码器 (TCN) ──
        self.tcn = TCNEncoder(motion_dim, list(tcn_channels), tcn_kernel, tcn_dropout)
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
        self._print_param_count()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [AnchorCalibTCN] Total params: {total:,} | Trainable: {trainable:,}")

    def forward(self, motion, calib):
        """
        Args:
            motion: [B, T, motion_dim]  运动学时间窗口
            calib:  [B, calib_dim]      个人校准向量
        Returns:
            [B, 2]  [biceps_ratio, triceps_ratio]
        """
        # 1. TCN 编码运动序列
        tcn_out = self.tcn(motion)                    # [B, T, tcn_channels[-1]]

        # 2. 时序池化：最后时刻 + 全局平均
        last = tcn_out[:, -1, :]                      # [B, C]
        avg = tcn_out.mean(dim=1)                     # [B, C]
        h_motion = self.motion_proj(torch.cat([last, avg], dim=-1))  # [B, 128]

        # 3. MLP 编码校准向量
        h_calib = self.calib_encoder(calib)            # [B, calib_hidden[-1]]

        # 4. 融合 + 回归
        fused = self.fusion(torch.cat([h_motion, h_calib], dim=-1))

        biceps = self.head_biceps(fused)
        triceps = self.head_triceps(fused)

        return torch.cat([biceps, triceps], dim=-1)


# ============================================================
# MLP Baseline（消融实验用）
# ============================================================
class AnchorCalibMLP(nn.Module):
    """
    单帧 MLP 基线：仅输入当前帧 + 校准向量，无时序建模。
    用于消融实验证明 TCN 时序建模的价值。
    """

    def __init__(self, motion_dim=6, calib_dim=16, hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        self.calib_encoder = CalibEncoder(calib_dim, [64, 128, 64], dropout)

        in_dim = motion_dim + 64
        layers = []
        d_in = in_dim
        for h in hidden:
            layers.extend([
                nn.Linear(d_in, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout),
            ])
            d_in = h
        self.body = nn.Sequential(*layers)
        self.head_b = nn.Linear(hidden[-1], 1)
        self.head_t = nn.Linear(hidden[-1], 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, motion, calib):
        # motion: [B, T, D] — 只取最后一帧
        x = motion[:, -1, :]                           # [B, D]
        c = self.calib_encoder(calib)                  # [B, 64]
        h = self.body(torch.cat([x, c], dim=-1))
        return torch.cat([self.head_b(h), self.head_t(h)], dim=-1)
