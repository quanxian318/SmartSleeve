#!/usr/bin/env python3
"""
migrate_weights.py — 原版 Conv1d 权重迁移到 BPU Conv2d 版
=============================================================
原理: Conv1d weight [out, in, kernel] 和 Conv2d weight [out, in, 1, kernel]
      只差一维 (H=1), 语义完全一致 → unsqueeze(2) 即可, 精度 100% 保留。

用法:
    python migrate_weights.py --checkpoint results/anchorcalib_tcn.pt --output results/anchorcalib_tcn_bpu_v2.pt
"""

import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AnchorCalibTCN
from model_bpu import AnchorCalibTCN_BPU


def migrate(checkpoint_path, output_path, verify=True):
    """加载原版 checkpoint -> 迁移权重 -> 保存 BPU 版 checkpoint."""

    print("=" * 60)
    print("  Conv1d -> Conv2d 权重迁移")
    print("=" * 60)

    # ── 1. 加载原版 checkpoint ──
    print(f"\n[1/5] 加载原版 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    old_state = ckpt['model_state_dict']
    cfg = ckpt.get('config', {})

    # 清理 _orig_mod 前缀 (torch.compile 残留)
    old_state_clean = {}
    for k, v in old_state.items():
        new_k = k.replace('_orig_mod.', '')
        old_state_clean[new_k] = v
    old_state = old_state_clean

    motion_dim  = cfg.get('motion_dim', 10)
    calib_dim   = cfg.get('calib_dim', 16)
    window_size = cfg.get('window_size', 64)
    tcn_ch      = tuple(cfg.get('tcn_channels', (128, 256, 256, 256)))
    calib_h     = tuple(cfg.get('calib_hidden', (96, 192, 96)))
    fusion_h    = tuple(cfg.get('fusion_hidden', (384, 192)))

    print(f"  motion_dim={motion_dim}, calib_dim={calib_dim}, window={window_size}")
    print(f"  tcn_channels={tcn_ch}, calib_hidden={calib_h}, fusion_hidden={fusion_h}")

    # ── 2. 创建两个模型 ──
    print(f"\n[2/5] 创建原版 + BPU 版模型...")
    old_model = AnchorCalibTCN(
        motion_dim=motion_dim, calib_dim=calib_dim, window_size=window_size,
        tcn_channels=tcn_ch, calib_hidden=calib_h, fusion_hidden=fusion_h,
    )
    old_model.load_state_dict(old_state)
    old_model.eval()

    new_model = AnchorCalibTCN_BPU(
        motion_dim=motion_dim, calib_dim=calib_dim, window_size=window_size,
        tcn_channels=tcn_ch, calib_hidden=calib_h, fusion_hidden=fusion_h,
    )

    n_old = sum(p.numel() for p in old_model.parameters())
    n_new = sum(p.numel() for p in new_model.parameters())
    print(f"  原版参数: {n_old:,}  |  BPU版参数: {n_new:,}")

    # ── 3. 权重映射 ──
    print(f"\n[3/5] 迁移权重...")
    new_state = new_model.state_dict()
    migrated_conv = 0
    migrated_other = 0
    missing = []

    # 建立 old_state 的快速查找表
    old_keys = list(old_state.keys())

    for new_key, new_param in new_state.items():
        if new_key in old_state:
            old_tensor = old_state[new_key]
            if old_tensor.shape == new_param.shape:
                # 形状一致, 直接复制
                new_state[new_key] = old_tensor
                migrated_other += 1
            elif old_tensor.ndim == 3 and new_param.ndim == 4 and (
                'conv' in new_key.lower() or 'downsample' in new_key.lower()
            ):
                # Conv1d [out, in, k] -> Conv2d [out, in, 1, k]
                # downsample Conv1d [out, in, 1] -> Conv2d [out, in, 1, 1]
                new_state[new_key] = old_tensor.unsqueeze(2)
                migrated_conv += 1
            else:
                print(f"  [SHAPE MISMATCH] {new_key}: old={old_tensor.shape} new={new_param.shape}")
                missing.append(new_key)
        else:
            missing.append(new_key)

    print(f"  Conv1d->Conv2d 迁移: {migrated_conv} 个")
    print(f"  直接复制:           {migrated_other} 个")
    if missing:
        print(f"  [WARN] 未匹配: {len(missing)} 个")
        for m in missing:
            print(f"         - {m}")

    # ── 4. 加载迁移后的权重 ──
    new_model.load_state_dict(new_state, strict=False)
    new_model.eval()

    # ── 5. 精度验证 ──
    if verify:
        print(f"\n[4/5] 精度验证...")
        torch.manual_seed(42)

        # 原版输入
        motion = torch.randn(4, window_size, motion_dim)
        calib  = torch.randn(4, calib_dim)

        with torch.no_grad():
            out_old = old_model(motion, calib)

        # BPU 版输入: 合并为 [B, 26, 1, 64]
        # motion: [B, 64, 10] -> [B, 10, 1, 64]
        motion_4d = motion.permute(0, 2, 1).unsqueeze(2)            # [B, 10, 1, 64]
        calib_4d  = calib[:, :, None, None].expand(-1, -1, 1, window_size)  # [B, 16, 1, 64]
        merged = torch.cat([motion_4d, calib_4d], dim=1)            # [B, 26, 1, 64]

        with torch.no_grad():
            out_new = new_model(merged)

        diff = (out_old - out_new).abs().max().item()
        print(f"  原版输出:  {out_old[0].numpy()}")
        print(f"  BPU版输出: {out_new[0].numpy()}")
        print(f"  Max delta: {diff:.2e}")

        if diff < 1e-5:
            print(f"  [OK] 权重迁移成功, 精度完全无损!")
        elif diff < 1e-3:
            print(f"  [OK] 差异在可接受范围内 (浮点误差)")
        else:
            print(f"  [WARN] 差异较大, 请检查未匹配的权重")

        # 多次验证
        max_diffs = []
        for i in range(10):
            motion = torch.randn(4, window_size, motion_dim)
            calib  = torch.randn(4, calib_dim)
            with torch.no_grad():
                o1 = old_model(motion, calib)
            motion_4d = motion.permute(0, 2, 1).unsqueeze(2)
            calib_4d  = calib[:, :, None, None].expand(-1, -1, 1, window_size)
            merged = torch.cat([motion_4d, calib_4d], dim=1)
            with torch.no_grad():
                o2 = new_model(merged)
            max_diffs.append((o1 - o2).abs().max().item())

        print(f"  10次随机测试: max={max(max_diffs):.2e}  mean={np.mean(max_diffs):.2e}")
        print(f"  结论: {'[OK] 迁移成功' if max(max_diffs) < 1e-5 else '[WARN] 需检查'}")

    # ── 保存 ──
    print(f"\n[5/5] 保存 BPU 版 checkpoint: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': new_model.state_dict(),
        'config': {
            **cfg,
            'model_type': 'tcn_bpu',
            'input_format': '4D merged [B, 26, 1, 64]',
            'note': 'Conv2d BPU-native version, weights migrated from Conv1d via unsqueeze(2)',
        }
    }, output_path)
    print(f"  文件大小: {os.path.getsize(output_path)/1024/1024:.1f} MB")
    print(f"\n{'='*60}")
    print(f"  完成! 输出: {output_path}")
    print(f"{'='*60}")

    return new_model


def main():
    parser = argparse.ArgumentParser(description='Conv1d -> Conv2d 权重迁移')
    parser.add_argument('--checkpoint', '-c',
                        default='results/anchorcalib_tcn.pt',
                        help='原版 .pt checkpoint 路径')
    parser.add_argument('--output', '-o',
                        default='results/anchorcalib_tcn_bpu_v2.pt',
                        help='输出 BPU 版 checkpoint 路径')
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过精度验证')
    args = parser.parse_args()

    migrate(args.checkpoint, args.output, verify=not args.no_verify)


if __name__ == '__main__':
    main()
