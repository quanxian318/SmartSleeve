#!/usr/bin/env python3
"""
export_bpu_v2.py — 从 BPU 版权重导出 opset=11 的 ONNX
=========================================================
- 全程 4D Conv2d, 无 squeeze/transpose 回 3D
- opset=11 (D-Robotics 兼容)
- 固定 shape, 无动态维度
- do_constant_folding=True 减少冗余节点

用法:
    python export_bpu_v2.py --checkpoint results/anchorcalib_tcn_bpu_v2.pt --output results/anchorcalib_tcn_bpu_v2.onnx
"""

import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_bpu import AnchorCalibTCN_BPU


def export(ckpt_path, onnx_path, verify=True):
    print("=" * 60)
    print("  BPU ONNX Export (opset=11, 4D native)")
    print("=" * 60)

    # ── 1. 加载模型 ──
    print(f"\n[1/4] 加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    state = ckpt['model_state_dict']

    md = cfg.get('motion_dim', 10)
    cd = cfg.get('calib_dim', 16)
    ws = cfg.get('window_size', 64)

    model = AnchorCalibTCN_BPU(
        motion_dim=md, calib_dim=cd, window_size=ws,
        tcn_channels=tuple(cfg.get('tcn_channels', (128, 256, 256, 256))),
        calib_hidden=tuple(cfg.get('calib_hidden', (96, 192, 96))),
        fusion_hidden=tuple(cfg.get('fusion_hidden', (384, 192))),
    )
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数: {n_params:,}  |  motion_dim={md}  calib_dim={cd}  window={ws}")

    # ── 2. 测试推理 ──
    total_dim = md + cd
    dummy = torch.randn(1, total_dim, 1, ws)   # [B, C, H, W]
    with torch.no_grad():
        out = model(dummy)
    print(f"  输入: [1, {total_dim}, 1, {ws}]  ->  输出: {out.numpy()[0]}")

    # ── 3. 导出 ONNX ──
    print(f"\n[2/4] 导出 ONNX (opset=11, 固定shape, 常量折叠)...")
    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)

    # 使用旧版 torch.onnx.export (dynamo=False) 以兼容 opset=11
    # PyTorch 2.x 默认 dynamo=True, 最低只支持 opset=18
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['merged_input'],
        output_names=['emg_ratio'],
        dynamic_axes={},           # 固定 shape, BPU 友好
        dynamo=False,              # 旧版导出器, 兼容低 opset
    )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  已导出: {onnx_path}  ({size_kb:.0f} KB)")

    # ── 4. 验证 ──
    if verify:
        print(f"\n[3/4] 验证 (PyTorch vs ONNX Runtime)...")
        verify_onnx(model, onnx_path, md, cd, ws)

    # ── 算子分析 ──
    print(f"\n[4/4] ONNX 算子分析...")
    analyze_ops(onnx_path)

    print(f"\n{'='*60}")
    print(f"  完成! 输出: {onnx_path}")
    print(f"  下一步: hb_mapper checker --model-type onnx --model {onnx_path}")
    print(f"{'='*60}")


def verify_onnx(model, onnx_path, motion_dim, calib_dim, window_size):
    """对比 PyTorch 和 ONNX Runtime 输出."""
    total_dim = motion_dim + calib_dim
    torch.manual_seed(42)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except ImportError:
        print("  [SKIP] onnxruntime 未安装")
        return

    max_diffs = []
    for i in range(10):
        x = torch.randn(1, total_dim, 1, window_size)
        with torch.no_grad():
            out_pt = model(x).numpy()
        out_ort = sess.run(None, {'merged_input': x.numpy().astype(np.float32)})[0]
        diff = np.abs(out_pt - out_ort).max()
        max_diffs.append(diff)

    print(f"  PyTorch vs ONNX Runtime: max={max(max_diffs):.2e}  mean={np.mean(max_diffs):.2e}")
    if max(max_diffs) < 1e-5:
        print(f"  [OK] ONNX 导出精度无损")
    else:
        print(f"  [WARN] 存在微小差异 (可能是 opset 版本差异)")


def analyze_ops(onnx_path):
    """分析 ONNX 算子分布, 评估 BPU 兼容性."""
    import onnx
    from collections import Counter

    model = onnx.load(onnx_path)
    ops = Counter()
    for node in model.graph.node:
        ops[node.op_type] += 1

    # 分类: BPU-friendly vs risky
    bpu_friendly = {'Conv', 'Relu', 'BatchNormalization', 'Add', 'Mul', 'Concat', 'Gemm'}
    bpu_risky = {'Pad', 'LayerNormalization', 'ReduceMean', 'Reshape', 'Slice', 'Gather',
                 'Squeeze', 'Unsqueeze', 'Transpose', 'ConstantOfShape', 'Cast', 'Shape'}

    friendly_count = 0
    risky_count = 0
    other_count = 0

    print(f"  {'Operator':<22s} {'Count':>6s}  {'BPU':>8s}")
    print(f"  {'-'*38}")
    for op, count in ops.most_common():
        if op in bpu_friendly:
            tag = 'OK'
            friendly_count += count
        elif op in bpu_risky:
            tag = 'RISKY'
            risky_count += count
        else:
            tag = '?'
            other_count += count
        print(f"  {op:<22s} {count:>6d}  {tag:>8s}")

    print(f"  {'-'*38}")
    print(f"  {'Total':<22s} {sum(ops.values()):>6d}")
    print(f"  BPU-friendly: {friendly_count}  |  BPU-risky: {risky_count}  |  Unknown: {other_count}")

    # 输入输出
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else '?' for d in inp.type.tensor_type.shape.dim]
        print(f"\n  输入: {inp.name} {shape}")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else '?' for d in out.type.tensor_type.shape.dim]
        print(f"  输出: {out.name} {shape}")
    print(f"  opset: {model.opset_import[0].version}")


def main():
    parser = argparse.ArgumentParser(description='BPU ONNX Export (opset=11, Conv2d)')
    parser.add_argument('--checkpoint', '-c',
                        default='results/anchorcalib_tcn_bpu_v2.pt',
                        help='BPU 版 .pt checkpoint')
    parser.add_argument('--output', '-o',
                        default='results/anchorcalib_tcn_bpu_v2.onnx',
                        help='输出 ONNX 路径')
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过 ONNX Runtime 验证')
    args = parser.parse_args()

    export(args.checkpoint, args.output, verify=not args.no_verify)


if __name__ == '__main__':
    main()
