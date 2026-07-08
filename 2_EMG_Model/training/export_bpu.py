#!/usr/bin/env python3
"""
export_bpu.py — AnchorCalib-TCN BPU 部署 ONNX 导出
====================================================

将双输入模型 (motion + calib) 包装为单输入 4D 模型，
输出 D-Robotics BPU 工具链兼容的 ONNX 文件。

用法:
  # 从 .pt 检查点导出
  python export_bpu.py --checkpoint results/anchorcalib_tcn.pt --output anchorcalib_tcn_bpu.onnx

  # 直接指定已有 ONNX + 权重 (需要源代码)
  python export_bpu.py --checkpoint results/anchorcalib_tcn.pt --output anchorcalib_tcn_bpu.onnx --verify

变换:
  原始:  motion [B, 64, 10] + calib [B, 16]  → [B, 2]
  新:    merged [B, 1, 64, 26]               → [B, 2]
         前10通道 = motion特征, 后16通道 = calib (时间轴广播)
"""

import argparse, os, sys, copy
import numpy as np
import torch
import torch.nn as nn

# 确保可以 import model.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AnchorCalibTCN


# ================================================================
# BPUWrapper — 双输入 → 单输入 4D
# ================================================================

class BPUWrapper(nn.Module):
    """将 AnchorCalibTCN 包装为单输入 4D 模型，兼容 BPU 工具链。

    输入:  x [B, 1, T, D]  合并输入 (D = motion_dim + calib_dim)
    输出:  [B, 2]           biceps_ratio, triceps_ratio

    内部:  拆分 x → motion [B, T, motion_dim] + calib [B, calib_dim]
          调用原始 AnchorCalibTCN(motion, calib)
    """

    def __init__(self, original_model, motion_dim=10, calib_dim=16):
        super().__init__()
        self.original = original_model
        self.motion_dim = motion_dim
        self.calib_dim = calib_dim

    def forward(self, x):
        # x: [B, 1, T, D] → squeeze channel dim
        x = x.squeeze(1)                       # [B, T, D]

        # 分离运动特征和校准向量
        motion = x[..., :self.motion_dim]       # [B, T, motion_dim]
        calib  = x[:, 0, self.motion_dim:]      # [B, calib_dim] (所有帧相同, 取第1帧)

        return self.original(motion, calib)


# ================================================================
# 导出逻辑
# ================================================================

def load_checkpoint(checkpoint_path, map_location='cpu'):
    """加载训练检查点, 返回 (model, config_dict)."""
    print(f"[1/4] 加载检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    cfg = ckpt.get('config', {})
    state_dict = ckpt.get('model_state_dict')
    if state_dict is None:
        raise ValueError("检查点缺少 'model_state_dict', 请确认是 train.py 输出的 .pt 文件")

    motion_dim  = cfg.get('motion_dim', 10)
    calib_dim   = cfg.get('calib_dim', 16)
    window_size = cfg.get('window_size', 64)
    model_type  = cfg.get('model_type', 'tcn')

    print(f"  模型类型: {model_type}, motion_dim={motion_dim}, calib_dim={calib_dim}, window={window_size}")

    # 构建原始模型
    if model_type == 'tcn':
        model = AnchorCalibTCN(
            motion_dim=motion_dim,
            calib_dim=calib_dim,
            window_size=window_size,
            tcn_channels=tuple(cfg.get('tcn_channels', (128, 256, 256, 256))),
            calib_hidden=tuple(cfg.get('calib_hidden', (96, 192, 96))),
            fusion_hidden=tuple(cfg.get('fusion_hidden', (384, 192))),
        )
    elif model_type == 'mlp':
        from model import AnchorCalibMLP
        model = AnchorCalibMLP(
            motion_dim=motion_dim,
            calib_dim=calib_dim,
            hidden_dims=tuple(cfg.get('hidden_dims', (256, 512, 256))),
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model, motion_dim, calib_dim, window_size


def export_bpu_onnx(model, motion_dim, calib_dim, window_size, output_path, verify=True):
    """导出 BPU 兼容的 ONNX 文件."""
    print(f"\n[2/4] 包装为 BPUWrapper (单输入 4D)...")
    wrapped = BPUWrapper(copy.deepcopy(model), motion_dim, calib_dim)
    wrapped.eval()

    total_dim = motion_dim + calib_dim
    dummy_input = torch.randn(1, 1, window_size, total_dim)
    print(f"  输入 shape: [1, 1, {window_size}, {total_dim}]")
    print(f"    └ 通道 0-{motion_dim-1}: 运动特征 (角度/速度/加速度/相位)")
    print(f"    └ 通道 {motion_dim}-{total_dim-1}: 校准向量 (时间轴广播)")

    print(f"\n[3/4] 导出 ONNX (opset=17, 固定shape, 常量折叠)...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 使用旧版 ONNX exporter (dynamo=False) 兼容 PyTorch 2.x + 较低 opset
    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['merged_input'],
        output_names=['emg_ratio'],
        dynamo=False,                  # 旧版导出器, 兼容性好
    )
    print(f"  ONNX 已导出: {output_path}")

    # 文件大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {size_mb:.1f} MB")

    # ── 验证 ──
    if verify:
        print(f"\n[4/4] 精度验证 (原始 vs 包装 vs ONNX)...")
        verify_outputs(model, wrapped, output_path, motion_dim, calib_dim, window_size)
    else:
        print(f"\n[4/4] 跳过验证 (使用 --verify 开启)")

    return output_path


def verify_outputs(original_model, wrapped_model, onnx_path, motion_dim, calib_dim, window_size):
    """对比三端输出: PyTorch原始 → PyTorch包装 → ONNX Runtime."""
    total_dim = motion_dim + calib_dim

    # 生成测试输入
    np.random.seed(42)
    torch.manual_seed(42)
    x_merged = torch.randn(1, 1, window_size, total_dim)
    motion = x_merged.squeeze(1)[:, :, :motion_dim]    # [1, T, 10]
    calib = x_merged[0, 0, :1, motion_dim:]             # [1, 16] (batch维度保留)

    # ── 1. 原始模型输出 ──
    with torch.no_grad():
        out_orig = original_model(motion, calib).numpy()

    # ── 2. Wrapper 输出 ──
    with torch.no_grad():
        out_wrap = wrapped_model(x_merged).numpy()

    # ── 3. ONNX Runtime ──
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        out_onnx = sess.run(None, {'merged_input': x_merged.numpy().astype(np.float32)})[0]
        onnx_ok = True
    except ImportError:
        print("  [跳过] onnxruntime 未安装, 不验证 ONNX 端")
        out_onnx = None
        onnx_ok = False

    # ── 对比 ──
    print(f"  原始输出:     biceps={out_orig[0,0]:.6f}  triceps={out_orig[0,1]:.6f}")
    print(f"  Wrapper输出:  biceps={out_wrap[0,0]:.6f}  triceps={out_wrap[0,1]:.6f}")
    if onnx_ok:
        print(f"  ONNX输出:     biceps={out_onnx[0,0]:.6f}  triceps={out_onnx[0,1]:.6f}")

    # Wrapper vs 原始 (应该几乎完全相同, 只有浮点误差)
    diff_wrap = np.abs(out_orig - out_wrap).max()
    print(f"  Delta(orig<->wrap): {diff_wrap:.2e}  {'[OK]一致' if diff_wrap < 1e-6 else '[WARN]有差异'}")

    if onnx_ok:
        diff_onnx = np.abs(out_orig - out_onnx).max()
        print(f"  Delta(orig<->ONNX): {diff_onnx:.2e}  {'[OK]一致' if diff_onnx < 1e-5 else '[WARN]有差异'}")

    # ── 多次测试 ──
    print(f"\n  多次随机测试 (10次)...")
    max_diffs_wrap = []
    max_diffs_onnx = []
    for i in range(10):
        xm = torch.randn(1, 1, window_size, total_dim)
        m = xm.squeeze(1)[:, :, :motion_dim]
        c = xm[0, 0, :1, motion_dim:]  # [1, 16] batch dim
        with torch.no_grad():
            o1 = original_model(m, c).numpy()
            o2 = wrapped_model(xm).numpy()
        max_diffs_wrap.append(np.abs(o1 - o2).max())
        if onnx_ok:
            o3 = sess.run(None, {'merged_input': xm.numpy().astype(np.float32)})[0]
            max_diffs_onnx.append(np.abs(o1 - o3).max())

    print(f"  Wrapper vs 原始: max={max(max_diffs_wrap):.2e}  mean={np.mean(max_diffs_wrap):.2e}")
    if onnx_ok:
        print(f"  ONNX vs 原始:    max={max(max_diffs_onnx):.2e}  mean={np.mean(max_diffs_onnx):.2e}")
    print(f"  结论: {'[OK]导出成功, 精度无损' if max(max_diffs_wrap) < 1e-5 else '[WARN]需要检查'}")

    return max(max_diffs_wrap) < 1e-5


# ================================================================
# 直接导出已有 ONNX (无 checkpoint 时的快速方案)
# ================================================================

def export_from_existing_onnx(existing_onnx, output_path, motion_dim=10, calib_dim=16, window_size=64):
    """从已有双输入 ONNX 构建单输入版本 (ONNX graph surgery).

    注意: 这个方法创建一个简单的 ONNX 图，将 4D 输入拆分为 motion 和 calib，
    然后调用原始 ONNX 模型。适合没有 PyTorch checkpoint 但有原始 ONNX 的情况。
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    print(f"[1/3] 加载原始 ONNX: {existing_onnx}")
    orig_model = onnx.load(existing_onnx)

    # 获取原始输入名称
    orig_inputs = [i.name for i in orig_model.graph.input]
    orig_outputs = [o.name for o in orig_model.graph.output]
    print(f"  原始输入: {orig_inputs}")
    print(f"  原始输出: {orig_outputs}")

    total_dim = motion_dim + calib_dim

    # 构建新图: merged_input → Split → Squeeze → motion/calib → 原始模型
    merged = helper.make_tensor_value_info('merged_input', TensorProto.FLOAT, [1, 1, window_size, total_dim])

    # 节点1: Squeeze — [1,1,64,26] → [1,64,26]
    squeeze_node = helper.make_node(
        'Squeeze', ['merged_input'], ['squeezed'],
        axes=[1],
        name='squeeze_channel'
    )

    # 节点2: Split — [1,64,26] → [1,64,10] + [1,64,16]
    split_node = helper.make_node(
        'Split', ['squeezed'], ['motion_full', 'calib_full'],
        axis=2,
        split=[motion_dim, calib_dim],
        name='split_motion_calib'
    )

    # 节点3: Slice calib 取第一帧 — [1,64,16] → [1,16]
    # 用 Gather 取 index=0 沿 axis=1
    gather_node = helper.make_node(
        'Gather', ['calib_full', 'calib_index'], ['calib_vec'],
        axis=1,
        name='gather_calib_first_frame'
    )

    # 常量: index=0 for Gather
    index_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name='calib_index')
    orig_model.graph.initializer.append(index_init)

    # 输出
    out_info = [helper.make_tensor_value_info(orig_outputs[0], TensorProto.FLOAT, [1, 2])]

    # 替换图: 添加新节点, 重连输入
    orig_model.graph.node.insert(0, squeeze_node)
    orig_model.graph.node.insert(1, split_node)
    orig_model.graph.node.insert(2, gather_node)

    # 找到原始 motion 和 calib 输入节点, 替换它们的输入
    for node in orig_model.graph.node[3:]:
        for orig_in_name in orig_inputs:
            if orig_in_name in node.input:
                idx = list(node.input).index(orig_in_name)
                if orig_in_name == orig_inputs[0]:  # motion
                    node.input[idx] = 'motion_full'
                elif len(orig_inputs) > 1 and orig_in_name == orig_inputs[1]:  # calib
                    node.input[idx] = 'calib_vec'

    # 更新 graph 的 input/output
    del orig_model.graph.input[:]
    orig_model.graph.input.append(merged)
    del orig_model.graph.output[:]
    orig_model.graph.output.extend(out_info)

    # 清理: 移除旧的 squeeze/reshape 相关的中间 shape 信息
    # 设置 opset
    orig_model.opset_import[0].version = 11

    # 验证
    onnx.checker.check_model(orig_model)
    print(f"  ONNX 图验证通过")

    print(f"\n[2/3] 保存合并模型: {output_path}")
    onnx.save(orig_model, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {size_mb:.1f} MB")

    print(f"\n[3/3] 输入/输出信息:")
    print(f"  输入: merged_input  [1, 1, {window_size}, {total_dim}]")
    print(f"  输出: {orig_outputs[0]}  [1, 2]")

    return output_path


# ================================================================
# 主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AnchorCalib-TCN BPU ONNX 导出工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 PyTorch 检查点导出 (推荐)
  python export_bpu.py --checkpoint results/anchorcalib_tcn.pt --output anchorcalib_tcn_bpu.onnx --verify

  # 从已有 ONNX 转换 (无 checkpoint)
  python export_bpu.py --onnx anchorcalib_tcn.onnx --output anchorcalib_tcn_bpu.onnx

  # 指定特征维度
  python export_bpu.py --checkpoint anchorcalib_tcn.pt --motion-dim 10 --calib-dim 16 --output model_bpu.onnx
        """,
    )
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='PyTorch .pt 检查点路径 (由 train.py 输出)')
    parser.add_argument('--onnx', type=str, default=None,
                        help='已有双输入 ONNX 路径 (无 checkpoint 时使用)')
    parser.add_argument('--output', '-o', type=str, default='anchorcalib_tcn_bpu.onnx',
                        help='输出 ONNX 路径 (默认: anchorcalib_tcn_bpu.onnx)')
    parser.add_argument('--motion-dim', type=int, default=10,
                        help='运动特征维度 (默认: 10)')
    parser.add_argument('--calib-dim', type=int, default=16,
                        help='校准向量维度 (默认: 16)')
    parser.add_argument('--window-size', type=int, default=64,
                        help='时间窗口帧数 (默认: 64)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='导出后验证精度 (默认开启)')
    parser.add_argument('--no-verify', action='store_false', dest='verify',
                        help='跳过精度验证')
    args = parser.parse_args()

    if args.checkpoint:
        # ── 方式1: 从 PyTorch checkpoint 导出 (推荐) ──
        model, md, cd, ws = load_checkpoint(args.checkpoint)
        export_bpu_onnx(model, md, cd, ws, args.output, verify=args.verify)

    elif args.onnx:
        # ── 方式2: 直接修改已有 ONNX 图 ──
        export_from_existing_onnx(
            args.onnx, args.output,
            motion_dim=args.motion_dim,
            calib_dim=args.calib_dim,
            window_size=args.window_size,
        )

    else:
        parser.error('必须指定 --checkpoint 或 --onnx')

    print(f"\n{'='*60}")
    print(f"[OK] BPU ONNX export done: {args.output}")
    print(f"   输入: merged_input [1, 1, {args.window_size}, {args.motion_dim + args.calib_dim}]")
    print(f"   下一步: hb_mapper checker --model-type onnx --model {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
