"""
AnchorMLP ONNX 底层算子级量化误差分析
================================================================================
使用 ONNX Runtime 展开计算图中每一个底层算子节点 (matmul/add/mul/relu/norm...)
比较 Float32 ONNX vs INT8 量化 ONNX 在每个中间 tensor 的余弦相似度
================================================================================
"""
import os, sys, io, csv, json, warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np

# ============================================================
# ===== 用户配置 =====
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PT_PATH = os.path.join(SCRIPT_DIR, 'results_loso', 'anchor_mlp_2.pt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results_loso', 'test_report')
ONNX_FP32_PATH = os.path.join(OUTPUT_DIR, 'anchor_mlp_fp32.onnx')
ONNX_INT8_PATH = os.path.join(OUTPUT_DIR, 'anchor_mlp_int8.onnx')
ONNX_FP32_ALLNODES = os.path.join(OUTPUT_DIR, 'anchor_mlp_fp32_allnodes.onnx')
ONNX_INT8_ALLNODES = os.path.join(OUTPUT_DIR, 'anchor_mlp_int8_allnodes.onnx')
N_TEST_SAMPLES = 500
RANDOM_SEED = 42
EPS = 1e-8

FEATURE_MEANS = np.array([161.3, 76.4, 22.0, 177.2, 69.1, 0.5, 45.0,
                          2.0, 0.6, 0.6, 45.0, 1.0, 0.0, 0.0], dtype=np.float32)
FEATURE_STDS  = np.array([96.2, 27.8, 1.3, 8.7, 7.6, 0.5, 37.5,
                          2.0, 0.3, 0.3, 15.0, 0.5, 50.0, 100.0], dtype=np.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


# ============================================================
# 模型定义
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, expansion=1.5, dropout=0.1):
        super().__init__()
        hidden = int(dim * expansion)
        self.l1 = nn.Linear(dim, hidden); self.l2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim); self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        r = F.relu(self.l1(x)); r = self.dropout(r); r = self.l2(r)
        return self.norm(x + r)

class AnchorMLP(nn.Module):
    def __init__(self, input_dim=14, embed_dim=256, n_blocks=5, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim,embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim, 2.0, dropout) for _ in range(n_blocks)])
        self.skip = nn.Sequential(nn.Linear(3,64), nn.ReLU(), nn.Dropout(dropout))
        self.neck = nn.Sequential(nn.Linear(embed_dim+64,256), nn.LayerNorm(256), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(256,128), nn.ReLU(), nn.Dropout(dropout))
        self.head_b = nn.Linear(128,1); self.head_t = nn.Linear(128,1)
    def forward(self, x):
        feat = self.blocks(self.embed(x))
        sk = self.skip(torch.cat([x[:,6:7], x[:,:2]], dim=1))
        shared = self.neck(torch.cat([feat,sk], dim=1))
        return torch.cat([self.head_b(shared), self.head_t(shared)], dim=1)


# ============================================================
# 导出 FP32 ONNX 并修改为输出所有中间 tensor
# ============================================================
def export_fp32_onnx_and_add_all_outputs(model, onnx_path, allnodes_path, input_shape=(1,14)):
    """导出 FP32 ONNX, 然后修改 graph 输出所有中间 tensor"""
    print("\n[1] 导出 Float32 ONNX...")

    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        do_constant_folding=False,  # 保留所有算子节点
    )
    print(f"  FP32 ONNX: {onnx_path}")

    # 加载并修改: 添加所有中间 tensor 为输出
    g = onnx.load(onnx_path)
    graph = g.graph

    # 收集所有已有的 value_info (中间 tensor)
    existing_outputs = {o.name for o in graph.output}
    all_value_names = set()

    # 从节点输入输出收集
    for node in graph.node:
        for inp in node.input:
            if inp and inp not in existing_outputs:
                all_value_names.add(inp)
        for outp in node.output:
            if outp and outp not in existing_outputs:
                all_value_names.add(outp)

    # 从 initializer 收集
    for init in graph.initializer:
        all_value_names.add(init.name)

    # 从 graph input 收集
    for inp in graph.input:
        all_value_names.add(inp.name)

    # 已有输出
    already_output = {o.name for o in graph.output}

    # 为所有中间值创建输出 (除了 graph input 本身)
    added_count = 0
    skipped = 0
    for name in sorted(all_value_names):
        if name in already_output:
            continue
        # 跳过 initializer (常量), 它们没有运行时值
        is_initializer = any(init.name == name for init in graph.initializer)
        if is_initializer:
            skipped += 1
            continue
        # 添加为输出
        intermediate_value_info = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        graph.output.append(intermediate_value_info)
        added_count += 1

    # 保存
    onnx.save(g, allnodes_path)
    print(f"  原始输出: {len(existing_outputs)} | 新增中间输出: {added_count} | 跳过常量: {skipped}")
    print(f"  总输出节点: {len(graph.output)}")
    print(f"  保存到: {allnodes_path}")

    return g, added_count


# ============================================================
# 生成 INT8 量化 ONNX
# ============================================================
def create_int8_onnx(fp32_onnx_path, int8_onnx_path):
    """使用 onnxruntime 量化工具将 FP32 ONNX 量化为 INT8"""
    print("\n[2] 生成 INT8 量化 ONNX...")

    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
        extra_options={'ActivationSymmetric': True},
    )
    print(f"  INT8 ONNX: {int8_onnx_path}")

    # 同样修改 INT8 ONNX 输出所有中间 tensor
    g = onnx.load(int8_onnx_path)
    graph = g.graph

    existing_outputs = {o.name for o in graph.output}
    all_value_names = set()
    for node in graph.node:
        for inp in node.input:
            if inp: all_value_names.add(inp)
        for outp in node.output:
            if outp: all_value_names.add(outp)
    for init in graph.initializer:
        all_value_names.add(init.name)

    already_output = {o.name for o in graph.output}
    added = 0
    for name in sorted(all_value_names):
        if name in already_output:
            continue
        is_init = any(init.name == name for init in graph.initializer)
        if is_init:
            continue
        intermediate_value_info = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        graph.output.append(intermediate_value_info)
        added += 1

    int8_allnodes_path = int8_onnx_path.replace('.onnx', '_allnodes.onnx')
    onnx.save(g, int8_allnodes_path)
    print(f"  INT8 新增中间输出: {added} | 总输出: {len(graph.output)}")
    print(f"  保存到: {int8_allnodes_path}")

    return g, added, int8_allnodes_path


# ============================================================
# 生成测试数据
# ============================================================
def generate_test_data(n_samples):
    np.random.seed(RANDOM_SEED)
    X = np.random.randn(n_samples, 14).astype(np.float32) * FEATURE_STDS + FEATURE_MEANS
    X[:,0] = np.clip(X[:,0], 10, 500); X[:,1] = np.clip(X[:,1], 10, 200)
    X[:,2] = np.clip(X[:,2], 16, 35); X[:,3] = np.clip(X[:,3], 150, 200)
    X[:,4] = np.clip(X[:,4], 40, 120); X[:,5] = np.clip(X[:,5], 0, 1)
    X[:,6] = np.clip(X[:,6], 0, 130); X[:,10] = np.clip(X[:,10], 0, 130)
    return X


# ============================================================
# ONNX Runtime 推理 — 获取所有中间 tensor
# ============================================================
def run_onnx_with_all_outputs(onnx_path, input_data):
    """用 ONNX Runtime 运行推理，返回所有输出的 dict"""
    # 强制使用 CPU (INT8 量化模型可能不支持 GPU)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.log_severity_level = 3  # ERROR only

    session = ort.InferenceSession(onnx_path, sess_options,
                                    providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    results = session.run(output_names, {input_name: input_data})

    output_dict = {}
    for name, value in zip(output_names, results):
        output_dict[name] = value

    return output_dict


# ============================================================
# 计算余弦相似度
# ============================================================
def compute_cosine(a, b):
    if a is None or b is None:
        return float('nan')
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    if len(a_flat) == 0 or len(b_flat) == 0:
        return float('nan')
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na < EPS or nb < EPS:
        return float('nan')
    return float(dot / (na * nb))


# ============================================================
# 从 ONNX graph 中获取 op_type 映射
# ============================================================
def get_op_type_map(onnx_path):
    """返回 {output_name: op_type} 映射"""
    g = onnx.load(onnx_path)
    op_map = {}
    for node in g.graph.node:
        for outp in node.output:
            op_map[outp] = node.op_type
    return op_map


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  AnchorMLP ONNX 底层算子级量化误差分析")
    print("  Float32 ONNX vs INT8 ONNX — 所有中间 tensor")
    print("=" * 70)

    # ---- 1. 加载 PyTorch 模型 ----
    print("\n[Step 1/7] 加载 PyTorch Float32 模型...")
    ckpt = torch.load(MODEL_PT_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model = AnchorMLP()
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  AnchorMLP | {n_params:,} params")

    # ---- 2. 导出 FP32 ONNX 并输出所有中间 tensor ----
    print("\n[Step 2/7] 导出 FP32 ONNX + 展开所有中间 tensor...")
    _, n_fp32_added = export_fp32_onnx_and_add_all_outputs(
        model, ONNX_FP32_PATH, ONNX_FP32_ALLNODES)

    # ---- 3. 生成 INT8 ONNX ----
    print("\n[Step 3/7] 生成 INT8 量化 ONNX...")
    _, n_int8_added, int8_allnodes_path = create_int8_onnx(ONNX_FP32_PATH, ONNX_INT8_PATH)

    # ---- 4. 生成测试数据 ----
    print(f"\n[Step 4/7] 生成测试数据 ({N_TEST_SAMPLES} 样本)...")
    X_test = generate_test_data(N_TEST_SAMPLES)
    print(f"  shape: {X_test.shape}")

    # ---- 5. ONNX Runtime 推理 ----
    print("\n[Step 5/7] 推理 FP32 + INT8 ONNX...")
    print("  FP32 推理中...")
    out_fp32 = run_onnx_with_all_outputs(ONNX_FP32_ALLNODES, X_test)
    print(f"  FP32 输出 tensor 数: {len(out_fp32)}")

    print("  INT8 推理中...")
    out_int8 = run_onnx_with_all_outputs(int8_allnodes_path, X_test)
    print(f"  INT8 输出 tensor 数: {len(out_int8)}")

    # ---- 6. 对齐 + 计算余弦相似度 ----
    print("\n[Step 6/7] 对齐中间 tensor + 计算余弦相似度...")

    # 获取 OP type 映射
    op_map_fp32 = get_op_type_map(ONNX_FP32_ALLNODES)

    # 找共同的输出名
    common_names = sorted(set(out_fp32.keys()) & set(out_int8.keys()))
    print(f"  共同中间 tensor: {len(common_names)}")

    # 过滤掉 initializer (常量) 和 graph input
    g_fp32 = onnx.load(ONNX_FP32_ALLNODES)
    init_names = {init.name for init in g_fp32.graph.initializer}
    input_names = {inp.name for inp in g_fp32.graph.input}

    results = []
    cosine_values = []
    skipped_reasons = {'init': 0, 'input': 0, 'shape_mismatch': 0, 'nan': 0, 'zero_size': 0}

    for name in common_names:
        # 跳过常量和输入
        if name in init_names:
            skipped_reasons['init'] += 1
            continue
        if name in input_names:
            skipped_reasons['input'] += 1
            continue

        fp32_val = out_fp32[name]
        int8_val = out_int8[name]

        # 检查形状
        if fp32_val.shape != int8_val.shape:
            skipped_reasons['shape_mismatch'] += 1
            continue

        # 跳过标量或零大小
        if fp32_val.size == 0:
            skipped_reasons['zero_size'] += 1
            continue

        cos_val = compute_cosine(fp32_val, int8_val)
        op_type = op_map_fp32.get(name, 'unknown')

        results.append({
            'node_index': len(results),
            'node_name': name,
            'op_type': op_type,
            'cosine_similarity': cos_val,
            'shape': str(tuple(fp32_val.shape)),
        })

        if np.isnan(cos_val):
            skipped_reasons['nan'] += 1
        else:
            cosine_values.append(cos_val)

    # 去重: 同名 tensor 取第一个
    seen = set()
    unique_results = []
    unique_cos = []
    for r in results:
        if r['node_name'] not in seen:
            seen.add(r['node_name'])
            unique_results.append(r)
            if not np.isnan(r['cosine_similarity']):
                unique_cos.append(r['cosine_similarity'])
    results = unique_results
    cosine_values = unique_cos

    # 重新编号
    for i, r in enumerate(results):
        r['node_index'] = i

    print(f"\n  跳过的 tensor:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"    {reason}: {count}")

    # 统计
    cosine_arr = np.array(cosine_values)
    mean_cos = float(np.mean(cosine_arr))
    std_cos  = float(np.std(cosine_arr))
    min_cos  = float(np.min(cosine_arr))
    max_cos  = float(np.max(cosine_arr))
    below_099 = int(np.sum(cosine_arr < 0.99))
    below_095 = int(np.sum(cosine_arr < 0.95))

    # 最差10个
    sorted_idx = np.argsort(cosine_arr)
    worst_10 = [(results[sorted_idx[i]]['node_name'], cosine_arr[sorted_idx[i]])
                for i in range(min(10, len(sorted_idx)))]

    print(f"\n  {'─'*55}")
    print(f"  有效中间 tensor:       {len(cosine_values)}")
    print(f"  平均余弦相似度:         {mean_cos:.6f}")
    print(f"  标准差:                {std_cos:.6f}")
    print(f"  最低余弦相似度:         {min_cos:.6f}")
    print(f"  最高余弦相似度:         {max_cos:.6f}")
    print(f"  Cosine < 0.99 节点:   {below_099}/{len(cosine_values)}")
    print(f"  Cosine < 0.95 节点:   {below_095}/{len(cosine_values)}")
    print(f"\n  最差前10节点:")
    for i, (name, val) in enumerate(worst_10):
        op = op_map_fp32.get(name, '?')
        print(f"    {i+1}. [{op}] {name[:50]}  cos={val:.6f}")

    # ---- 7. 保存文件 ----
    print(f"\n[Step 7/7] 保存结果到 {OUTPUT_DIR}/")

    # --- CSV ---
    csv_path = os.path.join(OUTPUT_DIR, 'onnx_node_cosine_similarity.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['node_index','node_name','op_type',
                                               'cosine_similarity','shape'])
        writer.writeheader()
        writer.writerows(results)
    print(f"  ✓ {csv_path}  ({len(results)} 行)")

    # --- JSON ---
    json_path = os.path.join(OUTPUT_DIR, 'onnx_node_cosine_similarity.json')
    stats = {
        'model': 'AnchorMLP',
        'method': 'ONNX intermediate tensor (FP32 vs INT8)',
        'n_test_samples': N_TEST_SAMPLES,
        'num_nodes_total': len(results),
        'num_nodes_valid': len(cosine_values),
        'mean_cosine': mean_cos,
        'std_cosine': std_cos,
        'min_cosine': min_cos,
        'max_cosine': max_cos,
        'nodes_below_0.99': below_099,
        'nodes_below_0.95': below_095,
        'skipped': skipped_reasons,
        'worst_10_nodes': [{'name': n, 'op_type': op_map_fp32.get(n,'?'), 'cosine': float(v)}
                          for n, v in worst_10],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {json_path}")

    # --- PNG ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    cn_fonts = []
    for fnt in fm.findSystemFonts():
        try:
            fname = fm.FontProperties(fname=fnt).get_name()
            if any(k in fname.lower() for k in ['yahei','simhei','microsoft yahei']):
                cn_fonts.append(fname)
        except: pass
    plt.rcParams['font.sans-serif'] = (cn_fonts + ['DejaVu Sans']) if cn_fonts else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    DPI = 300
    fig, ax = plt.subplots(figsize=(22, 8))

    x = np.arange(len(results))
    y = np.array([r['cosine_similarity'] for r in results])
    valid_mask = ~np.isnan(y)

    x_plot = np.arange(len(results))[valid_mask]
    y_plot = y[valid_mask]

    # 粉色填充
    ax.fill_between(x_plot, y_plot - 0.0005, y_plot + 0.0005,
                    alpha=0.2, color='#E8A0BF')

    # 粉红色曲线
    ax.plot(x_plot, y_plot, '-', color='#D6336C', linewidth=1.5, alpha=0.9,
            label='all', zorder=3)

    # 标记最低点
    worst_idx = x_plot[np.argmin(y_plot)]
    worst_val = y_plot.min()
    ax.annotate(f'{worst_val:.5f}',
                xy=(worst_idx, worst_val),
                xytext=(worst_idx, worst_val - 0.02),
                ha='center', fontsize=10, fontweight='bold', color='#D6336C',
                arrowprops=dict(arrowstyle='->', color='#D6336C', lw=1.5))

    # 0.99 参考线
    ax.axhline(y=0.99, color='gray', linestyle=':', linewidth=1.2, alpha=0.6,
               label='0.99 threshold')

    ax.set_xlabel('Node Index', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('non_quantize_node_accumulate_err_of_node', fontsize=16, fontweight='bold')
    ax.set_ylim(0.90, 1.005)
    ax.set_xlim(-2, len(results) + 2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=12)

    # 右上角统计框
    stats_text = (f'Total Nodes: {len(results)}\n'
                  f'Mean Cos:   {mean_cos:.6f}\n'
                  f'Std Cos:    {std_cos:.6f}\n'
                  f'Min Cos:    {min_cos:.6f}\n'
                  f'Max Cos:    {max_cos:.6f}\n'
                  f'< 0.99:     {below_099}/{len(results)}\n'
                  f'< 0.95:     {below_095}/{len(results)}')
    ax.text(0.985, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85))

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, 'non_quantize_node_accumulate_err_of_node.png')
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {png_path}")

    print(f"\n{'='*70}")
    print(f"  全部完成!")
    print(f"  ONNX 中间 tensor 层分析: {len(results)} 个底层算子节点")
    print(f"  输出目录: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
