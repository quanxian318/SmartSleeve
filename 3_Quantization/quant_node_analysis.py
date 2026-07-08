"""
AnchorMLP INT8 量化误差逐节点累计分析
================================================================================
比较 Float32 vs INT8 动态量化模型在每个中间节点的输出余弦相似度
输出: CSV + JSON + PNG (non_quantize_node_accumulate_err_of_node)
================================================================================
"""
import os, sys, io, csv, json, warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# ===== 用户配置 (按需修改) =====
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results_loso', 'anchor_mlp_2.pt')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'results_loso', 'test_report')
N_TEST_SAMPLES = 1000          # 测试样本数
RANDOM_SEED = 42
EPS = 1e-8

# 测试数据生成参数 (模拟真实特征分布)
FEATURE_MEANS = np.array([161.3, 76.4, 22.0, 177.2, 69.1, 0.5, 45.0,
                          2.0, 0.6, 0.6, 45.0, 1.0, 0.0, 0.0], dtype=np.float32)
FEATURE_STDS  = np.array([96.2, 27.8, 1.3, 8.7, 7.6, 0.5, 37.5,
                          2.0, 0.3, 0.3, 15.0, 0.5, 50.0, 100.0], dtype=np.float32)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 模型定义 (与 train_anchor.py 严格一致)
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
        self.embed = nn.Sequential(nn.Linear(input_dim, embed_dim),
                                   nn.LayerNorm(embed_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim, 2.0, dropout)
                                      for _ in range(n_blocks)])
        self.skip = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Dropout(dropout))
        self.neck = nn.Sequential(nn.Linear(embed_dim + 64, 256),
                                   nn.LayerNorm(256), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(256, 128),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.head_b = nn.Linear(128, 1); self.head_t = nn.Linear(128, 1)
    def forward(self, x):
        feat = self.blocks(self.embed(x))
        sk = self.skip(torch.cat([x[:, 6:7], x[:, :2]], dim=1))
        shared = self.neck(torch.cat([feat, sk], dim=1))
        return torch.cat([self.head_b(shared), self.head_t(shared)], dim=1)


# ============================================================
# 钩子注册 — 采集每个子模块的输出
# ============================================================
def register_output_hooks(model):
    """为模型的所有叶子模块注册 forward hook，返回 (hooks, node_names, outputs_dict)"""
    hooks = []
    node_names = []
    outputs = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            try:
                # 处理各种输出类型
                if isinstance(output, torch.Tensor):
                    outputs[name] = output.detach().clone()
                elif isinstance(output, (tuple, list)):
                    # 提取第一个 tensor
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            outputs[name] = item.detach().clone()
                            return
                    outputs[name] = None  # 找不到 tensor
                else:
                    outputs[name] = None
            except Exception as e:
                outputs[name] = None
        return hook_fn

    for name, module in model.named_modules():
        if name == '':
            continue
        # 跳过纯容器模块(有子模块), 只挂叶子模块
        if len(list(module.children())) > 0:
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))
        node_names.append(name)

    return hooks, node_names, outputs


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ============================================================
# 生成测试数据
# ============================================================
def generate_test_data(n_samples):
    np.random.seed(RANDOM_SEED)
    X = np.random.randn(n_samples, 14).astype(np.float32) * FEATURE_STDS + FEATURE_MEANS
    # 约束合理范围
    X[:, 0] = np.clip(X[:, 0], 10, 500)      # emg90_biceps
    X[:, 1] = np.clip(X[:, 1], 10, 200)      # emg90_triceps
    X[:, 2] = np.clip(X[:, 2], 16, 35)       # BMI
    X[:, 3] = np.clip(X[:, 3], 150, 200)     # height
    X[:, 4] = np.clip(X[:, 4], 40, 120)      # weight
    X[:, 5] = np.clip(X[:, 5], 0, 1)         # gender
    X[:, 6] = np.clip(X[:, 6], 0, 130)       # angle
    X[:, 10] = np.clip(X[:, 10], 0, 130)     # angle_deviation
    return torch.from_numpy(X)


# ============================================================
# 余弦相似度计算
# ============================================================
def compute_cosine_similarity(tensor_fp32, tensor_int8):
    """计算两个 tensor 展平后的余弦相似度"""
    if tensor_fp32 is None or tensor_int8 is None:
        return float('nan')
    a = tensor_fp32.detach().cpu().numpy().ravel().astype(np.float64)
    b = tensor_int8.detach().cpu().numpy().ravel().astype(np.float64)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < EPS or norm_b < EPS:
        return float('nan')
    return float(dot / (norm_a * norm_b))


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  AnchorMLP 量化误差逐节点累计分析")
    print("  Float32 vs INT8 Dynamic Quantization")
    print("=" * 70)

    # ---- 1. 加载体模型 ----
    print("\n[1/6] 加载 Float32 模型...")
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model_fp32 = AnchorMLP()
    model_fp32.load_state_dict(sd)
    model_fp32.eval()
    print(f"  模型: AnchorMLP | 参数: {sum(p.numel() for p in model_fp32.parameters()):,}")

    # ---- 2. INT8 动态量化 ----
    print("\n[2/6] INT8 动态量化...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32.to('cpu'), {nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()
    print("  量化完成 (Linear → qint8)")

    # ---- 3. 生成测试数据 ----
    print(f"\n[3/6] 生成测试数据 ({N_TEST_SAMPLES} 样本)...")
    X_test = generate_test_data(N_TEST_SAMPLES)
    print(f"  shape: {X_test.shape} | dtype: {X_test.dtype}")

    # ---- 4. 注册钩子 + 推理 ----
    print("\n[4/6] 注册 forward hooks + 推理...")
    hooks_fp32, nodes_fp32, out_fp32 = register_output_hooks(model_fp32)
    hooks_int8, nodes_int8, out_int8 = register_output_hooks(model_int8)

    with torch.no_grad():
        _ = model_fp32(X_test)
        _ = model_int8(X_test)

    remove_hooks(hooks_fp32)
    remove_hooks(hooks_int8)

    # 取并集: 两个模型共有的节点
    common_nodes = sorted(set(nodes_fp32) & set(nodes_int8),
                          key=lambda n: nodes_fp32.index(n) if n in nodes_fp32 else 999)
    print(f"  FP32 节点: {len(nodes_fp32)} | INT8 节点: {len(nodes_int8)} | 共有: {len(common_nodes)}")

    # ---- 5. 计算逐节点余弦相似度 ----
    print("\n[5/6] 计算逐节点余弦相似度...")

    results = []  # list of dicts
    cosine_values = []

    for idx, name in enumerate(common_nodes):
        ofp32 = out_fp32.get(name)
        oint8 = out_int8.get(name)

        fp32_shape = tuple(ofp32.shape) if ofp32 is not None and isinstance(ofp32, torch.Tensor) else 'N/A'
        int8_shape = tuple(oint8.shape) if oint8 is not None and isinstance(oint8, torch.Tensor) else 'N/A'

        if ofp32 is None or oint8 is None:
            cos_val = float('nan')
            skip_reason = 'missing output'
        elif not isinstance(ofp32, torch.Tensor) or not isinstance(oint8, torch.Tensor):
            cos_val = float('nan')
            skip_reason = 'non-tensor output'
        else:
            cos_val = compute_cosine_similarity(ofp32, oint8)
            skip_reason = ''

        results.append({
            'node_index': idx,
            'node_name': name,
            'cosine_similarity': cos_val,
            'fp32_shape': str(fp32_shape),
            'int8_shape': str(int8_shape),
        })
        if not np.isnan(cos_val):
            cosine_values.append(cos_val)
        print(f"  [{idx:3d}] {name:<30s}  cos={cos_val:.6f}" +
              (f"  [{skip_reason}]" if skip_reason else ""))

    # 统计
    cosine_arr = np.array(cosine_values)
    mean_cos = float(np.mean(cosine_arr))
    std_cos  = float(np.std(cosine_arr))
    min_cos  = float(np.min(cosine_arr))
    max_cos  = float(np.max(cosine_arr))
    below_099 = int(np.sum(cosine_arr < 0.99))
    below_095 = int(np.sum(cosine_arr < 0.95))

    # 最差10个节点
    sorted_indices = np.argsort(cosine_arr)
    worst_10 = [(common_nodes[i], cosine_arr[i]) for i in sorted_indices[:10]]

    print(f"\n  {'─'*50}")
    print(f"  总节点数:              {len(common_nodes)}")
    print(f"  有效节点(有cos值):     {len(cosine_values)}")
    print(f"  平均余弦相似度:         {mean_cos:.6f}")
    print(f"  标准差:                {std_cos:.6f}")
    print(f"  最低余弦相似度:         {min_cos:.6f}")
    print(f"  最高余弦相似度:         {max_cos:.6f}")
    print(f"  Cosine < 0.99 节点:   {below_099}/{len(common_nodes)}")
    print(f"  Cosine < 0.95 节点:   {below_095}/{len(common_nodes)}")
    print(f"\n  最差前10节点:")
    for i, (name, val) in enumerate(worst_10):
        print(f"    {i+1}. {name:<35s} cos={val:.6f}")

    # ---- 6. 保存文件 ----
    print(f"\n[6/6] 保存结果到 {OUTPUT_DIR}/")

    # --- CSV ---
    csv_path = os.path.join(OUTPUT_DIR, 'quant_node_cosine_similarity.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['node_index','node_name','cosine_similarity',
                                               'fp32_shape','int8_shape'])
        writer.writeheader()
        writer.writerows(results)
    print(f"  ✓ {csv_path}")

    # --- JSON ---
    json_path = os.path.join(OUTPUT_DIR, 'quant_node_cosine_similarity.json')
    stats = {
        'model': 'AnchorMLP',
        'quantization': 'INT8_dynamic',
        'n_test_samples': N_TEST_SAMPLES,
        'num_nodes_total': len(common_nodes),
        'num_nodes_valid': len(cosine_values),
        'mean_cosine': mean_cos,
        'std_cosine': std_cos,
        'min_cosine': min_cos,
        'max_cosine': max_cos,
        'nodes_below_0.99': below_099,
        'nodes_below_0.95': below_095,
        'worst_10_nodes': [{'name': n, 'cosine': float(v)} for n, v in worst_10],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {json_path}")

    # --- PNG ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 中文字体
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
    fig, ax = plt.subplots(figsize=(20, 8))

    x = np.arange(len(results))
    y = np.array([r['cosine_similarity'] for r in results])
    valid_mask = ~np.isnan(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # 粉色填充区域 (min-max range of all samples)
    # 这里只有单值平均值; 用 y 自身做填充 (绕开需要多样本的问题)
    ax.fill_between(x_valid, y_valid - 0.001, y_valid + 0.001,
                    alpha=0.2, color='#E8A0BF')

    # 粉色/洋红色曲线
    ax.plot(x_valid, y_valid, '-', color='#D6336C', linewidth=2.0, alpha=0.95,
            label='all', zorder=3)

    # 标注最低点
    worst_valid_idx = x_valid[np.argmin(y_valid)]
    worst_valid_val = y_valid.min()
    ax.annotate(f'{worst_valid_val:.4f}',
                xy=(worst_valid_idx, worst_valid_val),
                xytext=(worst_valid_idx + 1, worst_valid_val - 0.015),
                ha='left', fontsize=11, fontweight='bold', color='#D6336C',
                arrowprops=dict(arrowstyle='->', color='#D6336C', lw=1.5))

    # 参考线
    ax.axhline(y=0.99, color='gray', linestyle=':', linewidth=1.2, alpha=0.6,
               label='0.99 threshold')

    # 坐标轴
    ax.set_xlabel('Node Index', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('non_quantize_node_accumulate_err_of_node', fontsize=16, fontweight='bold')

    ax.set_ylim(0.90, 1.005)
    ax.set_xlim(-0.5, len(results) - 0.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=12)

    # 统计信息框
    stats_text = (f'Nodes: {len(common_nodes)}\n'
                  f'Mean: {mean_cos:.5f}\n'
                  f'Std:  {std_cos:.5f}\n'
                  f'Min:  {min_cos:.5f}\n'
                  f'Max:  {max_cos:.5f}\n'
                  f'<0.99: {below_099}/{len(common_nodes)}\n'
                  f'<0.95: {below_095}/{len(common_nodes)}')
    ax.text(0.985, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85))

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, 'non_quantize_node_accumulate_err_of_node.png')
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {png_path}")

    print(f"\n{'='*70}")
    print(f"  完成! 输出目录: {OUTPUT_DIR}/")
    print(f"  文件: quant_node_cosine_similarity.csv")
    print(f"        quant_node_cosine_similarity.json")
    print(f"        non_quantize_node_accumulate_err_of_node.png")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
