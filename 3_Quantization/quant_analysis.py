"""
AnchorMLP 量化精度分析 — 逐节点余弦相似度
比较 float32 vs INT8 量化模型在各层输出的相似度
"""
import os, sys, io, warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results_loso', 'test_report')
DPI = 300

# 字体
cn_fonts = []
for f in fm.findSystemFonts():
    try:
        name = fm.FontProperties(fname=f).get_name()
        if any(k in name.lower() for k in ['yahei','simhei','microsoft yahei','noto sans cjk']):
            cn_fonts.append(name)
    except: pass
plt.rcParams['font.sans-serif'] = (cn_fonts + ['DejaVu Sans']) if cn_fonts else ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    'axes.labelsize': 13, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'legend.framealpha': 0.9, 'figure.facecolor': 'white',
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

# ============================================================
# 模型定义 (与 train_anchor.py 完全一致)
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
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim,2.0,dropout) for _ in range(n_blocks)])
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
# 钩子注册 — 捕获每层输出
# ============================================================
def register_all_hooks(model):
    """注册钩子，捕获每个子模块的输出"""
    hooks = []
    node_names = []

    def make_hook(name):
        def hook_fn(module, input, output):
            pass  # 由 forward hook 处理
        return hook_fn

    # Pre-forward hooks (获取输入)
    fwd_hooks = []
    fwd_outputs = {}

    def make_fwd_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                fwd_outputs[name] = output.detach().clone()
        return hook_fn

    # 遍历所有命名模块
    for name, module in model.named_modules():
        if name == '' or len(list(module.children())) > 0:
            continue  # 跳过容器模块
        fwd_hooks.append(module.register_forward_hook(make_fwd_hook(name)))
        node_names.append(name)

    return fwd_hooks, node_names, fwd_outputs


def get_node_outputs(model, x):
    """手动逐层运行，捕获每层输出"""
    outputs = {}

    # embed
    e0 = model.embed[0](x)      # Linear
    outputs['embed.0'] = e0.clone()
    e1 = model.embed[1](e0)     # LayerNorm
    outputs['embed.1'] = e1.clone()
    e2 = model.embed[2](e1)     # ReLU
    outputs['embed.2'] = e2.clone()

    # blocks
    feat = e2
    for i, blk in enumerate(model.blocks):
        # norm
        h = blk.norm(feat)
        outputs[f'blocks.{i}.norm'] = h.clone()
        # l1
        h = F.relu(blk.l1(h))
        outputs[f'blocks.{i}.l1'] = h.clone()
        # dropout + l2
        h = blk.dropout(h)
        h = blk.l2(h)
        outputs[f'blocks.{i}.l2'] = h.clone()
        # residual
        feat = blk.norm(feat + h)  # 注意：原forward中的norm(x+r)，即对加完后做norm
        # 修正：正确的forward是 norm(x + l2(drop(relu(l1(norm(x))))))
        # 上面我们分开算了，现在重新算feat
    # 重新正确计算blocks
    feat = e2
    for i, blk in enumerate(model.blocks):
        feat = blk(feat)
        outputs[f'blocks.{i}'] = feat.clone()

    # skip
    sk_input = torch.cat([x[:,6:7], x[:,:2]], dim=1)
    outputs['skip.input'] = sk_input.clone()
    s0 = model.skip[0](sk_input)
    outputs['skip.0'] = s0.clone()
    s1 = model.skip[1](s0)
    outputs['skip.1'] = s1.clone()
    s2 = model.skip[2](s1)
    outputs['skip.2'] = s2.clone()

    # neck
    cat = torch.cat([feat, s2], dim=1)
    outputs['neck.cat'] = cat.clone()
    n0 = model.neck[0](cat)
    outputs['neck.0'] = n0.clone()
    n1 = model.neck[1](n0)
    outputs['neck.1'] = n1.clone()
    n2 = model.neck[2](n1)
    outputs['neck.2'] = n2.clone()
    n3 = model.neck[3](n2)
    outputs['neck.3'] = n3.clone()
    n4 = model.neck[4](n3)
    outputs['neck.4'] = n4.clone()
    n5 = model.neck[5](n4)
    outputs['neck.5'] = n5.clone()
    n6 = model.neck[6](n5)
    outputs['neck.6'] = n6.clone()

    # heads
    hb = model.head_b(n6)
    outputs['head_b'] = hb.clone()
    ht = model.head_t(n6)
    outputs['head_t'] = ht.clone()

    return outputs


# ============================================================
# INT8 量化
# ============================================================
def quantize_to_int8(model, calibration_data):
    """动态量化模型为 INT8"""
    model.eval()
    # 使用 PyTorch 动态量化 (适用于 Linear 层)
    # 先准备量化模型
    model_fp32 = model.cpu()
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},  # 只量化 Linear 层
        dtype=torch.qint8
    )
    return model_int8


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  AnchorMLP 量化精度分析 — 逐节点余弦相似度")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1] 加载模型...")
    model_dir = os.path.join(SCRIPT_DIR, 'results_loso')
    ckpt = torch.load(os.path.join(model_dir, 'anchor_mlp_2.pt'), map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.',''): v for k,v in ckpt['model_state_dict'].items()}
    model_fp32 = AnchorMLP()
    model_fp32.load_state_dict(sd)
    model_fp32.eval()
    print("  Float32 模型加载完成")

    # 2. INT8 量化
    print("\n[2] INT8 动态量化...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    print("  INT8 量化完成")

    # 3. 生成测试数据
    print("\n[3] 生成测试数据...")
    np.random.seed(42)
    # 模拟真实的14维特征分布 (来自S1-S10的统计)
    # mean/std 从 scaler 近似
    means = np.array([161.3, 76.4, 22.0, 177.2, 69.1, 0.5, 45.0,
                      2.0, 0.6, 0.6, 45.0, 1.0, 0.0, 0.0])
    stds = np.array([96.2, 27.8, 1.3, 8.7, 7.6, 0.5, 37.5,
                     2.0, 0.3, 0.3, 15.0, 0.5, 50.0, 100.0])
    n_samples = 500
    X_test = np.random.randn(n_samples, 14) * stds + means
    X_test = X_test.astype(np.float32)

    # 约束合理范围
    X_test[:,0] = np.clip(X_test[:,0], 10, 500)    # emg90_b
    X_test[:,1] = np.clip(X_test[:,1], 10, 200)    # emg90_t
    X_test[:,2] = np.clip(X_test[:,2], 16, 35)     # BMI
    X_test[:,3] = np.clip(X_test[:,3], 150, 200)   # height
    X_test[:,4] = np.clip(X_test[:,4], 40, 120)    # weight
    X_test[:,5] = np.clip(X_test[:,5], 0, 1)       # gender
    X_test[:,6] = np.clip(X_test[:,6], 0, 130)     # angle
    X_test[:,10] = np.clip(X_test[:,10], 0, 130)   # angle_deviation

    X_tensor = torch.from_numpy(X_test)
    print(f"  测试样本: {n_samples}")

    # 4. 获取两组输出
    print("\n[4] 计算逐节点输出...")

    with torch.no_grad():
        # Float32 输出
        outputs_fp32 = get_node_outputs(model_fp32, X_tensor)
        # INT8 输出
        outputs_int8 = get_node_outputs(model_int8, X_tensor)

    print(f"  节点数: {len(outputs_fp32)}")

    # 5. 计算每个节点的余弦相似度
    print("\n[5] 计算余弦相似度...")
    node_names = list(outputs_fp32.keys())
    cosine_scores = []
    cosine_all_samples = []

    for name in node_names:
        fp32_out = outputs_fp32[name].numpy()
        int8_out = outputs_int8[name].numpy()

        # 每样本余弦相似度
        eps = 1e-8
        dot = np.sum(fp32_out * int8_out, axis=1)
        norm_fp32 = np.linalg.norm(fp32_out, axis=1) + eps
        norm_int8 = np.linalg.norm(int8_out, axis=1) + eps
        per_sample_cos = dot / (norm_fp32 * norm_int8)

        cosine_all_samples.append(per_sample_cos)
        cosine_scores.append(np.mean(per_sample_cos))

        print(f"  {name:<22} cos={cosine_scores[-1]:.6f}  min={per_sample_cos.min():.4f}")

    cosine_scores = np.array(cosine_scores)
    cosine_all_samples = np.array(cosine_all_samples)  # (n_nodes, n_samples)

    # 6. 绘图
    print("\n[6] 生成量化精度分析图...")

    fig, ax = plt.subplots(figsize=(18, 7))

    x = np.arange(len(node_names))
    x_labels = [n.replace('blocks.','B').replace('neck.','N').replace('skip.','S')
                 .replace('embed.','E').replace('head_','H') for n in node_names]

    # 所有节点的所有样本余弦相似度 (粉色填充区域)
    mean_cos = cosine_all_samples.mean(axis=1)
    min_cos = cosine_all_samples.min(axis=1)
    max_cos = cosine_all_samples.max(axis=1)

    # 粉色填充: min-max range
    ax.fill_between(x, min_cos, max_cos, alpha=0.25, color='#E8A0BF',
                    label=f'Range (min={min_cos.min():.4f})')

    # 粉色曲线: all samples mean
    ax.plot(x, mean_cos, '-', color='#D6336C', linewidth=2.5, alpha=0.9,
            label=f'Mean Cosine ({mean_cos.mean():.4f})', zorder=3)

    # 标注最低点
    worst_idx = np.argmin(mean_cos)
    ax.annotate(f'{mean_cos[worst_idx]:.4f}',
                xy=(worst_idx, mean_cos[worst_idx]),
                xytext=(worst_idx, mean_cos[worst_idx] - 0.04),
                ha='center', fontsize=11, fontweight='bold', color='#D6336C',
                arrowprops=dict(arrowstyle='->', color='#D6336C', lw=1.5))

    # 0.99 参考线
    ax.axhline(y=0.99, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='0.99 threshold')

    # 坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_xlabel('Node Index (Layer Output)', fontsize=14)
    ax.set_title('AnchorMLP Quantization Accuracy — Per-Node Cosine Similarity\n'
                 '(Float32 vs INT8 Dynamic Quantization)', fontsize=16, fontweight='bold')

    ax.set_ylim(0.85, 1.02)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加统计信息文本框
    stats_text = (f'Nodes: {len(node_names)}\n'
                  f'Mean Cosine: {mean_cos.mean():.5f}\n'
                  f'Std Cosine:  {mean_cos.std():.5f}\n'
                  f'Min Cosine:  {mean_cos.min():.5f}\n'
                  f'Nodes < 0.99: {(mean_cos < 0.99).sum()}/{len(node_names)}')
    ax.text(0.98, 0.92, stats_text, transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    save_path = f'{OUTPUT_DIR}/fig_17_quantization_accuracy.png'
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")

    # 7. 汇总表
    print("\n[7] 汇总")
    print(f"  总节点数:       {len(node_names)}")
    print(f"  平均余弦相似度:  {mean_cos.mean():.5f}")
    print(f"  最差节点:        {node_names[worst_idx]} ({mean_cos[worst_idx]:.5f})")
    print(f"  低于0.99的节点:  {(mean_cos < 0.99).sum()}/{len(node_names)}")
    print(f"  低于0.95的节点:  {(mean_cos < 0.95).sum()}/{len(node_names)}")

    # 结论
    if mean_cos.min() > 0.95:
        print("\n  结论: INT8量化精度损失极小，适合端侧部署。")
    elif mean_cos.min() > 0.90:
        print("\n  结论: INT8量化存在轻微精度损失，大部分节点可接受。")
    else:
        print("\n  结论: 部分节点量化误差较大，建议使用混合精度量化。")

    print("\nDone!")


if __name__ == '__main__':
    main()
