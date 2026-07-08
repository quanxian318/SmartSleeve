"""
AnchorMLP 底层算子级量化误差分析 (torch.fx 版)
============================================================================
使用 torch.fx 符号追踪 + 中间张量拦截，展开所有底层算子
比较 Float32 vs INT8 量化模型在每个算子输出的余弦相似度
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
# 配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PT_PATH = os.path.join(SCRIPT_DIR, 'results_loso', 'anchor_mlp_2.pt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results_loso', 'test_report')
N_TEST = 500; RANDOM_SEED = 42; EPS = 1e-8
os.makedirs(OUTPUT_DIR, exist_ok=True)

MEANS = np.array([161.3,76.4,22.0,177.2,69.1,0.5,45.0,2.0,0.6,0.6,45.0,1.0,0.0,0.0], dtype=np.float32)
STDS  = np.array([96.2,27.8,1.3,8.7,7.6,0.5,37.5,2.0,0.3,0.3,15.0,0.5,50.0,100.0], dtype=np.float32)

# ============================================================
# 模型定义 (与 train_anchor.py 严格一致)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, expansion=1.5, dropout=0.1):
        super().__init__()
        hidden = int(dim*expansion)
        self.l1=nn.Linear(dim,hidden); self.l2=nn.Linear(hidden,dim)
        self.norm=nn.LayerNorm(dim); self.dropout=nn.Dropout(dropout)
    def forward(self, x):
        r=F.relu(self.l1(x)); r=self.dropout(r); r=self.l2(r)
        return self.norm(x+r)

class AnchorMLP(nn.Module):
    def __init__(self, input_dim=14, embed_dim=256, n_blocks=5, dropout=0.1):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(input_dim,embed_dim),nn.LayerNorm(embed_dim),nn.ReLU())
        self.blocks=nn.Sequential(*[ResidualBlock(embed_dim,2.0,dropout) for _ in range(n_blocks)])
        self.skip=nn.Sequential(nn.Linear(3,64),nn.ReLU(),nn.Dropout(dropout))
        self.neck=nn.Sequential(nn.Linear(embed_dim+64,256),nn.LayerNorm(256),nn.ReLU(),
                                 nn.Dropout(dropout),nn.Linear(256,128),nn.ReLU(),nn.Dropout(dropout))
        self.head_b=nn.Linear(128,1); self.head_t=nn.Linear(128,1)
    def forward(self, x):
        feat=self.blocks(self.embed(x))
        sk=self.skip(torch.cat([x[:,6:7],x[:,:2]],dim=1))
        shared=self.neck(torch.cat([feat,sk],dim=1))
        return torch.cat([self.head_b(shared),self.head_t(shared)],dim=1)


# ============================================================
# 手动逐算子执行 + 记录所有中间输出 (底层算子级)
# ============================================================
class OpTracer:
    """手动执行模型每一层，记录每个底层操作的输出 tensor"""
    def __init__(self):
        self.outputs = {}  # {op_index: (name, tensor)}
        self.idx = 0

    def record(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            self.outputs[self.idx] = (name, tensor.detach().clone())
            self.idx += 1

    def linear(self, name, x, weight, bias):
        # MatMul
        y = torch.mm(x, weight.t())
        self.record(f'{name}.matmul', y)
        # Add bias
        if bias is not None:
            y = y + bias
            self.record(f'{name}.add_bias', y)
        return y

    def layer_norm(self, name, x, weight, bias, eps=1e-5):
        # x - mean
        mean = x.mean(dim=-1, keepdim=True)
        self.record(f'{name}.mean', mean)
        x_centered = x - mean
        self.record(f'{name}.centered', x_centered)
        # var
        var = x_centered.pow(2).mean(dim=-1, keepdim=True)
        self.record(f'{name}.var', var)
        # rstd
        rstd = torch.rsqrt(var + eps)
        self.record(f'{name}.rstd', rstd)
        # normalize
        x_norm = x_centered * rstd
        self.record(f'{name}.normalized', x_norm)
        # scale
        if weight is not None:
            x_norm = x_norm * weight
            self.record(f'{name}.scaled', x_norm)
        if bias is not None:
            x_norm = x_norm + bias
            self.record(f'{name}.shifted', x_norm)
        return x_norm

    def relu(self, name, x):
        y = F.relu(x)
        self.record(f'{name}', y)
        return y

    def dropout(self, name, x, p, training):
        # eval mode: identity
        self.record(f'{name}', x)
        return x


def run_fp32_trace(model, x):
    """手动执行 FP32 模型并记录所有中间 operator 输出"""
    t = OpTracer()
    t.record('input', x)

    # --- embed ---
    w, b = model.embed[0].weight, model.embed[0].bias
    e0 = t.linear('embed.0', x, w, b)
    e1 = t.layer_norm('embed.1', e0, model.embed[1].weight, model.embed[1].bias)
    e2 = t.relu('embed.2', e1)

    # --- blocks ---
    feat = e2
    for i, blk in enumerate(model.blocks):
        # norm
        feat_norm = t.layer_norm(f'blocks.{i}.norm', feat, blk.norm.weight, blk.norm.bias)
        # l1
        h = t.linear(f'blocks.{i}.l1', feat_norm, blk.l1.weight, blk.l1.bias)
        h = t.relu(f'blocks.{i}.l1.relu', h)
        h = t.dropout(f'blocks.{i}.dropout', h, blk.dropout.p, False)
        # l2
        h = t.linear(f'blocks.{i}.l2', h, blk.l2.weight, blk.l2.bias)
        t.record(f'blocks.{i}.residual_add', feat + h)
        # residual + norm
        feat = t.layer_norm(f'blocks.{i}.final_norm', feat + h, blk.norm.weight, blk.norm.bias)
        t.record(f'blocks.{i}.output', feat)

    # --- skip ---
    sk_cat = torch.cat([x[:, 6:7], x[:, :2]], dim=1)
    t.record('skip.cat', sk_cat)
    sk0 = t.linear('skip.0', sk_cat, model.skip[0].weight, model.skip[0].bias)
    sk1 = t.relu('skip.1', sk0)
    sk2 = t.dropout('skip.2', sk1, model.skip[2].p, False)
    t.record('skip.output', sk2)

    # --- neck ---
    neck_cat = torch.cat([feat, sk2], dim=1)
    t.record('neck.cat', neck_cat)
    n0 = t.linear('neck.0', neck_cat, model.neck[0].weight, model.neck[0].bias)
    n1 = t.layer_norm('neck.1', n0, model.neck[1].weight, model.neck[1].bias)
    n2 = t.relu('neck.2', n1)
    n3 = t.dropout('neck.3', n2, model.neck[3].p, False)
    n4 = t.linear('neck.4', n3, model.neck[4].weight, model.neck[4].bias)
    n5 = t.relu('neck.5', n4)
    n6 = t.dropout('neck.6', n5, model.neck[6].p, False)

    # --- heads ---
    hb = t.linear('head_b', n6, model.head_b.weight, model.head_b.bias)
    ht = t.linear('head_t', n6, model.head_t.weight, model.head_t.bias)
    t.record('output', torch.cat([hb, ht], dim=1))

    return t.outputs


# ============================================================
# INT8 量化版本的 tracer (使用量化后的权重)
# ============================================================
def quantize_weight_int8(w):
    """将 float32 权重量化为 int8 (模拟)"""
    w_fp = w.detach().cpu().float()
    # Per-channel symmetric quantization
    scale = w_fp.abs().max(dim=1, keepdim=True)[0] / 127.0
    scale = torch.clamp(scale, min=1e-8)
    w_q = torch.round(w_fp / scale).clamp(-127, 127)
    w_deq = w_q * scale  # 反量化回 float
    return w_deq.to(w.device)

def quantize_layer_norm_int8(module):
    """对 LayerNorm 做 INT8 量化模拟"""
    # LayerNorm weight 和 bias 也可以量化
    w_q = quantize_weight_int8(module.weight.unsqueeze(1)).squeeze(1) if module.weight is not None else None
    b_q = module.bias  # bias 不量化, 保持 float
    return w_q, b_q


def run_int8_trace(model, x):
    """执行 INT8 量化模拟的模型, 记录所有中间 operator 输出"""
    t = OpTracer()
    t.record('input', x)

    # --- embed ---
    w_q = quantize_weight_int8(model.embed[0].weight)
    b = model.embed[0].bias  # bias 不量化
    e0 = t.linear('embed.0', x, w_q, b)
    ln_w_q, ln_b = quantize_layer_norm_int8(model.embed[1])
    e1 = t.layer_norm('embed.1', e0, ln_w_q, ln_b if ln_b is not None else model.embed[1].bias)
    e2 = t.relu('embed.2', e1)

    # --- blocks ---
    feat = e2
    for i, blk in enumerate(model.blocks):
        ln_w_q, ln_b_q = quantize_layer_norm_int8(blk.norm)
        feat_norm = t.layer_norm(f'blocks.{i}.norm', feat,
                                  ln_w_q, ln_b_q if ln_b_q is not None else blk.norm.bias)
        l1_w_q = quantize_weight_int8(blk.l1.weight)
        h = t.linear(f'blocks.{i}.l1', feat_norm, l1_w_q, blk.l1.bias)
        h = t.relu(f'blocks.{i}.l1.relu', h)
        h = t.dropout(f'blocks.{i}.dropout', h, blk.dropout.p, False)
        l2_w_q = quantize_weight_int8(blk.l2.weight)
        h = t.linear(f'blocks.{i}.l2', h, l2_w_q, blk.l2.bias)
        t.record(f'blocks.{i}.residual_add', feat + h)
        feat = t.layer_norm(f'blocks.{i}.final_norm', feat + h,
                             ln_w_q, ln_b_q if ln_b_q is not None else blk.norm.bias)
        t.record(f'blocks.{i}.output', feat)

    # --- skip ---
    sk_cat = torch.cat([x[:,6:7], x[:,:2]], dim=1)
    t.record('skip.cat', sk_cat)
    sk_w_q = quantize_weight_int8(model.skip[0].weight)
    sk0 = t.linear('skip.0', sk_cat, sk_w_q, model.skip[0].bias)
    sk1 = t.relu('skip.1', sk0)
    sk2 = t.dropout('skip.2', sk1, model.skip[2].p, False)
    t.record('skip.output', sk2)

    # --- neck ---
    neck_cat = torch.cat([feat, sk2], dim=1)
    t.record('neck.cat', neck_cat)
    n0_w_q = quantize_weight_int8(model.neck[0].weight)
    n0 = t.linear('neck.0', neck_cat, n0_w_q, model.neck[0].bias)
    ln_w, ln_b = quantize_layer_norm_int8(model.neck[1])
    n1 = t.layer_norm('neck.1', n0, ln_w, ln_b if ln_b is not None else model.neck[1].bias)
    n2 = t.relu('neck.2', n1)
    n3 = t.dropout('neck.3', n2, model.neck[3].p, False)
    n4_w_q = quantize_weight_int8(model.neck[4].weight)
    n4 = t.linear('neck.4', n3, n4_w_q, model.neck[4].bias)
    n5 = t.relu('neck.5', n4)
    n6 = t.dropout('neck.6', n5, model.neck[6].p, False)

    # --- heads ---
    hb_w_q = quantize_weight_int8(model.head_b.weight)
    ht_w_q = quantize_weight_int8(model.head_t.weight)
    hb = t.linear('head_b', n6, hb_w_q, model.head_b.bias)
    ht = t.linear('head_t', n6, ht_w_q, model.head_t.bias)
    t.record('output', torch.cat([hb, ht], dim=1))

    return t.outputs


# ============================================================
# 生成测试数据
# ============================================================
def gen_data(n):
    np.random.seed(RANDOM_SEED)
    X = np.random.randn(n,14).astype(np.float32)*STDS+MEANS
    X[:,0]=np.clip(X[:,0],10,500); X[:,1]=np.clip(X[:,1],10,200)
    X[:,2]=np.clip(X[:,2],16,35); X[:,3]=np.clip(X[:,3],150,200)
    X[:,4]=np.clip(X[:,4],40,120); X[:,5]=np.clip(X[:,5],0,1)
    X[:,6]=np.clip(X[:,6],0,130); X[:,10]=np.clip(X[:,10],0,130)
    return torch.from_numpy(X)


def cosine(a, b):
    a = a.ravel().numpy().astype(np.float64)
    b = b.ravel().numpy().astype(np.float64)
    if len(a)==0: return float('nan')
    dot=np.dot(a,b); na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<EPS or nb<EPS: return float('nan')
    return float(dot/(na*nb))


# ============================================================
# 主流程
# ============================================================
def main():
    print("="*65)
    print("  AnchorMLP INT8 量化误差 — 底层算子级分析")
    print("  Float32 vs INT8 (per-channel weight quantization)")
    print("="*65)

    # 1. 加载模型
    print("\n[1] 加载模型...")
    ckpt = torch.load(MODEL_PT_PATH, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.',''): v for k,v in ckpt['model_state_dict'].items()}
    model = AnchorMLP(); model.load_state_dict(sd); model.eval()
    print(f"  AnchorMLP | {sum(p.numel() for p in model.parameters()):,} params")

    # 2. 生成数据
    print(f"\n[2] 测试数据 ({N_TEST} samples)...")
    X = gen_data(N_TEST)
    print(f"  shape: {X.shape}")

    # 3. 运行两个 tracer
    print("\n[3] 运行 FP32 + INT8 tracer...")
    with torch.no_grad():
        out_fp32 = run_fp32_trace(model, X)
        out_int8 = run_int8_trace(model, X)

    print(f"  FP32 算子输出: {len(out_fp32)}")
    print(f"  INT8 算子输出: {len(out_int8)}")

    # 4. 对齐 + 计算余弦相似度
    print("\n[4] 计算逐算子余弦相似度...")

    # 按 key (op index) 对齐
    common_keys = sorted(set(out_fp32.keys()) & set(out_int8.keys()))
    print(f"  共同算子: {len(common_keys)}")

    results = []
    cos_vals = []
    for k in common_keys:
        name_fp, t_fp = out_fp32[k]
        name_int, t_int = out_int8[k]
        # 确保形状一致
        if t_fp.shape != t_int.shape:
            # 尝试 reshape
            continue
        c = cosine(t_fp, t_int)
        # 推断 op_type
        if 'matmul' in name_fp: op='MatMul'
        elif 'add_bias' in name_fp: op='Add'
        elif 'mean' in name_fp: op='ReduceMean'
        elif 'centered' in name_fp: op='Sub'
        elif 'var' in name_fp: op='Mul+ReduceMean'
        elif 'rstd' in name_fp: op='Rsqrt'
        elif 'normalized' in name_fp: op='Mul'
        elif 'scaled' in name_fp: op='Mul'
        elif 'shifted' in name_fp: op='Add'
        elif 'residual_add' in name_fp: op='Add'
        elif 'relu' in name_fp.lower(): op='Relu'
        elif 'dropout' in name_fp.lower(): op='Identity'
        elif 'cat' in name_fp.lower(): op='Concat'
        elif 'output' in name_fp.lower() or 'input' in name_fp.lower(): op='Identity'
        elif 'norm' in name_fp.lower(): op='LayerNorm'
        else: op='Other'

        results.append({
            'node_index': len(results),
            'node_name': name_fp,
            'op_type': op,
            'cosine_similarity': c,
            'shape': str(tuple(t_fp.shape)),
        })
        if not np.isnan(c):
            cos_vals.append(c)

    cos_arr = np.array(cos_vals)
    mean_c=float(np.mean(cos_arr)); std_c=float(np.std(cos_arr))
    min_c=float(np.min(cos_arr)); max_c=float(np.max(cos_arr))
    below_099=int(np.sum(cos_arr<0.99)); below_095=int(np.sum(cos_arr<0.95))

    # 最差10个
    si=np.argsort(cos_arr)
    worst_10=[(results[si[i]]['node_name'],cos_arr[si[i]]) for i in range(min(10,len(si)))]

    print(f"\n  {'─'*55}")
    print(f"  有效算子节点:          {len(cos_vals)}")
    print(f"  平均余弦相似度:         {mean_c:.6f}")
    print(f"  标准差:                {std_c:.6f}")
    print(f"  最低余弦相似度:         {min_c:.6f}")
    print(f"  最高余弦相似度:         {max_c:.6f}")
    print(f"  Cosine < 0.99:        {below_099}/{len(cos_vals)}")
    print(f"  Cosine < 0.95:        {below_095}/{len(cos_vals)}")
    print(f"\n  最差前10节点:")
    for i,(n,v) in enumerate(worst_10):
        op_t = next((r['op_type'] for r in results if r['node_name']==n), '?')
        print(f"    {i+1}. [{op_t}] {n:<40s} cos={v:.6f}")

    # 5. 保存
    print(f"\n[5] 保存结果...")

    csv_path=os.path.join(OUTPUT_DIR,'onnx_node_cosine_similarity.csv')
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['node_index','node_name','op_type','cosine_similarity','shape'])
        w.writeheader(); w.writerows(results)
    print(f"  ✓ {csv_path} ({len(results)} 行)")

    json_path=os.path.join(OUTPUT_DIR,'onnx_node_cosine_similarity.json')
    with open(json_path,'w',encoding='utf-8') as f:
        json.dump({
            'model':'AnchorMLP','method':'Manual per-operator trace (FP32 vs INT8 weight quant)',
            'n_test_samples':N_TEST,'num_nodes':len(results),'num_valid':len(cos_vals),
            'mean_cosine':mean_c,'std_cosine':std_c,'min_cosine':min_c,'max_cosine':max_c,
            'nodes_below_0.99':below_099,'nodes_below_0.95':below_095,
            'worst_10':[{'name':n,'cosine':float(v)} for n,v in worst_10],
        },f,indent=2,ensure_ascii=False)
    print(f"  ✓ {json_path}")

    # 6. 图
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    cn=[];
    for fnt in fm.findSystemFonts():
        try:
            n=fm.FontProperties(fname=fnt).get_name()
            if any(k in n.lower() for k in ['yahei','simhei','microsoft yahei']): cn.append(n)
        except: pass
    plt.rcParams['font.sans-serif']=(cn+['DejaVu Sans']) if cn else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False

    DPI=300
    fig,ax=plt.subplots(figsize=(24,8))
    ax.set_facecolor('white')
    y=np.array([r['cosine_similarity'] for r in results])
    vm=~np.isnan(y); xp=np.arange(len(results))[vm]; yp=y[vm]
    ax.plot(xp,yp,'-',color='#D6336C',linewidth=1.6,alpha=0.95,label='all',zorder=3)

    wi=xp[np.argmin(yp)]; wv=yp.min()
    ax.annotate(f'{wv:.4f}',xy=(wi,wv),xytext=(wi+3,wv-0.02),
                ha='left',fontsize=10,fontweight='bold',color='#D6336C',
                arrowprops=dict(arrowstyle='->',color='#D6336C',lw=1.5))
    ax.axhline(y=0.99,color='gray',linestyle=':',linewidth=1.2,alpha=0.6,label='0.99 threshold')
    ax.set_xlabel('Node Index',fontsize=14); ax.set_ylabel('Cosine Similarity',fontsize=14)
    ax.set_title('non_quantize_node_accumulate_err_of_node',fontsize=16,fontweight='bold')
    ax.set_ylim(0.90, 1.005); ax.set_xlim(-2,len(results)+2)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.grid(True,alpha=0.3,linestyle='--'); ax.legend(loc='lower left',fontsize=12)
    stxt=(f'Total Nodes: {len(results)}\nMean Cos:   {mean_c:.6f}\nStd Cos:    {std_c:.6f}\n'
          f'Min Cos:    {min_c:.6f}\nMax Cos:    {max_c:.6f}\n'
          f'< 0.99:     {below_099}/{len(results)}\n< 0.95:     {below_095}/{len(results)}')
    ax.text(0.985,0.97,stxt,transform=ax.transAxes,fontsize=9,fontfamily='monospace',
            va='top',ha='right',bbox=dict(boxstyle='round,pad=0.5',facecolor='lightyellow',alpha=0.85))
    fig.tight_layout()
    png_path=os.path.join(OUTPUT_DIR,'non_quantize_node_accumulate_err_of_node.png')
    fig.savefig(png_path,dpi=DPI,bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ {png_path}")

    # 6b. 局部放大图 (Y轴 0.9994-1.00002, 展示细微波动)
    fig2,ax2=plt.subplots(figsize=(24,8))
    ax2.set_facecolor('white')
    ax2.plot(xp,yp,'-',color='#D6336C',linewidth=1.6,alpha=0.95,label='all',zorder=3)
    ax2.annotate(f'{wv:.6f}',xy=(wi,wv),xytext=(wi+10,wv-0.00015),
                ha='left',fontsize=10,fontweight='bold',color='#D6336C',
                arrowprops=dict(arrowstyle='->',color='#D6336C',lw=1.5))
    ax2.axhline(y=mean_c,color='gray',linestyle=':',linewidth=1.2,alpha=0.6,
               label=f'Mean ({mean_c:.6f})')
    ax2.set_xlabel('Node Index',fontsize=14); ax2.set_ylabel('Cosine Similarity',fontsize=14)
    ax2.set_title('non_quantize_node_accumulate_err_of_node (Zoomed)',fontsize=16,fontweight='bold')
    ax2.set_ylim(0.9994, 1.00002); ax2.set_xlim(-2,len(results)+2)
    ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax2.grid(True,alpha=0.3,linestyle='--'); ax2.legend(loc='lower left',fontsize=12)
    ztxt=(f'Total Nodes: {len(results)}\nMean Cos:   {mean_c:.6f}\nStd Cos:    {std_c:.6f}\n'
          f'Min Cos:    {min_c:.6f}\nMax Cos:    {max_c:.6f}\n'
          f'< 0.99:     {below_099}/{len(results)}\n< 0.95:     {below_095}/{len(results)}')
    ax2.text(0.985,0.97,ztxt,transform=ax2.transAxes,fontsize=9,fontfamily='monospace',
            va='top',ha='right',bbox=dict(boxstyle='round,pad=0.5',facecolor='lightyellow',alpha=0.85))
    fig2.tight_layout()
    zoom_path=os.path.join(OUTPUT_DIR,'non_quantize_node_accumulate_err_of_node_zoomed.png')
    fig2.savefig(zoom_path,dpi=DPI,bbox_inches='tight'); plt.close(fig2)
    print(f"  ✓ {zoom_path} (局部放大)")

    print(f"\n{'='*65}")
    print(f"  完成! {len(results)} 个底层算子节点已分析")
    print(f"  输出: {OUTPUT_DIR}/")
    print(f"{'='*65}")

if __name__=='__main__':
    main()
