"""
AnchorMLP 专业测试图生成 — model_2
================================================================================
生成 16 张学术期刊风格测试图 + JSON 指标 + TXT 报告
基于 train_anchor.py 已验证的数据管线
用法: python gen_figures.py
================================================================================
"""
import os, sys, io, json, time, warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import convolve1d
from scipy.stats import pearsonr, spearmanr, gaussian_kde, probplot, norm
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results_loso', 'test_report')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS, DOWNSAMPLE = 2000, 20
CH_BICEPS, CH_TRICEPS = 10, 11
N_MUSCLES, N_FEATURES = 2, 14
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DPI = 300

COLOR_B = '#1F77B4'
COLOR_T = '#D62728'
MUSCLE_NAMES = ['Biceps', 'Triceps']
MUSCLE_CN = ['肱二头肌', '肱三头肌']
COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F']

# 中文字体
cn_fonts = []
for f in fm.findSystemFonts():
    try:
        name = fm.FontProperties(fname=f).get_name()
        if any(k in name.lower() for k in ['yahei','simhei','microsoft yahei','noto sans cjk']):
            cn_fonts.append(name)
    except: pass
if cn_fonts:
    plt.rcParams['font.sans-serif'] = cn_fonts + ['DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    'axes.labelsize': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
    'legend.framealpha': 0.9, 'figure.facecolor': 'white',
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

print(f"[设备] {DEVICE} | [字体] {cn_fonts[0] if cn_fonts else 'English'} | [输出] {OUTPUT_DIR}")

# ============================================================
# 模型 (与 train_anchor.py 一致)
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
# 信号处理
# ============================================================
def extract_emg_envelope(emg_signal, fs=2000):
    b_n, a_n = iirnotch(50, 30, fs); sig = filtfilt(b_n, a_n, emg_signal, axis=0)
    b_bp, a_bp = butter(4, [20/(0.5*fs), 450/(0.5*fs)], btype='band'); sig = filtfilt(b_bp, a_bp, sig, axis=0)
    sig = np.abs(sig); b_lp, a_lp = butter(4, 6/(0.5*fs), btype='low')
    return filtfilt(b_lp, a_lp, sig, axis=0)

def create_gaussian_kernel(size, sigma):
    x = np.linspace(-size//2, size//2, size); kernel = np.exp(-0.5*(x/sigma)**2)
    return kernel/kernel.sum()

def find_data_matrix(obj):
    if isinstance(obj, np.ndarray) and obj.ndim==2 and obj.shape[0]>100 and obj.dtype!='O': return obj
    if isinstance(obj, np.ndarray) and obj.dtype=='O':
        for item in obj.flat:
            r = find_data_matrix(item)
            if r is not None: return r
    return None

def extract_angle(mat, fs=2000):
    if 'angle' in mat:
        d = find_data_matrix(mat['angle'])
        if d is not None:
            a = d.flatten().astype(np.float64)
            if a.max()>180: a = (a-a.min())/(a.max()-a.min()+1e-9)*130.0
            return np.clip(a,0,130),'real_sensor'
    if 'inclin' in mat:
        d = find_data_matrix(mat['inclin'])
        if d is not None:
            a = d.flatten().astype(np.float64); emg_raw = find_data_matrix(mat['emg'])
            b_env = extract_emg_envelope(emg_raw[:,CH_BICEPS],fs)
            corr = np.corrcoef(a,b_env)[0,1]
            if not np.isnan(corr) and corr<0: a = -a
            a = (a-a.min())/(a.max()-a.min()+1e-9)*130.0; return a,'inclinometer'
    if 'restimulus' in mat:
        restim = find_data_matrix(mat['restimulus']).flatten()
        kernel = create_gaussian_kernel(3000,700)
        smooth = convolve1d((restim>0).astype(float),kernel)
        a = (smooth/(smooth.max()+1e-9))*130.0; return a,'restimulus_derived'
    raise ValueError("No angle data")

def extract_anchor_emg(angle, emg_envelopes, angular_vel, anchor_angle=90.0,
                       tolerance=2.0, vel_threshold=30.0):
    eps = 1e-8
    mask_strict = ((angle>=anchor_angle-tolerance)&(angle<=anchor_angle+tolerance)&(np.abs(angular_vel)<vel_threshold))
    n_strict = mask_strict.sum()
    if n_strict>=10: anchor=np.median(emg_envelopes[mask_strict],axis=0); method='median_strict'
    elif n_strict>=3: anchor=np.mean(emg_envelopes[mask_strict],axis=0); method='mean_strict'
    else:
        mask_loose = (angle>=anchor_angle-tolerance)&(angle<=anchor_angle+tolerance)
        if mask_loose.sum()>=3: anchor=np.mean(emg_envelopes[mask_loose],axis=0); method='mean_loose'
        else:
            anchor=np.zeros(N_MUSCLES)
            for m in range(N_MUSCLES):
                si=np.argsort(angle); sa,se=angle[si],emg_envelopes[si,m]; ua,ue=[],[]
                for a in np.unique(np.round(sa,1)):
                    ma=np.abs(sa-a)<0.5; ua.append(a); ue.append(np.mean(se[ma]))
                ua,ue=np.array(ua),np.array(ue)
                if ua[0]<=anchor_angle<=ua[-1]: anchor[m]=np.interp(anchor_angle,ua,ue)
                else: anchor[m]=ue[np.argmin(np.abs(ua-anchor_angle))]
            method='interpolated'
    noise_floor_t=np.percentile(emg_envelopes[:,1],5); min_anchor_t=noise_floor_t*2.0
    if anchor[1]<min_anchor_t: anchor[1]=min_anchor_t; method+='_triceps_clamped'
    return anchor, method

def build_features(mat, fs=2000):
    eps, dt = 1e-8, 1.0/fs
    gender = 1 if 'f' in str(mat.get('gender','m')).lower() else 0
    weight = float(mat['weight'][0][0]) if 'weight' in mat else 70.0
    height = float(mat['height'][0][0]) if 'height' in mat else 175.0
    bmi = weight/((height/100)**2)
    emg_raw = find_data_matrix(mat['emg'])
    envelopes = np.column_stack([extract_emg_envelope(emg_raw[:,CH_BICEPS],fs),
                                  extract_emg_envelope(emg_raw[:,CH_TRICEPS],fs)])
    angle, angle_source = extract_angle(mat, fs)
    raw_vel = np.gradient(angle, dt)
    anchor_emg, anchor_method = extract_anchor_emg(angle, envelopes, raw_vel, 90.0)
    emg90_b, emg90_t = anchor_emg[0], anchor_emg[1]
    emg90_ratio = emg90_b/(emg90_t+eps)
    rad = angle/130.0*np.pi
    sin_a, cos_a = np.sin(rad), np.cos(rad)
    angle_dev = np.abs(angle-90.0); angle_rel = angle/90.0
    b_lp,a_lp = butter(2,15/(0.5*fs),btype='low'); angular_vel = filtfilt(b_lp,a_lp,raw_vel)
    angular_acc = np.gradient(angular_vel, dt)
    N = len(angle)
    indiv = np.tile([emg90_b,emg90_t,bmi,height,weight,gender],(N,1))
    temporal = np.column_stack([angle,np.full(N,emg90_ratio),sin_a,cos_a,angle_dev,angle_rel,angular_vel,angular_acc])
    features = np.column_stack([indiv,temporal])
    targets = envelopes/(anchor_emg+eps)
    features = features[::DOWNSAMPLE]; targets = targets[::DOWNSAMPLE]
    targets = np.nan_to_num(targets,nan=0.0,posinf=0.0,neginf=0.0)
    targets = np.clip(targets,0.0,np.percentile(targets,99.5,axis=0))
    return features.astype(np.float32), targets.astype(np.float32), anchor_emg, angle, envelopes, emg90_b, emg90_t

# ============================================================
# 指标
# ============================================================
def cos_sim(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))

def compute_metrics(yt,yp):
    yt,yp=np.asarray(yt).flatten(),np.asarray(yp).flatten()
    rmse=float(np.sqrt(mean_squared_error(yt,yp)))
    return {'Pearson_r':float(pearsonr(yt,yp)[0]),'Spearman_rho':float(spearmanr(yt,yp)[0]),
            'R2':float(r2_score(yt,yp)),'RMSE':rmse,'MAE':float(mean_absolute_error(yt,yp)),
            'Cosine_Similarity':cos_sim(yt,yp),
            'NRMSE':float(rmse/(np.max(yt)-np.min(yt)+1e-10))}

# ============================================================
# 加载模型和数据
# ============================================================
print("\n[1] 加载模型...")
model_dir = os.path.join(SCRIPT_DIR, 'results_loso')
ckpt = torch.load(os.path.join(model_dir, 'anchor_mlp_2.pt'), map_location=DEVICE, weights_only=False)
sd = ckpt['model_state_dict']
sd = {k.replace('_orig_mod.',''): v for k,v in sd.items()}
model = AnchorMLP().to(DEVICE); model.load_state_dict(sd); model.eval()
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
print(f"  model_2 | AnchorMLP 1.44M | scaler: {scaler.n_features_in_} features")

print("\n[2] 加载 S1-S10 数据...")
per_sub = {}
yt_all, yp_all, feat_all, angle_all = [], [], [], []
t0 = time.time()

for s in range(1, 11):
    mat = scipy.io.loadmat(os.path.join(SCRIPT_DIR, f'S{s}_E3_A1.mat'))
    features, ratios_true, anchor_emg, angle_raw, envelopes, emg90_b, emg90_t = build_features(mat)
    emg_true = ratios_true * anchor_emg[np.newaxis, :]
    X_norm = scaler.transform(features)
    with torch.no_grad():
        ratios_pred = model(torch.from_numpy(X_norm).to(DEVICE)).float().cpu().numpy()
    ratios_pred = np.clip(ratios_pred, 0, None)
    emg_pred = ratios_pred * anchor_emg[np.newaxis, :]

    mb = compute_metrics(emg_true[:,0], emg_pred[:,0])
    mt = compute_metrics(emg_true[:,1], emg_pred[:,1])

    per_sub[f'S{s}'] = {'bmi': float(mat['weight'][0][0])/(float(mat['height'][0][0])/100)**2,
        'emg90_b': float(emg90_b), 'emg90_t': float(emg90_t),
        'Biceps': mb, 'Triceps': mt,
        'emg_true': emg_true, 'emg_pred': emg_pred,
        'features': features, 'angle': angle_raw[::DOWNSAMPLE],
        'angles_full': angle_raw, 'envelopes': envelopes}

    yt_all.append(emg_true); yp_all.append(emg_pred)
    feat_all.append(features); angle_all.append(angle_raw[::DOWNSAMPLE])
    print(f"  S{s}: B_r={mb['Pearson_r']:.4f} T_r={mt['Pearson_r']:.4f}  EMG90=[{emg90_b:.0f},{emg90_t:.0f}]")

yt_all=np.concatenate(yt_all); yp_all=np.concatenate(yp_all)
feat_all=np.concatenate(feat_all); angle_all=np.concatenate(angle_all)
print(f"  总样本: {len(yt_all):,} | 耗时: {time.time()-t0:.1f}s")

# 整体指标
mb_overall = compute_metrics(yt_all[:,0], yp_all[:,0])
mt_overall = compute_metrics(yt_all[:,1], yp_all[:,1])
all_metrics = {'Biceps': mb_overall, 'Triceps': mt_overall, 'per_subject': {
    k: {'Biceps': v['Biceps'], 'Triceps': v['Triceps']} for k,v in per_sub.items()}}

print(f"  整体: Biceps r={mb_overall['Pearson_r']:.4f}  Triceps r={mt_overall['Pearson_r']:.4f}")

# ============================================================
# 16 张图
# ============================================================
print("\n[3] 生成图片...")

def save(fig, name):
    path = f'{OUTPUT_DIR}/{name}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")

# --- Fig 1: 预测曲线 ---
print("  [1/16]", end=" ")
fig, axes = plt.subplots(3, 3, figsize=(22, 16))
axes = axes.flatten()
for i, s in enumerate(range(1, 10)):
    ps = per_sub[f'S{s}']
    yt, yp = ps['emg_true'], ps['emg_pred']
    angle = ps['angle']
    sort_idx = np.argsort(angle)
    ax = axes[i]
    for m_idx, (m_name, c) in enumerate([('Biceps',COLOR_B),('Triceps',COLOR_T)]):
        ax.scatter(angle[::5], yt[::5,m_idx], alpha=0.15, s=3, color=c)
        ax.plot(angle[sort_idx], yp[sort_idx,m_idx], '-', color=c, linewidth=2,
                label=f'{m_name} (r={pearsonr(yt[:,m_idx],yp[:,m_idx])[0]:.3f})')
    ax.set_title(f'S{s} (BMI={ps["bmi"]:.1f})', fontsize=12); ax.legend(fontsize=8)
    ax.set_xlabel('Angle (°)'); ax.set_ylabel('EMG (μV)')
fig.suptitle('EMG Prediction Curves — Per Subject', fontsize=16, fontweight='bold', y=1.01)
fig.tight_layout(); save(fig, 'fig_01_prediction_curves.png')

# --- Fig 2: 回归散点 + Bland-Altman ---
print("[2/16]", end=" ")
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
for col, (m_name, m_idx, c) in enumerate([('Biceps',0,COLOR_B),('Triceps',1,COLOR_T)]):
    yt, yp = yt_all[:,m_idx], yp_all[:,m_idx]
    # 散点
    ax1 = fig.add_subplot(gs[0, col])
    subsample = np.random.choice(len(yt), min(5000,len(yt)), replace=False)
    ax1.scatter(yt[subsample], yp[subsample], alpha=0.3, s=2, color=c)
    lim = [min(yt.min(),yp.min()), max(yt.max(),yp.max())]
    ax1.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.7)
    r = pearsonr(yt,yp)[0]; rmse_val = np.sqrt(mean_squared_error(yt,yp))
    ax1.set_xlabel('True EMG (μV)'); ax1.set_ylabel('Predicted EMG (μV)')
    ax1.set_title(f'{m_name} Regression (r={r:.4f}, RMSE={rmse_val:.1f})')
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    # Bland-Altman
    ax2 = fig.add_subplot(gs[1, col])
    bias = np.mean(yp-yt); loa = 1.96*np.std(yp-yt)
    ax2.scatter((yt+yp)/2, yp-yt, alpha=0.3, s=2, color=c)
    ax2.axhline(bias, color='k', linestyle='--', linewidth=1.5, label=f'Bias={bias:.1f}')
    ax2.axhline(bias+loa, color='gray', linestyle=':', linewidth=1); ax2.axhline(bias-loa, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlabel('Mean (μV)'); ax2.set_ylabel('Difference (μV)')
    ax2.set_title(f'{m_name} Bland-Altman'); ax2.legend(fontsize=9)
    # 误差分布
    ax3 = fig.add_subplot(gs[:, 2]) if col == 0 else None
    if col == 0:
        for m_name2, m_idx2, c2 in [('Biceps',0,COLOR_B),('Triceps',1,COLOR_T)]:
            err = yp_all[:,m_idx2]-yt_all[:,m_idx2]
            ax3.hist(err, bins=100, alpha=0.5, color=c2, density=True, label=f'{m_name2} (σ={np.std(err):.1f})')
            x = np.linspace(err.min(), err.max(), 200)
            ax3.plot(x, norm.pdf(x, np.mean(err), np.std(err)), '-', color=c2, linewidth=2)
        ax3.set_xlabel('Error (μV)'); ax3.set_ylabel('Density'); ax3.set_title('Error Distribution'); ax3.legend()
fig.suptitle('Regression Analysis & Bland-Altman', fontsize=16, fontweight='bold')
save(fig, 'fig_02_regression_blandaltman.png')

# --- Fig 3: 残差诊断 ---
print("[3/16]", end=" ")
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
for col, (m_name, m_idx, c) in enumerate([('Biceps',0,COLOR_B),('Triceps',1,COLOR_T)]):
    yt, yp = yt_all[:,m_idx], yp_all[:,m_idx]; resid = yp-yt
    subsample = np.random.choice(len(yt), min(3000,len(yt)), replace=False)
    ax1=fig.add_subplot(gs[0,col*2]); ax1.scatter(yp[subsample],resid[subsample],alpha=0.3,s=3,color=c)
    ax1.axhline(0,color='k',linestyle='--',linewidth=1); ax1.set_xlabel('Fitted'); ax1.set_ylabel('Residuals')
    ax1.set_title(f'{m_name} Residuals vs Fitted')
    ax2=fig.add_subplot(gs[0,col*2+1])
    vals, bins = np.histogram(resid, bins=80, density=True)
    ax2.hist(resid, bins=80, density=True, alpha=0.7, color=c)
    x=np.linspace(resid.min(),resid.max(),200); ax2.plot(x,norm.pdf(x,np.mean(resid),np.std(resid)),'k-',linewidth=2)
    ax2.set_xlabel('Residual'); ax2.set_title(f'{m_name} Residual Histogram')
    ax3=fig.add_subplot(gs[1,col*2])
    probplot(resid, dist="norm", plot=ax3); ax3.get_lines()[0].set_color(c); ax3.get_lines()[1].set_color('k')
    ax3.set_title(f'{m_name} Q-Q Plot')
    ax4=fig.add_subplot(gs[1,col*2+1])
    sort_idx=np.argsort(angle_all[:len(resid)])
    ax4.scatter(angle_all[:len(resid)][sort_idx][::3],resid[sort_idx][::3],alpha=0.2,s=2,color=c)
    ax4.axhline(0,color='k',linestyle='--',linewidth=1); ax4.set_xlabel('Angle (°)'); ax4.set_ylabel('Residual')
    ax4.set_title(f'{m_name} Residuals vs Angle')
save(fig, 'fig_03_residuals.png')

# --- Fig 4: 误差分布 ---
print("[4/16]", end=" ")
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
for i, s in enumerate(range(1, 11)):
    ax = axes[i//5, i%5]
    ps = per_sub[f'S{s}']
    data_b = ps['emg_pred'][:,0]-ps['emg_true'][:,0]
    data_t = ps['emg_pred'][:,1]-ps['emg_true'][:,1]
    parts_b = ax.violinplot(data_b[::5], positions=[0], showmeans=True, showmedians=True)
    for pc in parts_b['bodies']: pc.set_facecolor(COLOR_B); pc.set_alpha(0.6)
    parts_t = ax.violinplot(data_t[::5], positions=[1], showmeans=True, showmedians=True)
    for pc in parts_t['bodies']: pc.set_facecolor(COLOR_T); pc.set_alpha(0.6)
    ax.set_title(f'S{s}'); ax.set_xticks([0,1]); ax.set_xticklabels(['B','T'])
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
fig.suptitle('Error Distribution by Subject', fontsize=16, fontweight='bold')
fig.tight_layout(); save(fig, 'fig_04_error_distribution.png')

# --- Fig 5: Taylor Diagram ---
print("[5/16]", end=" ")
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
ref_std = np.sqrt(np.mean([np.std(yt_all[:,i]) for i in range(2)]))
for m_name, m_idx, c, mkr in [('Biceps',0,COLOR_B,'o'),('Triceps',1,COLOR_T,'s')]:
    yt, yp = yt_all[:,m_idx], yp_all[:,m_idx]; r = pearsonr(yt,yp)[0]
    std_ratio = np.std(yp)/np.std(yt)
    theta = np.arccos(min(r, 0.999)); rho = std_ratio*ref_std
    ax.scatter(theta, rho, s=200, color=c, marker=mkr, edgecolors='k', linewidth=1, zorder=5, label=m_name)
ax.set_ylim(0, ref_std*1.5)
ax.set_title('Taylor Diagram', fontsize=16, fontweight='bold', pad=20)
rmse_lines = np.linspace(0.2, 1.2, 6)*ref_std
ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1)); fig.tight_layout()
save(fig, 'fig_05_taylor_diagram.png')

# --- Fig 6: EMG vs Angle ---
print("[6/16]", end=" ")
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
for i, s in enumerate(range(1, 11)):
    ax = axes[i//5, i%5]; ps = per_sub[f'S{s}']
    yt, yp = ps['emg_true'], ps['emg_pred']; angle = ps['angle']
    sort_idx = np.argsort(angle)
    ax.scatter(angle[::10], yt[::10,0], alpha=0.1, s=2, color=COLOR_B); ax.scatter(angle[::10], yt[::10,1], alpha=0.1, s=2, color=COLOR_T)
    ax.plot(angle[sort_idx], yp[sort_idx,0], '-', color=COLOR_B, linewidth=2)
    ax.plot(angle[sort_idx], yp[sort_idx,1], '-', color=COLOR_T, linewidth=2)
    ax.set_xlabel('Angle (°)'); ax.set_ylabel('EMG'); ax.set_title(f'S{s}')
fig.suptitle('EMG-Angle Relationship (Blue=Biceps, Red=Triceps)', fontsize=16, fontweight='bold')
fig.tight_layout(); save(fig, 'fig_06_emg_vs_angle.png')

# --- Fig 7: 时间轨迹 ---
print("[7/16]", end=" ")
fig, axes = plt.subplots(2, 1, figsize=(22, 10))
for col, (m_name, m_idx, c) in enumerate([('Biceps',0,COLOR_B),('Triceps',1,COLOR_T)]):
    n_show=min(5000, len(yt_all)); ax=axes[col]
    ax.plot(yt_all[:n_show,m_idx], '-', color=c, alpha=0.5, linewidth=0.5, label='True')
    ax.plot(yp_all[:n_show,m_idx], '--', color='k', alpha=0.6, linewidth=0.8, label='Predicted')
    ax.set_ylabel(f'{m_name} EMG (μV)'); ax.legend(); ax.set_title(f'{m_name} Temporal Trace')
axes[1].set_xlabel('Sample Index (100Hz)')
fig.suptitle('EMG Prediction — Temporal Trace', fontsize=16, fontweight='bold')
fig.tight_layout(); save(fig, 'fig_07_temporal.png')

# --- Fig 8: 综合仪表盘 ---
print("[8/16]", end=" ")
fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
# A: 整体散点
subsample = np.random.choice(len(yt_all), min(3000,len(yt_all)), replace=False)
ax=fig.add_subplot(gs[0,0])
for m_idx, m_name, c in [(0,'Biceps',COLOR_B),(1,'Triceps',COLOR_T)]:
    ax.scatter(yt_all[subsample,m_idx], yp_all[subsample,m_idx], alpha=0.3, s=3, color=c, label=f'{m_name} (r={pearsonr(yt_all[:,m_idx],yp_all[:,m_idx])[0]:.3f})')
lim=[yt_all.min(),yt_all.max()]; ax.plot(lim,lim,'k--',linewidth=1); ax.set_xlabel('True'); ax.set_ylabel('Predicted')
ax.set_title('Overall Prediction'); ax.legend(fontsize=9)
# B: 指标热图
ax=fig.add_subplot(gs[0,1])
metrics_mat = np.array([[mb_overall['Pearson_r'],mb_overall['R2'],mb_overall['NRMSE']],
                         [mt_overall['Pearson_r'],mt_overall['R2'],mt_overall['NRMSE']]])
im=ax.imshow(metrics_mat, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(['Pearson r','R²','1-NRMSE'])
ax.set_yticks(range(2)); ax.set_yticklabels(['Biceps','Triceps'])
for i in range(2):
    for j in range(3):
        ax.text(j,i,f'{metrics_mat[i,j]:.3f}',ha='center',va='center',fontsize=12,fontweight='bold')
ax.set_title('Metric Heatmap'); plt.colorbar(im, ax=ax)
# C: 误差分布
ax=fig.add_subplot(gs[0,2])
for m_name,c,m_idx in [('Biceps',COLOR_B,0),('Triceps',COLOR_T,1)]:
    err=yp_all[:,m_idx]-yt_all[:,m_idx]; ax.hist(err,bins=80,alpha=0.5,color=c,density=True,label=f'{m_name} MAE={np.mean(np.abs(err)):.1f}')
ax.set_xlabel('Error (μV)'); ax.set_title('Error Distribution'); ax.legend(fontsize=9)
# D-G: 前4位受试者详细
for i, s in enumerate([1,2,5,6]):
    ax=fig.add_subplot(gs[1+(i//2), i%2])
    ps=per_sub[f'S{s}']; yt,yp,angle=ps['emg_true'],ps['emg_pred'],ps['angle']
    sort_idx=np.argsort(angle)
    for m_idx,m_name,c in [(0,'Biceps',COLOR_B),(1,'Triceps',COLOR_T)]:
        ax.scatter(angle[::10],yt[::10,m_idx],alpha=0.1,s=2,color=c)
        ax.plot(angle[sort_idx],yp[sort_idx,m_idx],'-',color=c,linewidth=2)
    ax.set_xlabel('Angle (°)'); ax.set_ylabel('EMG'); ax.set_title(f'S{s}')
# H-I: 指标摘要
ax=fig.add_subplot(gs[2,2])
metrics_text = f"""AnchorMLP Model Performance
{'─'*30}
Biceps:
  Pearson r = {mb_overall['Pearson_r']:.4f}
  R² = {mb_overall['R2']:.4f}
  RMSE = {mb_overall['RMSE']:.1f} μV
  MAE = {mb_overall['MAE']:.1f} μV
  Cosine = {mb_overall['Cosine_Similarity']:.4f}
  NRMSE = {mb_overall['NRMSE']:.4f}

Triceps:
  Pearson r = {mt_overall['Pearson_r']:.4f}
  R² = {mt_overall['R2']:.4f}
  RMSE = {mt_overall['RMSE']:.1f} μV
  MAE = {mt_overall['MAE']:.1f} μV
  Cosine = {mt_overall['Cosine_Similarity']:.4f}
  NRMSE = {mt_overall['NRMSE']:.4f}"""
ax.text(0.05,0.95,metrics_text,transform=ax.transAxes,fontsize=9,fontfamily='monospace',va='top')
ax.axis('off')
save(fig, 'fig_08_dashboard.png')

# --- Fig 10: Pearson 独立图 ---
print("[9/16]", end=" ")
fig, ax = plt.subplots(figsize=(12, 8))
subjs = [f'S{i}' for i in range(1,11)]
x = np.arange(len(subjs)); w=0.35
b_r = [per_sub[s]['Biceps']['Pearson_r'] for s in subjs]
t_r = [per_sub[s]['Triceps']['Pearson_r'] for s in subjs]
bars1=ax.bar(x-w/2,b_r,w,color=COLOR_B,alpha=0.8,label='Biceps',edgecolor='k',linewidth=0.5)
bars2=ax.bar(x+w/2,t_r,w,color=COLOR_T,alpha=0.8,label='Triceps',edgecolor='k',linewidth=0.5)
# 标注数值
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{bar.get_height():.3f}',ha='center',fontsize=8)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{bar.get_height():.3f}',ha='center',fontsize=8)
ax.axhline(y=np.mean(b_r),color=COLOR_B,linestyle='--',linewidth=1,alpha=0.6,label=f'Mean B={np.mean(b_r):.3f}')
ax.axhline(y=np.mean(t_r),color=COLOR_T,linestyle='--',linewidth=1,alpha=0.6,label=f'Mean T={np.mean(t_r):.3f}')
ax.set_xticks(x); ax.set_xticklabels(subjs); ax.set_ylabel('Pearson r'); ax.set_title('Pearson Correlation by Subject')
ax.legend(); ax.set_ylim(0, 1); fig.tight_layout()
save(fig, 'fig_10_pearson.png')

# --- Fig 11: Cosine 独立图 ---
print("[10/16]", end=" ")
fig, ax = plt.subplots(figsize=(12, 8))
b_c=[per_sub[s]['Biceps']['Cosine_Similarity'] for s in subjs]
t_c=[per_sub[s]['Triceps']['Cosine_Similarity'] for s in subjs]
bars1=ax.bar(x-w/2,b_c,w,color=COLOR_B,alpha=0.8,label='Biceps',edgecolor='k',linewidth=0.5)
bars2=ax.bar(x+w/2,t_c,w,color=COLOR_T,alpha=0.8,label='Triceps',edgecolor='k',linewidth=0.5)
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{bar.get_height():.3f}',ha='center',fontsize=8)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{bar.get_height():.3f}',ha='center',fontsize=8)
ax.axhline(y=np.mean(b_c),color=COLOR_B,linestyle='--',linewidth=1,alpha=0.6,label=f'Mean B={np.mean(b_c):.3f}')
ax.axhline(y=np.mean(t_c),color=COLOR_T,linestyle='--',linewidth=1,alpha=0.6,label=f'Mean T={np.mean(t_c):.3f}')
ax.set_xticks(x); ax.set_xticklabels(subjs); ax.set_ylabel('Cosine Similarity'); ax.set_title('Cosine Similarity by Subject')
ax.legend(); ax.set_ylim(0,1); fig.tight_layout()
save(fig, 'fig_11_cosine.png')

# --- Fig 12: 互相关 ---
print("[11/16]", end=" ")
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for i, s in enumerate(range(1, 11)):
    ax = axes[i//5, i%5]; ps = per_sub[f'S{s}']
    for m_name, m_idx, c in [('Biceps',0,COLOR_B),('Triceps',1,COLOR_T)]:
        yt, yp = ps['emg_true'][:,m_idx], ps['emg_pred'][:,m_idx]
        lags=range(-50,51); cc=[np.corrcoef(yt[:len(yt)-abs(l)],yp[abs(l):len(yp)])[0,1] if l>=0 else np.corrcoef(yp[:len(yp)-abs(l)],yt[abs(l):len(yt)])[0,1] for l in lags]
        ax.plot(lags,cc,'-',color=c,linewidth=2,label=m_name)
    ax.set_xlabel('Lag'); ax.set_title(f'S{s}'); ax.axvline(0,color='gray',linestyle=':',linewidth=0.5)
    if i==0: ax.legend(fontsize=7)
fig.suptitle('Cross-Correlation Analysis', fontsize=16, fontweight='bold')
fig.tight_layout(); save(fig, 'fig_12_cross_correlation.png')

# --- Fig 13: Pearson vs Cosine 散点 ---
print("[12/16]", end=" ")
fig, ax = plt.subplots(figsize=(10, 8))
for s in range(1, 11):
    ps = per_sub[f'S{s}']
    ax.scatter(ps['Biceps']['Pearson_r'],ps['Biceps']['Cosine_Similarity'],color=COLOR_B,s=80,marker='o',edgecolors='k',linewidth=0.5,label='Biceps' if s==1 else '')
    ax.scatter(ps['Triceps']['Pearson_r'],ps['Triceps']['Cosine_Similarity'],color=COLOR_T,s=80,marker='s',edgecolors='k',linewidth=0.5,label='Triceps' if s==1 else '')
    ax.annotate(f'S{s}',(ps['Biceps']['Pearson_r'],ps['Biceps']['Cosine_Similarity']),fontsize=7,ha='right')
    ax.annotate(f'S{s}',(ps['Triceps']['Pearson_r'],ps['Triceps']['Cosine_Similarity']),fontsize=7,ha='right')
ax.set_xlabel('Pearson r'); ax.set_ylabel('Cosine Similarity'); ax.set_title('Pearson vs Cosine Similarity')
ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1); fig.tight_layout()
save(fig, 'fig_13_pearson_vs_cosine.png')

# --- Fig 14: 叠加对比 ---
print("[13/16]", end=" ")
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
for idx, s in enumerate([1, 3, 5, 7, 9, 10]):
    ax = axes[idx//3, idx%3]; ps = per_sub[f'S{s}']
    yt, yp, angle = ps['emg_true'], ps['emg_pred'], ps['angle']
    sort_idx = np.argsort(angle)
    for m_idx, m_name, c in [(0,'Biceps',COLOR_B),(1,'Triceps',COLOR_T)]:
        ax.fill_between(angle[sort_idx][::5], yt[sort_idx][::5,m_idx], yp[sort_idx][::5,m_idx], alpha=0.2, color=c)
        ax.plot(angle[sort_idx], yt[sort_idx,m_idx], '-', color=c, linewidth=1.5, alpha=0.7, label=f'{m_name} True')
        ax.plot(angle[sort_idx], yp[sort_idx,m_idx], '--', color=c, linewidth=1.5, label=f'{m_name} Pred')
    ax.set_xlabel('Angle (°)'); ax.set_title(f'S{s}'); ax.legend(fontsize=7)
fig.suptitle('True vs Predicted Overlay Comparison', fontsize=16, fontweight='bold')
fig.tight_layout(); save(fig, 'fig_14_overlay_comparison.png')

# --- Fig 15: 特征重要性 ---
print("[14/16]", end=" ")
FEATURE_NAMES = ['EMG90_B','EMG90_T','BMI','Height','Weight','Gender','Angle',
                  'EMG90_Ratio','Sin(Angle)','Cos(Angle)','Angle_Dev','Angle_Rel','Ang_Vel','Ang_Acc']
from sklearn.base import BaseEstimator, RegressorMixin
class ModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model): self.model = model
    def fit(self, X, y): return self
    def predict(self, X):
        self.model.eval()
        loader = DataLoader(torch.from_numpy(X.astype(np.float32)), 16384, shuffle=False)
        preds = []
        with torch.no_grad():
            for bx in loader: preds.append(self.model(bx.to(DEVICE)).float().detach().cpu().numpy())
        return np.clip(np.concatenate(preds), 0, None)

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
X_sample = feat_all[::20][:20000]
for m_idx, m_name, ax in [(0,'Biceps',axes[0]),(1,'Triceps',axes[1])]:
    try:
        wrapper = ModelWrapper(model)
        n_repeats = 5
        importances = np.zeros((n_repeats, N_FEATURES))
        y_ref = wrapper.predict(scaler.transform(X_sample))[:, m_idx]
        base_score = r2_score(y_ref, y_ref)
        for rep in range(n_repeats):
            X_perm = scaler.transform(X_sample).copy()
            for j in range(N_FEATURES):
                X_perm_j = X_perm.copy()
                np.random.shuffle(X_perm_j[:, j])
                y_perm = wrapper.predict(X_perm_j)[:, m_idx]
                importances[rep, j] = max(0, base_score - r2_score(y_ref, y_perm))
        imp_mean = importances.mean(axis=0)
        imp_std = importances.std(axis=0)
        idx_sorted = np.argsort(imp_mean)[::-1]
        ax.barh(range(N_FEATURES), imp_mean[idx_sorted], xerr=imp_std[idx_sorted],
                color=COLOR_B if m_idx==0 else COLOR_T, alpha=0.8, edgecolor='k', linewidth=0.5)
        ax.set_yticks(range(N_FEATURES)); ax.set_yticklabels([FEATURE_NAMES[i] for i in idx_sorted])
        ax.set_xlabel('Importance (ΔR²)'); ax.set_title(f'{m_name} Feature Importance')
    except Exception as e:
        ax.text(0.5,0.5,f'Importance estimation error:\n{e}',ha='center',va='center',transform=ax.transAxes)
fig.suptitle('Permutation Feature Importance', fontsize=16, fontweight='bold')
fig.tight_layout(); save(fig, 'fig_15_feature_importance.png')

# --- Fig 16: 锚点校准流程 ---
print("[15/16]", end=" ")
fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.3)
for i, s in enumerate([1, 3, 5, 7, 9]):
    ax1=fig.add_subplot(gs[0,i])
    ps=per_sub[f'S{s}']; s_idx=np.argsort(ps['angle'])
    ax1.plot(ps['angle'][s_idx], ps['emg_true'][s_idx,0], '-', color=COLOR_B, linewidth=2, alpha=0.8)
    ax1.plot(ps['angle'][s_idx], ps['emg_pred'][s_idx,0], '--', color='k', linewidth=1.5, alpha=0.7)
    ax1.axvline(90, color='gray', linestyle=':', linewidth=2)
    ax1.axhline(ps['emg90_b'], color=COLOR_B, linestyle=':', linewidth=1)
    ax1.set_xlabel('Angle'); ax1.set_ylabel('Biceps EMG'); ax1.set_title(f'S{s}')
    ax2=fig.add_subplot(gs[1,i])
    ax2.plot(ps['angle'][s_idx], ps['emg_true'][s_idx,1], '-', color=COLOR_T, linewidth=2, alpha=0.8)
    ax2.plot(ps['angle'][s_idx], ps['emg_pred'][s_idx,1], '--', color='k', linewidth=1.5, alpha=0.7)
    ax2.axvline(90, color='gray', linestyle=':', linewidth=2)
    ax2.axhline(ps['emg90_t'], color=COLOR_T, linestyle=':', linewidth=1)
    ax2.set_xlabel('Angle'); ax2.set_ylabel('Triceps EMG'); ax2.set_title(f'S{s}')
fig.suptitle('Anchor Calibration Pipeline (90° Ref Line)', fontsize=16, fontweight='bold')
save(fig, 'fig_16_anchor_calibration.png')

# --- Fig 9: LOSO 对比 (使用本地保存的结果或跳过) ---
print("[16/16]", end=" ")
loso_path = os.path.join(model_dir, 'loso_results.json')
loso_results = None
if os.path.exists(loso_path):
    with open(loso_path) as f: loso_results = json.load(f)

fig, ax = plt.subplots(figsize=(12, 8))
b_loso = loso_results.get('Biceps_ratio',{}).get('Pearson_r',{}).get('mean',0.0) if loso_results else 0
t_loso = loso_results.get('Triceps_ratio',{}).get('Pearson_r',{}).get('mean',0.0) if loso_results else 0
b_full = all_metrics['Biceps']['Pearson_r']
t_full = all_metrics['Triceps']['Pearson_r']
x_labels = ['Biceps LOSO', 'Triceps LOSO', 'Biceps Full', 'Triceps Full']
values = [b_loso, t_loso, b_full, t_full]
colors_bar = [COLOR_B, COLOR_T, COLOR_B, COLOR_T]
bars = ax.bar(x_labels, values, color=colors_bar, alpha=0.85, edgecolor='k', linewidth=1)
# LOSO bars semi-transparent
if loso_results:
    bars[0].set_alpha(0.5); bars[1].set_alpha(0.5)
for i, v in enumerate(values):
    ax.text(i, v+0.01, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Pearson r'); ax.set_title('LOSO vs Full-Train Performance')
ax.set_ylim(0, 1)
if not loso_results:
    ax.text(0.5, 0.5, 'LOSO results not available\n(showing Full-Train only)', ha='center', va='center',
            transform=ax.transAxes, fontsize=14, alpha=0.5)
fig.tight_layout(); save(fig, 'fig_09_loso_comparison.png')

# ============================================================
# 保存指标和报告
# ============================================================
print("\n[4] 保存指标和报告...")

# JSON
def convert(obj):
    if isinstance(obj, dict): return {k: convert(v) for k,v in obj.items()}
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(f'{OUTPUT_DIR}/test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(convert(all_metrics), f, indent=2, ensure_ascii=False)
print(f"  test_metrics.json")

# TXT Report
rpt = []
rpt.append("="*70)
rpt.append("  AnchorMLP (model_2) — Comprehensive Test Report")
rpt.append("="*70)
rpt.append(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
rpt.append(f"  Model: AnchorMLP (1.44M params, Residual MLP ×5)")
rpt.append(f"  Features: 14-dim (7 user + 7 derived)")
rpt.append(f"  Output: [r_biceps, r_triceps] → EMG = r × EMG90")
rpt.append("")
rpt.append("-"*70)
rpt.append("  Overall Metrics")
rpt.append("-"*70)
rpt.append(f"  {'Metric':<25} {'Biceps':>14} {'Triceps':>14}")
rpt.append(f"  {'-'*25} {'-'*14} {'-'*14}")
for metric in ['Pearson_r','Spearman_rho','R2','RMSE','MAE','Cosine_Similarity','NRMSE']:
    rpt.append(f"  {metric:<25} {all_metrics['Biceps'][metric]:>14.4f} {all_metrics['Triceps'][metric]:>14.4f}")
rpt.append("")
rpt.append("-"*70)
rpt.append("  Per-Subject Breakdown")
rpt.append("-"*70)
rpt.append(f"  {'Subj':<6} {'Biceps r':>10} {'Biceps R²':>10} {'Triceps r':>10} {'Triceps R²':>10} {'B RMSE':>10} {'T RMSE':>10}")
rpt.append(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for s in range(1, 11):
    ps = all_metrics['per_subject'][f'S{s}']
    rpt.append(f"  S{s:<5} {ps['Biceps']['Pearson_r']:>10.4f} {ps['Biceps']['R2']:>10.4f} "
              f"{ps['Triceps']['Pearson_r']:>10.4f} {ps['Triceps']['R2']:>10.4f} "
              f"{ps['Biceps']['RMSE']:>10.1f} {ps['Triceps']['RMSE']:>10.1f}")
rpt.append("")
rpt.append("-"*70)
rpt.append("  Generated Figures (16)")
rpt.append("-"*70)
figs = ['fig_01_prediction_curves.png', 'fig_02_regression_blandaltman.png',
    'fig_03_residuals.png', 'fig_04_error_distribution.png', 'fig_05_taylor_diagram.png',
    'fig_06_emg_vs_angle.png', 'fig_07_temporal.png', 'fig_08_dashboard.png',
    'fig_09_loso_comparison.png', 'fig_10_pearson.png', 'fig_11_cosine.png',
    'fig_12_cross_correlation.png', 'fig_13_pearson_vs_cosine.png',
    'fig_14_overlay_comparison.png', 'fig_15_feature_importance.png',
    'fig_16_anchor_calibration.png']
for f in figs: rpt.append(f"  {f}")

rpt_text = '\n'.join(rpt)
with open(f'{OUTPUT_DIR}/test_report.txt', 'w', encoding='utf-8') as f:
    f.write(rpt_text)
print(f"  test_report.txt")

print(f"\n[5] Done! 输出: {OUTPUT_DIR}/")
print(f"  总文件: 16 张图 + test_metrics.json + test_report.txt")
