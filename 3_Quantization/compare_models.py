"""
全方位对比两个模型的预测精度
  Set 1: train2.py 新训练 (log-ratio, 21特征, SmoothL1+Pearson+Peak, 7-seed Ensemble)
  Set 2: 旧 LOSO 模型 (ratio, 14特征, MSE+Pearson+Cosine+Peak, 单模型)
"""
import torch, joblib, os, sys, io, time
import numpy as np
from contextlib import nullcontext
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import scipy.io
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE}, AMP: {USE_AMP}")

FS, DOWNSAMPLE = 2000, 20
CH_BICEPS, CH_TRICEPS = 10, 11

def extract_emg_envelope(emg_signal, fs=2000):
    b_n, a_n = iirnotch(50, 30, fs)
    sig = filtfilt(b_n, a_n, emg_signal, axis=0)
    b_bp, a_bp = butter(4, [20 / (0.5 * fs), 450 / (0.5 * fs)], btype='band')
    sig = filtfilt(b_bp, a_bp, sig, axis=0)
    sig = np.abs(sig)
    b_lp, a_lp = butter(4, 6 / (0.5 * fs), btype='low')
    return filtfilt(b_lp, a_lp, sig, axis=0)

def find_data_matrix(obj):
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[0] > 100 and obj.dtype != 'O':
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype == 'O':
        for item in obj.flat:
            r = find_data_matrix(item)
            if r is not None:
                return r
    return None

def extract_angle(mat, fs=2000):
    if 'angle' in mat:
        d = find_data_matrix(mat['angle'])
        if d is not None:
            a = d.flatten().astype(np.float64)
            if a.max() > 180:
                a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 130.0
            return np.clip(a, 0, 130), 'real_sensor'
    if 'inclin' in mat:
        d = find_data_matrix(mat['inclin'])
        if d is not None:
            a = d.flatten().astype(np.float64)
            emg_raw = find_data_matrix(mat['emg'])
            b_env = extract_emg_envelope(emg_raw[:, CH_BICEPS], fs)
            corr = np.corrcoef(a, b_env)[0, 1]
            if not np.isnan(corr) and corr < 0:
                a = -a
            a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 130.0
            return a, 'inclinometer'
    if 'restimulus' in mat:
        from scipy.ndimage import convolve1d
        restim = find_data_matrix(mat['restimulus']).flatten()
        x = np.linspace(-1500, 1500, 3000)
        kernel = np.exp(-0.5 * (x / 700) ** 2)
        kernel /= kernel.sum()
        smooth = convolve1d((restim > 0).astype(float), kernel)
        a = (smooth / (smooth.max() + 1e-9)) * 130.0
        return a, 'restimulus_derived'
    raise ValueError("No angle data")

# ── 模型定义 ──
class ResidualBlock(nn.Module):
    def __init__(self, dim, expansion=1.5, dropout=0.1):
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.LayerNorm(dim)
        self.l1 = nn.Linear(dim, hidden)
        self.l2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        r = x
        x = self.norm(x)
        x = F.relu(self.l1(x))
        x = self.drop(x)
        x = self.l2(x)
        return F.relu(x + r)

class AnchorMLP_Old(nn.Module):
    """14维输入, ratio输出"""
    def __init__(self, input_dim=14, embed_dim=256, n_blocks=5, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim, 2.0, dropout) for _ in range(n_blocks)])
        self.skip = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Dropout(dropout))
        self.neck = nn.Sequential(
            nn.Linear(embed_dim + 64, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout))
        self.head_b = nn.Linear(128, 1)
        self.head_t = nn.Linear(128, 1)
    def forward(self, x):
        feat = self.blocks(self.embed(x))
        sk = self.skip(torch.cat([x[:, 6:7], x[:, :2]], dim=1))
        shared = self.neck(torch.cat([feat, sk], dim=1))
        return torch.cat([self.head_b(shared), self.head_t(shared)], dim=1)

class AnchorMLP_New(nn.Module):
    """21维输入, log-ratio输出"""
    def __init__(self, input_dim=21, embed_dim=512, n_blocks=7, n_outputs=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim, 2.0, dropout) for _ in range(n_blocks)])
        self.skip = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Dropout(dropout))
        neck_dim = embed_dim // 2
        self.neck = nn.Sequential(
            nn.Linear(embed_dim + 64, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, neck_dim), nn.LayerNorm(neck_dim), nn.ReLU(),
            nn.Dropout(dropout))
        self.head_b = nn.Linear(neck_dim, 1)
        self.head_t = nn.Linear(neck_dim, 1)
        self.n_outputs = n_outputs
    def forward(self, x):
        feat = self.blocks(self.embed(x))
        sk = self.skip(torch.cat([x[:, 11:12], x[:, :2]], dim=1))
        shared = self.neck(torch.cat([feat, sk], dim=1))
        if self.n_outputs == 2:
            return torch.cat([self.head_b(shared), self.head_t(shared)], dim=1)
        else:
            return self.head_b(shared)

# ── 特征构建 ──
def build_features_old(mat, fs=2000):
    """旧版: 14维特征, ratio目标"""
    eps, dt = 1e-8, 1.0 / fs
    gender = 1 if 'f' in str(mat.get('gender', 'm')).lower() else 0
    weight = float(mat['weight'][0][0]) if 'weight' in mat else 70.0
    height = float(mat['height'][0][0]) if 'height' in mat else 175.0
    bmi = weight / ((height / 100) ** 2)

    emg_raw = find_data_matrix(mat['emg'])
    envelopes = np.column_stack([
        extract_emg_envelope(emg_raw[:, CH_BICEPS], fs),
        extract_emg_envelope(emg_raw[:, CH_TRICEPS], fs)])

    angle, angle_source = extract_angle(mat, fs)
    N = min(len(angle), len(envelopes)); angle, envelopes = angle[:N], envelopes[:N]
    raw_vel = np.gradient(angle, dt)

    mask = ((angle >= 88) & (angle <= 92) & (np.abs(raw_vel) < 30))
    if mask.sum() >= 10:
        anchor = np.median(envelopes[mask], axis=0)
    elif mask.sum() >= 3:
        anchor = np.mean(envelopes[mask], axis=0)
    else:
        mask_l = (angle >= 88) & (angle <= 92)
        if mask_l.sum() >= 3:
            anchor = np.mean(envelopes[mask_l], axis=0)
        else:
            anchor = np.zeros(2)
            for m in range(2):
                si = np.argsort(angle); sa, se = angle[si], envelopes[si, m]
                ua, ue = [], []
                for av in np.unique(np.round(sa, 1)):
                    ma = np.abs(sa - av) < 0.5; ua.append(av); ue.append(np.mean(se[ma]))
                ua, ue = np.array(ua), np.array(ue)
                anchor[m] = np.interp(90, ua, ue) if ua[0] <= 90 <= ua[-1] else ue[np.argmin(np.abs(ua - 90))]
    anchor = np.maximum(anchor, np.percentile(envelopes, 5, axis=0) * 2.0)

    emg90_b, emg90_t = anchor[0], anchor[1]
    emg90_ratio = emg90_b / (emg90_t + eps)
    rad = angle / 130.0 * np.pi
    b_lp, a_lp = butter(2, 15 / (0.5 * fs), btype='low')
    angular_vel = filtfilt(b_lp, a_lp, raw_vel)
    angular_acc = np.gradient(angular_vel, dt)

    indiv = np.tile([emg90_b, emg90_t, bmi, height, weight, gender], (N, 1))
    temporal = np.column_stack([angle, np.full(N, emg90_ratio), np.sin(rad), np.cos(rad),
                                np.abs(angle - 90.0), angle / 90.0, angular_vel, angular_acc])
    features = np.column_stack([indiv, temporal])
    targets = envelopes / (anchor + eps)

    features, targets = features[::DOWNSAMPLE], targets[::DOWNSAMPLE]
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.clip(targets, 0.0, np.percentile(targets, 99.5, axis=0))
    return features.astype(np.float32), targets.astype(np.float32), float(emg90_b), float(emg90_t)


def build_features_new(mat, fs=2000):
    """新版: 21维特征, log-ratio目标"""
    eps, dt = 1e-8, 1.0 / fs
    gender = 1 if 'f' in str(mat.get('gender', 'm')).lower() else 0
    weight = float(mat['weight'][0][0]) if 'weight' in mat else 70.0
    height = float(mat['height'][0][0]) if 'height' in mat else 175.0
    bmi = weight / ((height / 100) ** 2)

    emg_raw = find_data_matrix(mat['emg'])
    envelopes = np.column_stack([
        extract_emg_envelope(emg_raw[:, CH_BICEPS], fs),
        extract_emg_envelope(emg_raw[:, CH_TRICEPS], fs)])

    angle, angle_source = extract_angle(mat, fs)
    N = min(len(angle), len(envelopes)); angle, envelopes = angle[:N], envelopes[:N]

    b_ang, a_ang = butter(2, 10 / (0.5 * fs), btype='low')
    angle_smooth = filtfilt(b_ang, a_ang, angle)
    raw_vel = np.gradient(angle_smooth, dt)

    mask = ((angle_smooth >= 85) & (angle_smooth <= 95) & (np.abs(raw_vel) < 20))
    n_samples = mask.sum()
    if n_samples >= 10:
        anchor = np.median(envelopes[mask], axis=0)
    elif n_samples >= 3:
        anchor = np.mean(envelopes[mask], axis=0)
    else:
        mask_l = (angle_smooth >= 85) & (angle_smooth <= 95)
        n_l = mask_l.sum()
        if n_l >= 3:
            anchor = np.mean(envelopes[mask_l], axis=0); n_samples = n_l
        else:
            anchor = np.zeros(2)
            for m in range(2):
                si = np.argsort(angle_smooth); sa, se = angle_smooth[si], envelopes[si, m]
                ua, ue = [], []
                for av in np.unique(np.round(sa, 1)):
                    ma = np.abs(sa - av) < 0.5; ua.append(av); ue.append(np.mean(se[ma]))
                ua, ue = np.array(ua), np.array(ue)
                anchor[m] = np.interp(90, ua, ue) if ua[0] <= 90 <= ua[-1] else ue[np.argmin(np.abs(ua - 90))]
            n_samples = 0

    active = mask if mask.sum() >= 3 else ((angle_smooth >= 85) & (angle_smooth <= 95))
    anchor_std = np.std(envelopes[active], axis=0) if active.sum() >= 3 else np.zeros(2)
    anchor_cv = anchor_std / (anchor + 1e-8)
    for m in range(2):
        nf = np.percentile(envelopes[:, m], 5)
        if anchor[m] < max(nf * 2.0, 1e-3): anchor[m] = max(nf * 2.0, 1e-3)

    emg90_b, emg90_t = anchor[0], anchor[1]
    emg90_ratio = emg90_b / (emg90_t + eps)
    rad = angle_smooth / 130.0 * np.pi
    b_lp, a_lp = butter(2, 15 / (0.5 * fs), btype='low')
    angular_vel = filtfilt(b_lp, a_lp, raw_vel)
    angular_acc = np.gradient(angular_vel, dt)
    movement_phase = np.zeros(N)
    movement_phase[angular_vel > 20] = 1; movement_phase[angular_vel < -20] = -1
    angle_prev = np.roll(angle_smooth, 1); angle_prev[0] = angle_smooth[0]

    indiv = np.tile([emg90_b, emg90_t, bmi, height, weight, gender,
                     anchor_std[0], anchor_std[1], anchor_cv[0], anchor_cv[1],
                     n_samples], (N, 1))
    temporal = np.column_stack([angle_smooth, np.full(N, emg90_ratio),
                                np.sin(rad), np.cos(rad),
                                np.abs(angle_smooth - 90.0), angle_smooth / 90.0,
                                angular_vel, angular_acc, movement_phase, angle_prev])
    features = np.column_stack([indiv, temporal])
    targets = np.log(envelopes / (anchor + eps) + eps)

    features, targets = features[::DOWNSAMPLE], targets[::DOWNSAMPLE]
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.clip(targets, -5.0, 5.0)
    return features.astype(np.float32), targets.astype(np.float32), float(emg90_b), float(emg90_t)


def calc_metrics(yt, yp):
    a, b = yt.flatten(), yp.flatten()
    pr, _ = pearsonr(a, b)
    rmse = float(np.sqrt(mean_squared_error(a, b)))
    return {'R2': float(r2_score(a, b)), 'MAE': float(mean_absolute_error(a, b)),
            'RMSE': rmse, 'Pearson_r': float(pr),
            'NRMSE': rmse / (a.max() - a.min() + 1e-9)}

@torch.no_grad()
def eval_old(model, feats, targets, eb, et, scaler):
    feats_s = scaler.transform(feats).astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(feats_s), torch.from_numpy(targets))
    loader = DataLoader(ds, 16384, shuffle=False)
    yp_list, yt_list = [], []
    model.eval()
    with autocast('cuda', dtype=torch.bfloat16) if USE_AMP else nullcontext():
        for bx, by in loader:
            pred = model(bx.to(DEVICE))
            yp_list.append(pred.float().cpu().numpy()); yt_list.append(by.float().numpy())
    r_pred = np.clip(np.concatenate(yp_list), 0, None)
    r_true = np.concatenate(yt_list)
    emg_pred, emg_true = r_pred * [eb, et], r_true * [eb, et]
    results = {}
    for i, name in enumerate(['Biceps', 'Triceps']):
        results[f'{name}_ratio'] = calc_metrics(r_true[:, i], r_pred[:, i])
        results[f'{name}_emg']   = calc_metrics(emg_true[:, i], emg_pred[:, i])
    return results

@torch.no_grad()
def eval_new_single(model, feats, targets, eb, et, scaler):
    feats_s = scaler.transform(feats).astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(feats_s), torch.from_numpy(targets))
    loader = DataLoader(ds, 16384, shuffle=False)
    yp_list, yt_list = [], []
    model.eval()
    with autocast('cuda', dtype=torch.bfloat16) if USE_AMP else nullcontext():
        for bx, by in loader:
            pred = model(bx.to(DEVICE))
            yp_list.append(pred.float().cpu().numpy()); yt_list.append(by.float().numpy())
    ypl, ytl = np.concatenate(yp_list), np.concatenate(yt_list)
    r_pred = np.exp(np.clip(ypl, -10, 10)); r_true = np.exp(ytl)
    emg_pred = np.clip(r_pred * [eb, et], 0, None); emg_true = r_true * [eb, et]
    results = {}
    for i, name in enumerate(['Biceps', 'Triceps']):
        results[f'{name}_ratio'] = calc_metrics(r_true[:, i], r_pred[:, i])
        results[f'{name}_emg']   = calc_metrics(emg_true[:, i], emg_pred[:, i])
    return results

@torch.no_grad()
def eval_new_ensemble(models, feats, targets, eb, et, scaler):
    feats_s = scaler.transform(feats).astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(feats_s), torch.from_numpy(targets))
    loader = DataLoader(ds, 16384, shuffle=False)
    all_preds = []
    for model in models:
        model.eval(); yp = []
        with autocast('cuda', dtype=torch.bfloat16) if USE_AMP else nullcontext():
            for bx, by in loader:
                yp.append(model(bx.to(DEVICE)).float().cpu().numpy())
        all_preds.append(np.concatenate(yp))
    ypl = np.mean(all_preds, axis=0)
    yt_list = [by.float().numpy() for bx, by in loader]; ytl = np.concatenate(yt_list)
    r_pred = np.exp(np.clip(ypl, -10, 10)); r_true = np.exp(ytl)
    emg_pred = np.clip(r_pred * [eb, et], 0, None); emg_true = r_true * [eb, et]
    results = {}
    for i, name in enumerate(['Biceps', 'Triceps']):
        results[f'{name}_ratio'] = calc_metrics(r_true[:, i], r_pred[:, i])
        results[f'{name}_emg']   = calc_metrics(emg_true[:, i], emg_pred[:, i])
    return results

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
print("\nLoading models...")

# Set 1: 新模型
scaler1 = joblib.load(r"c:\Users\LENOVO\Downloads\scaler (1).pkl")
ckpt1 = torch.load(r"c:\Users\LENOVO\Downloads\anchor_mlp (1).pt", map_location='cpu', weights_only=False)
model1 = AnchorMLP_New().to(DEVICE); model1.load_state_dict(ckpt1['model_state_dict'])
ens1 = torch.load(r"c:\Users\LENOVO\Downloads\anchor_mlp_ensemble.pt", map_location='cpu', weights_only=False)
ensemble_models = []
for sd in ens1['models']:
    m = AnchorMLP_New().to(DEVICE); m.load_state_dict(sd); ensemble_models.append(m)
print(f"Set 1 (New): {ckpt1['config'].get('target_type','?')}, "
      f"{ckpt1['config'].get('features',['?'])[:3]}... | Ensemble: {len(ensemble_models)} models")

# Set 2: 旧模型
scaler2 = joblib.load(r"c:\Users\LENOVO\Downloads\model_export\results_loso\scaler_2.pkl")
ckpt2 = torch.load(r"c:\Users\LENOVO\Downloads\model_export\results_loso\anchor_mlp_2.pt", map_location='cpu', weights_only=False)
model2 = AnchorMLP_Old().to(DEVICE); model2.load_state_dict(ckpt2['model_state_dict'])
print(f"Set 2 (Old): {ckpt2['config'].get('output_type','?')}, "
      f"{ckpt2['config'].get('internal_features',['?'])[:3]}...")

# ══════════════════════════════════════════════════════════════
t0 = time.time()
print(f"\n{'='*85}")
print(f"  Per-Subject EMG Prediction Comparison")
print(f"{'='*85}")
print(f"  {'S':<4s} {'B r':>6s} {'B R²':>7s} {'T r':>6s} {'T R²':>7s}"
      f"  │  {'B r':>6s} {'B R²':>7s} {'T r':>6s} {'T R²':>7s}"
      f"  │  {'B r':>6s} {'B R²':>7s} {'T r':>6s} {'T R²':>7s}")
print(f"  {'':4s} {'Set 1 (New)':>30s}  │  {'Set 1 (Ens)':>30s}  │  {'Set 2 (Old)':>30s}")
print(f"  {'-'*4} {'-'*6} {'-'*7} {'-'*6} {'-'*7}--┼--{'-'*6} {'-'*7} {'-'*6} {'-'*7}--┼--{'-'*6} {'-'*7} {'-'*6} {'-'*7}")

all_new_s, all_new_e, all_old = [], [], []

for s in range(1, 11):
    fp = os.path.join(DATA_DIR, f'S{s}_E3_A1.mat')
    mat = scipy.io.loadmat(fp)
    fo, to, ebo, eto = build_features_old(mat)
    fn, tn, ebn, etn = build_features_new(mat)

    r1 = eval_new_single(model1, fn, tn, ebn, etn, scaler1)
    re = eval_new_ensemble(ensemble_models, fn, tn, ebn, etn, scaler1)
    r2 = eval_old(model2, fo, to, ebo, eto, scaler2)

    all_new_s.append(r1); all_new_e.append(re); all_old.append(r2)

    print(f"  S{s:<3d} {r1['Biceps_emg']['Pearson_r']:>6.4f} {r1['Biceps_emg']['R2']:>7.4f} "
          f"{r1['Triceps_emg']['Pearson_r']:>6.4f} {r1['Triceps_emg']['R2']:>7.4f}"
          f"  │  {re['Biceps_emg']['Pearson_r']:>6.4f} {re['Biceps_emg']['R2']:>7.4f} "
          f"{re['Triceps_emg']['Pearson_r']:>6.4f} {re['Triceps_emg']['R2']:>7.4f}"
          f"  │  {r2['Biceps_emg']['Pearson_r']:>6.4f} {r2['Biceps_emg']['R2']:>7.4f} "
          f"{r2['Triceps_emg']['Pearson_r']:>6.4f} {r2['Triceps_emg']['R2']:>7.4f}")

# ── 汇总 ──
print(f"\n{'='*85}")
print(f"  Grand Summary (mean ± std over 10 subjects)")
print(f"{'='*85}")

for label, data in [("Set 1 (New, single)", all_new_s),
                     ("Set 1 (New, ensemble)", all_new_e),
                     ("Set 2 (Old, single)", all_old)]:
    print(f"\n  {label}")
    print(f"  {'Muscle':<10s} {'Pearson r':>16s} {'R²':>14s} {'RMSE':>12s} {'NRMSE':>10s} {'MAE':>10s}")
    print(f"  {'-'*10} {'-'*16} {'-'*14} {'-'*12} {'-'*10} {'-'*10}")
    for m in ['Biceps_emg', 'Triceps_emg']:
        r  = np.mean([x[m]['Pearson_r'] for x in data])
        rs = np.std([x[m]['Pearson_r'] for x in data])
        r2  = np.mean([x[m]['R2'] for x in data])
        r2s = np.std([x[m]['R2'] for x in data])
        rm  = np.mean([x[m]['RMSE'] for x in data])
        nr  = np.mean([x[m]['NRMSE'] for x in data])
        ma  = np.mean([x[m]['MAE'] for x in data])
        print(f"  {m:<10s} {r:>+.4f}±{rs:.4f}  {r2:>+.4f}±{r2s:.4f}  "
              f"{rm:>8.1f}  {nr:>.4f}  {ma:>8.1f}")

# ── 获胜统计 ──
print(f"\n{'='*85}")
print(f"  Head-to-Head: Set 1 Ensemble vs Set 2 Old")
print(f"{'='*85}")

for m in ['Biceps_emg', 'Triceps_emg']:
    wins_new, wins_old, ties = 0, 0, 0
    for i in range(10):
        rn = all_new_e[i][m]['Pearson_r']
        ro = all_old[i][m]['Pearson_r']
        if rn > ro + 0.005: wins_new += 1
        elif ro > rn + 0.005: wins_old += 1
        else: ties += 1
    name = m.replace('_emg','')
    print(f"  {name}: Set1 Ensemble wins={wins_new}, Set2 Old wins={wins_old}, ties={ties} / 10")

# ── 架构对比 ──
print(f"\n{'='*85}")
print(f"  Architecture Comparison")
print(f"{'='*85}")
print(f"  {'':<30s} {'Set 1 (New)':>22s} {'Set 2 (Old)':>22s}")
print(f"  {'-'*30} {'-'*22} {'-'*22}")
print(f"  {'Input dim':<30s} {21:>22d} {14:>22d}")
print(f"  {'Embed dim':<30s} {512:>22d} {256:>22d}")
print(f"  {'Residual blocks':<30s} {7:>22d} {5:>22d}")
print(f"  {'Target':<30s} {'log-ratio':>22s} {'ratio':>22s}")
print(f"  {'Loss':<30s} {'SmoothL1+Pearson+Peak':>22s} {'MSE+Pearson+Cos+Peak':>22s}")
print(f"  {'Anchor range':<30s} {'85-95° vel<20':>22s} {'88-92° vel<30':>22s}")
print(f"  {'Angle smoothing':<30s} {'Yes (10Hz LP)':>22s} {'No':>22s}")
print(f"  {'Anchor quality feat':<30s} {'Yes (5 dims)':>22s} {'No':>22s}")
print(f"  {'Movement phase':<30s} {'Yes':>22s} {'No':>22s}")
print(f"  {'Ensemble':<30s} {f'{len(ensemble_models)} models':>22s} {'1 model':>22s}")
print(f"  {'Val-based selection':<30s} {'Yes':>22s} {'No (train loss)':>22s}")

elapsed = time.time() - t0
print(f"\n  Done in {elapsed:.1f}s")
