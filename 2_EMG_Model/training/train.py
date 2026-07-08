#!/usr/bin/env python3
"""
AnchorCalib-TCN 训练脚本
=========================
TCN + 校准向量 → 个性化肌电激活比例预测。

设计目标:
  - 用户完成标准校准动作 → 提取个人校准向量
  - TCN 编码时序运动特征 (角度/角速度/角加速度/相位)
  - 融合校准向量 → 预测肌肉激活比例 (相对于个人锚点)
  - LOSO 交叉验证评估跨个体泛化能力
  - RTX 5090 满血训练 (torch.compile + AMP bf16 + 大 batch)

使用:
  python train.py --data_dir ../  --output_dir ./results/

兼容性:
  - use_hardware_emg=False: 从 .mat 原始 EMG 提取包络 (模拟硬件前端)
  - use_hardware_emg=True:  直接使用 ADC 包络电压 (新硬件采集数据时用)
"""

import os, sys, time, io, json, warnings, argparse
from datetime import datetime

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import convolve1d
from scipy.stats import pearsonr
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# 导入模型和损失
from model import AnchorCalibTCN, AnchorCalibMLP
from losses import CombinedLoss

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║                       CONFIGURATION                         ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, '..')          # 默认 .mat 数据目录
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')   # 输出目录

# ── 设备 ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
USE_COMPILE = torch.cuda.is_available()  # TCN 较大，compile 收益明显

# ── 数据模式 ──
USE_HARDWARE_EMG = False   # True: 新硬件ADC包络 | False: 从 .mat 原始EMG提取包络
ADC_VREF = 3.3             # ADC 参考电压 (V)
ADC_MAX = 4095             # ADC 最大值 (12-bit)

# ── 角度范围 ──
ANGLE_MAX = 180.0          # 真实肘关节角度范围 0-180° (伸直=180°, 曲肘≈30°)
                           # 老 Ninapro 数据用 130°, 新 RDK X5 数据用 180°

# ── 校准设置 ──
CALIB_DURATION_SEC = 20.0  # 只用前 N 秒数据提取校准特征 (防止泄漏到正式动作)
                           # 设为 0 表示使用全部数据 (仅适用于纯校准文件)

# ── 采样参数 ──
FS_RAW = 2000               # 原始采样率 (Hz)
DOWNSAMPLE = 20             # 降采样因子: 2000Hz → 100Hz
FS = FS_RAW // DOWNSAMPLE   # 训练采样率 100Hz
CH_BICEPS = 10              # 肱二头肌通道
CH_TRICEPS = 11             # 肱三头肌通道
N_MUSCLES = 2

# ── 窗口参数 ──
WINDOW_SIZE = 64            # 时间窗口 (帧), 64@100Hz = 0.64s
WINDOW_STRIDE = 4           # 窗口步长, stride=4 → ~25Hz 输出率

# ── 运动特征维度 ──
# [angle_norm, angular_vel, angular_acc, sin_angle, cos_angle, phase_0..4 (one-hot)]
N_PHASES = 5               # 相位类别数: rest/flexion/hold/extension/training
MOTION_DIM = 5 + N_PHASES  # 5 连续特征 + 5 one-hot 相位 = 10

# ── 校准特征维度 ──
# [b_rest, b_90, b_peak, b_auc, b_slope, b_cv90,
#  t_rest, t_90, t_peak, t_auc, t_slope, t_cv90,
#  height, weight, BMI, gender]
CALIB_DIM = 16

# ── 模型超参数 ──
TCN_CHANNELS = (128, 256, 256, 256)   # TCN 各层通道
TCN_KERNEL = 5                         # 卷积核大小
TCN_DROPOUT = 0.2                      # TCN dropout
CALIB_HIDDEN = (96, 192, 96)           # 校准编码器隐藏层
FUSION_HIDDEN = (384, 192)             # 融合层

# ── 训练超参数 ──
BATCH_SIZE = 2048                      # 批次大小 (RTX 5090 32GB 可开到 8192+)
NUM_WORKERS = 8                        # 数据加载线程
EPOCHS = 150                           # 训练轮数
LR_MAX = 3e-3                          # OneCycleLR 峰值学习率
WEIGHT_DECAY = 1e-3                    # AdamW 权重衰减
GRAD_CLIP = 2.0                        # 梯度裁剪

# ── 损失权重 ──
LAMBDA_MSE = 1.0
LAMBDA_PEARSON = 0.5
LAMBDA_MAE = 0.3

# ── 随机种子 ──
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║              SIGNAL PROCESSING (仅 use_hardware_emg=False)  ║
# ╚══════════════════════════════════════════════════════════════╝

def extract_emg_envelope(emg_signal, fs=2000):
    """
    从原始 EMG 提取包络。
    注意: 新硬件已在前端完成此步骤, 使用硬件ADC数据时跳过此函数。
    步骤: 50Hz陷波 → 20-450Hz带通 → 全波整流 → 6Hz低通
    """
    b_n, a_n = iirnotch(50, 30, fs)
    sig = filtfilt(b_n, a_n, emg_signal, axis=0)
    b_bp, a_bp = butter(4, [20 / (0.5 * fs), 450 / (0.5 * fs)], btype='band')
    sig = filtfilt(b_bp, a_bp, sig, axis=0)
    sig = np.abs(sig)
    b_lp, a_lp = butter(4, 6 / (0.5 * fs), btype='low')
    return filtfilt(b_lp, a_lp, sig, axis=0)


def light_smooth(signal, window=5):
    """轻度滑动平均 (硬件已处理过的ADC包络只需轻度平滑)."""
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def create_gaussian_kernel(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def find_data_matrix(obj):
    """递归查找 .mat 文件中的数值矩阵."""
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[0] > 100 and obj.dtype != 'O':
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype == 'O':
        for item in obj.flat:
            r = find_data_matrix(item)
            if r is not None:
                return r
    return None


def adc_to_voltage(adc, vref=ADC_VREF, adc_max=ADC_MAX):
    """ADC 数值 → 电压 (V). 前端硬件已输出放大/滤波/整流/平滑后的包络."""
    return adc.astype(np.float64) / adc_max * vref


def unwrap_model(model):
    """取回 torch.compile 包装前的原始模型 (用于保存和 ONNX 导出)."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


# ╔══════════════════════════════════════════════════════════════╗
# ║                     ANGLE EXTRACTION                        ║
# ╚══════════════════════════════════════════════════════════════╝

def extract_angle(mat, fs=2000):
    """
    从 .mat 数据中提取角度轨迹。
    优先: angle → inclin → restimulus
    返回: (angle_array, source_string)
    """
    if 'angle' in mat:
        d = find_data_matrix(mat['angle'])
        if d is not None:
            a = d.flatten().astype(np.float64)
            if a.max() > ANGLE_MAX * 1.2:
                # 数据范围异常大 → 归一化到 [0, ANGLE_MAX]
                a = (a - a.min()) / (a.max() - a.min() + 1e-9) * ANGLE_MAX
            return np.clip(a, 0, ANGLE_MAX), 'angle_sensor'

    if 'inclin' in mat:
        d = find_data_matrix(mat['inclin'])
        if d is not None:
            a = d.flatten().astype(np.float64)
            emg_raw = find_data_matrix(mat['emg'])
            b_env = extract_emg_envelope(emg_raw[:, CH_BICEPS], fs)
            corr = np.corrcoef(a, b_env)[0, 1]
            if not np.isnan(corr) and corr < 0:
                a = -a
            a = (a - a.min()) / (a.max() - a.min() + 1e-9) * ANGLE_MAX
            return a, 'inclinometer'

    if 'restimulus' in mat:
        restim = find_data_matrix(mat['restimulus']).flatten()
        kernel = create_gaussian_kernel(3000, 700)
        smooth = convolve1d((restim > 0).astype(float), kernel)
        a = (smooth / (smooth.max() + 1e-9)) * ANGLE_MAX
        return a, 'restimulus_derived'

    raise ValueError("No angle data found in .mat file.")


# ╔══════════════════════════════════════════════════════════════╗
# ║                  PHASE DETECTION                            ║
# ╚══════════════════════════════════════════════════════════════╝

def detect_phases(angle, angular_vel, fs=100):
    """
    从角度轨迹自动检测运动相位。
    Returns:
        phase_id: array (same length), values:
            0 = rest     (静息: 角度接近最大值 + 低速度)
            1 = flexion  (曲肘: 角速度明显为负)
            2 = hold     (保持: 角度≈90° + 低速度, 90°是解剖学中立位)
            3 = extension(伸展: 角速度明显为正)
            4 = training (其他/正式训练动作)
    """
    n = len(angle)
    phase = np.full(n, 4, dtype=np.int32)  # 默认 = training

    # 平滑角速度用于相位判定
    vel_abs = np.convolve(np.abs(angular_vel), np.ones(50) / 50, mode='same')

    max_angle = np.percentile(angle, 95)

    # 静息: 角度接近最大 + 角速度低
    rest_margin = ANGLE_MAX * 0.05  # 5% 容差
    rest_mask = (angle > max_angle - rest_margin) & (vel_abs < 5.0)
    phase[rest_mask] = 0

    # 90°保持: 角度接近90° + 角速度低 (90° 是肘关节中立位, 与 ANGLE_MAX 无关)
    hold_margin = 10.0
    hold_mask = (np.abs(angle - 90) < hold_margin) & (vel_abs < 5.0)
    phase[hold_mask] = 2

    # 曲肘: 角速度明显为负 (角度在减小)
    flexion_mask = (angular_vel < -10) & ~hold_mask & ~rest_mask
    phase[flexion_mask] = 1

    # 伸展: 角速度明显为正 (角度在增大)
    extension_mask = (angular_vel > 10) & ~hold_mask & ~rest_mask
    phase[extension_mask] = 3

    return phase


# ╔══════════════════════════════════════════════════════════════╗
# ║              CALIBRATION FEATURE EXTRACTION                 ║
# ╚══════════════════════════════════════════════════════════════╝

def extract_calibration_features(voltage_b, voltage_t, angle, angular_vel, phase, fs=100):
    """
    从检测到的相位中提取个人校准向量。

    Args:
        voltage_b:   肱二头肌包络电压 (已处理)
        voltage_t:   肱三头肌包络电压 (已处理)
        angle:       角度轨迹
        angular_vel: 角速度 (已平滑)
        phase:       相位标签 (0-4)
        fs:          采样率

    Returns:
        calib_vec: [16] 校准特征向量
        calib_meta: dict 诊断信息
    """
    eps = 1e-8
    max_angle_obs = float(np.percentile(angle, 95))  # 该受试者实际最大角度
    dt = 1.0 / fs

    def _extract_muscle(v, phase_arr):
        """提取单个肌肉通道的校准特征."""
        rest_data = v[phase_arr == 0]
        hold_data = v[phase_arr == 2]
        flex_data = v[phase_arr == 1]

        # 静息基线
        v_rest = float(np.median(rest_data)) if len(rest_data) > 10 else float(np.percentile(v, 10))
        v_rest_std = float(np.std(rest_data)) if len(rest_data) > 10 else 0.0

        # 90° 锚点
        if len(hold_data) > 5:
            # 取中间 50% 最稳定数据
            n_half = max(len(hold_data) // 4, 3)
            hold_sorted = np.sort(hold_data)
            hold_stable = hold_sorted[n_half:-n_half] if len(hold_sorted) > 2 * n_half else hold_data
            v_90 = float(np.mean(hold_stable))
            cv90 = float(np.std(hold_stable) / (v_90 + eps))
        else:
            # 兜底：取角度在 85-95° 的电压中位数
            near_90 = v[(angle > 85) & (angle < 95)]
            v_90 = float(np.median(near_90)) if len(near_90) > 5 else float(np.median(v))
            cv90 = float(np.std(near_90) / (v_90 + eps)) if len(near_90) > 5 else 0.1

        # 动态曲肘特征
        v_peak = float(np.max(flex_data)) if len(flex_data) > 0 else v_90
        # AUC = 电压-时间积分 (V·s), 除 fs 使结果与采样率无关
        auc = float(np.sum(flex_data) * dt) if len(flex_data) > 0 else v_90 * len(flex_data) * dt

        # 角度-肌电斜率 (线性拟合角度→电压)
        flex_mask = phase_arr == 1
        if flex_mask.sum() > 10:
            a_flex = angle[flex_mask]
            v_flex = v[flex_mask]
            A = np.vstack([a_flex, np.ones_like(a_flex)]).T
            slope, _ = np.linalg.lstsq(A, v_flex, rcond=None)[0]
            slope = float(slope)
        else:
            # 兜底: 用首尾两点估算
            slope = (v_90 - v_rest) / (90.0 - max_angle_obs + eps) if abs(max_angle_obs - 90.0) > 1 else 0.0

        return {
            'rest': v_rest,
            'rest_std': v_rest_std,
            'v90': v_90,
            'peak': v_peak,
            'auc': auc,
            'slope': slope,
            'cv90': cv90,
        }

    b = _extract_muscle(voltage_b, phase)
    t = _extract_muscle(voltage_t, phase)

    calib_vec = np.array([
        b['rest'], b['v90'], b['peak'], b['auc'], b['slope'], b['cv90'],
        t['rest'], t['v90'], t['peak'], t['auc'], t['slope'], t['cv90'],
        0.0,  # height (placeholder, filled by caller)
        0.0,  # weight
        0.0,  # BMI
        0.0,  # gender
    ], dtype=np.float32)

    calib_meta = {
        'biceps': b, 'triceps': t,
        'n_rest': int((phase == 0).sum()),
        'n_flex': int((phase == 1).sum()),
        'n_hold': int((phase == 2).sum()),
        'n_ext': int((phase == 3).sum()),
    }

    return calib_vec, calib_meta


# ╔══════════════════════════════════════════════════════════════╗
# ║              DATA LOADING & WINDOW CONSTRUCTION             ║
# ╚══════════════════════════════════════════════════════════════╝

def load_subject(mat_path):
    """
    加载单个受试者的 .mat 数据, 返回:
        features:   [N, MOTION_DIM]  运动特征 (100Hz)
        targets:    [N, 2]           肌电比例标签 [biceps_ratio, triceps_ratio]
        voltage:    [N, 2]           包络电压 [biceps_voltage, triceps_voltage]
        calib_vec:  [CALIB_DIM]      校准向量
        meta:       dict             元数据

    校准数据泄漏防护:
        只用前 CALIB_DURATION_SEC 秒的数据提取校准向量 (模拟真实"先校准后使用"流程).
        若 CALIB_DURATION_SEC=0 则用全部数据 (仅适用于纯校准文件).
    """
    mat = scipy.io.loadmat(mat_path)

    # ── 身体参数 ──
    gender = 1.0 if 'f' in str(mat.get('gender', 'm')).lower() else 0.0
    weight = float(mat['weight'][0][0]) if 'weight' in mat else 70.0
    height = float(mat['height'][0][0]) if 'height' in mat else 175.0
    bmi = weight / ((height / 100) ** 2)

    # ── 提取 EMG 包络 ──
    emg_raw = find_data_matrix(mat['emg'])
    if USE_HARDWARE_EMG:
        # 新硬件: ADC→电压→轻度平滑 (前端已完成放大/滤波/整流/包络)
        v_b = adc_to_voltage(emg_raw[:, CH_BICEPS])
        v_t = adc_to_voltage(emg_raw[:, CH_TRICEPS])
        v_b = light_smooth(v_b)
        v_t = light_smooth(v_t)
    else:
        # 旧 .mat: 原始EMG → 软件包络提取 (模拟硬件前端)
        v_b = extract_emg_envelope(emg_raw[:, CH_BICEPS], FS_RAW)
        v_t = extract_emg_envelope(emg_raw[:, CH_TRICEPS], FS_RAW)

    # ── 提取角度 ──
    angle, angle_source = extract_angle(mat, FS_RAW)

    # ── 降采样 2000Hz → 100Hz ──
    v_b = v_b[::DOWNSAMPLE].astype(np.float64)
    v_t = v_t[::DOWNSAMPLE].astype(np.float64)
    angle = angle[::DOWNSAMPLE].astype(np.float64)

    # ── 计算角速度 & 角加速度 ──
    dt = 1.0 / FS
    raw_vel = np.gradient(angle, dt)
    b_lp, a_lp = butter(2, 10 / (0.5 * FS), btype='low')
    angular_vel = filtfilt(b_lp, a_lp, raw_vel)
    angular_acc = np.gradient(angular_vel, dt)

    # ── 相位检测 ──
    phase = detect_phases(angle, angular_vel, FS)

    # ── 校准向量提取 (只使用前 CALIB_DURATION_SEC 秒, 防止泄漏) ──
    if CALIB_DURATION_SEC > 0:
        calib_cutoff = int(CALIB_DURATION_SEC * FS)
        calib_cutoff = min(calib_cutoff, len(v_b))
    else:
        calib_cutoff = len(v_b)

    v_b_calib = v_b[:calib_cutoff]
    v_t_calib = v_t[:calib_cutoff]
    angle_calib = angle[:calib_cutoff]
    av_calib = angular_vel[:calib_cutoff]
    phase_calib = phase[:calib_cutoff]

    calib_vec, calib_meta = extract_calibration_features(
        v_b_calib, v_t_calib, angle_calib, av_calib, phase_calib, FS,
    )
    calib_vec[12] = height
    calib_vec[13] = weight
    calib_vec[14] = bmi
    calib_vec[15] = gender

    # ── 构造比例标签 (使用全部数据) ──
    v_rest_b, v_90_b = calib_vec[0], calib_vec[1]
    v_rest_t, v_90_t = calib_vec[6], calib_vec[7]

    ratio_b = (v_b - v_rest_b) / (v_90_b - v_rest_b + 1e-8)
    ratio_t = (v_t - v_rest_t) / (v_90_t - v_rest_t + 1e-8)

    # 裁剪极端值
    ratio_b = np.clip(np.nan_to_num(ratio_b, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 5.0)
    ratio_t = np.clip(np.nan_to_num(ratio_t, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 5.0)

    # ── 构造运动特征 ──
    angle_norm = angle / ANGLE_MAX
    rad = np.deg2rad(angle)
    sin_angle = np.sin(rad)
    cos_angle = np.cos(rad)

    # One-hot 编码相位 (避免 StandardScaler 把离散值当连续值)
    phase_onehot = np.eye(N_PHASES, dtype=np.float32)[phase]

    features = np.column_stack([
        angle_norm,
        angular_vel,
        angular_acc,
        sin_angle,
        cos_angle,
        phase_onehot,
    ]).astype(np.float32)

    targets = np.column_stack([ratio_b, ratio_t]).astype(np.float32)
    voltages = np.column_stack([v_b, v_t]).astype(np.float32)

    meta = {
        'gender': gender, 'weight': weight, 'height': height, 'bmi': bmi,
        'v_rest_b': float(v_rest_b), 'v_90_b': float(v_90_b),
        'v_rest_t': float(v_rest_t), 'v_90_t': float(v_90_t),
        'angle_source': angle_source,
        'calib_meta': calib_meta,
        'calib_cutoff_sec': CALIB_DURATION_SEC,
        'n_samples': len(features),
    }

    return features, targets, voltages, calib_vec, meta


def build_windows(features, targets, calib_vec, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    从时间序列构建滑动窗口数据集。

    Args:
        features:  [N, MOTION_DIM]
        targets:   [N, 2]
        calib_vec: [CALIB_DIM]
        window_size: 窗口帧数
        stride:      滑动步长

    Returns:
        X_motion: [n_windows, window_size, MOTION_DIM]
        X_calib:  [n_windows, CALIB_DIM]  (每个窗口重复同一校准向量)
        Y:        [n_windows, 2]
    """
    n = len(features)
    if n < window_size:
        return None, None, None

    start_indices = np.arange(0, n - window_size + 1, stride)  # +1 确保最后一个合法窗口被取到
    n_windows = len(start_indices)

    # 预分配
    X_motion = np.zeros((n_windows, window_size, MOTION_DIM), dtype=np.float32)
    X_calib = np.tile(calib_vec.astype(np.float32), (n_windows, 1))
    Y = np.zeros((n_windows, 2), dtype=np.float32)

    for i, start in enumerate(start_indices):
        end = start + window_size
        X_motion[i] = features[start:end]
        Y[i] = targets[end - 1]  # 预测窗口最后一帧的肌电比例

    return X_motion, X_calib, Y


def load_all_subjects(data_dir):
    """自动发现 data_dir 下所有 *_E*_A*.mat 文件并加载."""
    import glob
    mat_files = sorted(glob.glob(os.path.join(data_dir, '*_E*_A*.mat')))

    if not mat_files:
        # 回退：尝试旧格式 S1-S10
        for s in range(1, 11):
            fp = os.path.join(data_dir, f'S{s}_E3_A1.mat')
            if os.path.exists(fp):
                mat_files.append(fp)

    subjects = {}
    print(f"\n  Loading {len(mat_files)} subjects from {data_dir}...")

    for fp in mat_files:
        fname = os.path.basename(fp).replace('.mat', '')

        try:
            features, targets, voltages, calib_vec, meta = load_subject(fp)
        except Exception as e:
            print(f"    [WARN] {fname}: load failed: {e}")
            continue

        Xm, Xc, Y = build_windows(features, targets, calib_vec)

        if Xm is None:
            print(f"    [WARN] {fname}: too few samples ({meta.get('n_samples', 0)}), skipping")
            continue

        # 用有序整数ID作为key，便于LOSO（sorted(keys())要求可比较）
        sid = len(subjects) + 1
        subjects[sid] = {
            'X_motion': Xm,
            'X_calib': Xc,
            'Y': Y,
            'voltages': voltages,
            'calib_vec': calib_vec,
            'meta': meta,
            'filename': fname,
        }

        v90b = meta.get('v_90_b', 0)
        v90t = meta.get('v_90_t', 0)
        print(f"    [{sid:02d}] {fname}: {meta['n_samples']:,} samples → {len(Xm):,} windows | "
              f"90° B={v90b:.0f} T={v90t:.0f} uV | "
              f"BMI={meta.get('bmi', 0):.1f} | {meta.get('angle_source', '?')}")

    print(f"  Total: {len(subjects)} subjects loaded\n")
    return subjects


# ╔══════════════════════════════════════════════════════════════╗
# ║                         DATASET                             ║
# ╚══════════════════════════════════════════════════════════════╝

class WindowDataset(Dataset):
    """时间窗口数据集."""

    def __init__(self, X_motion, X_calib, Y):
        self.Xm = torch.from_numpy(X_motion)
        self.Xc = torch.from_numpy(X_calib)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.Xm)

    def __getitem__(self, idx):
        return self.Xm[idx], self.Xc[idx], self.Y[idx]


# ╔══════════════════════════════════════════════════════════════╗
# ║                     TRAINING ONE ROUND                      ║
# ╚══════════════════════════════════════════════════════════════╝

def train_one_round(train_ds, val_ds=None, model_type='tcn'):
    """训练一轮 (单个 LOSO fold 或全量训练)."""
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=True, persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, BATCH_SIZE * 2, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    # ── 创建模型 ──
    if model_type == 'tcn':
        model = AnchorCalibTCN(
            motion_dim=MOTION_DIM, calib_dim=CALIB_DIM,
            window_size=WINDOW_SIZE,
            tcn_channels=TCN_CHANNELS, tcn_kernel=TCN_KERNEL,
            tcn_dropout=TCN_DROPOUT,
            calib_hidden=CALIB_HIDDEN,
            fusion_hidden=FUSION_HIDDEN,
        ).to(DEVICE)
    elif model_type == 'mlp':
        model = AnchorCalibMLP(
            motion_dim=MOTION_DIM, calib_dim=CALIB_DIM,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if USE_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  [torch.compile] enabled (reduce-overhead)")
        except Exception as e:
            print(f"  [torch.compile] failed: {e}, using eager mode")

    criterion = CombinedLoss(LAMBDA_MSE, LAMBDA_PEARSON, LAMBDA_MAE)
    opt = optim.AdamW(
        model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY,
        fused=True if DEVICE.type == 'cuda' else False,
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR_MAX, total_steps=steps_per_epoch * EPOCHS,
        pct_start=0.08, div_factor=25, final_div_factor=1000,
        anneal_strategy='cos',
    )
    grad_scaler = GradScaler('cuda') if USE_AMP else None

    best_val_loss = float('inf')
    best_state = None
    train_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        for bx_m, bx_c, by in train_loader:
            bx_m = bx_m.to(DEVICE, non_blocking=True)
            bx_c = bx_c.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.bfloat16) if USE_AMP else torch.no_grad() if False else torch.enable_grad():
                pred = model(bx_m, bx_c)
                total_tensor, loss_dict = criterion(pred, by)

            if USE_AMP:
                grad_scaler.scale(total_tensor).backward()
                grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                total_tensor.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            scheduler.step()
            epoch_losses.append(loss_dict)

        # ── Epoch 统计 ──
        avg_loss = np.mean([d['total'] for d in epoch_losses])
        avg_r_b = np.mean([d['r_biceps'] for d in epoch_losses])
        avg_r_t = np.mean([d['r_triceps'] for d in epoch_losses])

        # ── 验证 ──
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                with autocast('cuda', dtype=torch.bfloat16) if USE_AMP else torch.no_grad():
                    for bx_m, bx_c, by in val_loader:
                        bx_m = bx_m.to(DEVICE, non_blocking=True)
                        bx_c = bx_c.to(DEVICE, non_blocking=True)
                        by = by.to(DEVICE, non_blocking=True)
                        pred = model(bx_m, bx_c)
                        _, ld = criterion(pred, by)
                        val_losses.append(ld)
            val_loss = np.mean([d['total'] for d in val_losses])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            val_str = f"val={val_loss:.4f}" if val_loss is not None else ""
            print(f"  E{epoch+1:3d} | loss={avg_loss:.4f} | "
                  f"r_b={avg_r_b:.3f} r_t={avg_r_t:.3f} | {val_str} | lr={lr_now:.2e}")

        train_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'r_biceps': avg_r_b,
            'r_triceps': avg_r_t,
        })

    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_history


# ╔══════════════════════════════════════════════════════════════╗
# ║                       EVALUATION                            ║
# ╚══════════════════════════════════════════════════════════════╝

@torch.no_grad()
def evaluate_model(model, val_ds, voltages, calib_vec):
    """
    评估模型在验证集上的表现。
    返回比例预测指标 + 还原电压指标。
    """
    loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()

    yp_list, yt_list = [], []
    with autocast('cuda', dtype=torch.bfloat16) if USE_AMP else torch.no_grad():
        for bx_m, bx_c, by in loader:
            pred = model(bx_m.to(DEVICE, non_blocking=True),
                        bx_c.to(DEVICE, non_blocking=True))
            yp_list.append(pred.float().cpu().numpy())
            yt_list.append(by.float().numpy())

    r_pred = np.concatenate(yp_list)  # [N, 2] 预测比例
    r_true = np.concatenate(yt_list)  # [N, 2] 真实比例

    # 还原为 EMG 包络电压: V = r × (V_90 - V_rest) + V_rest
    v_rest_b, v_90_b = calib_vec[0], calib_vec[1]
    v_rest_t, v_90_t = calib_vec[6], calib_vec[7]

    scale = np.array([[v_90_b - v_rest_b, v_90_t - v_rest_t]], dtype=np.float32)
    offset = np.array([[v_rest_b, v_rest_t]], dtype=np.float32)

    r_pred_clipped = np.clip(r_pred, 0, None)

    emg_pred = r_pred_clipped * scale + offset
    emg_true = r_true * scale + offset

    def compute_metrics(yt, yp):
        a, b = yt.flatten(), yp.flatten()
        pr, _ = pearsonr(a, b)
        rmse = float(np.sqrt(mean_squared_error(a, b)))
        return {
            'R2': float(r2_score(a, b)),
            'MAE': float(mean_absolute_error(a, b)),
            'RMSE': rmse,
            'Pearson_r': float(pr),
            'NRMSE': rmse / (a.max() - a.min() + 1e-9) if a.max() > a.min() else 0.0,
        }

    results = {}
    for i, name in enumerate(['Biceps', 'Triceps']):
        results[f'{name}_ratio'] = compute_metrics(r_true[:, i], r_pred[:, i])
        results[f'{name}_emg'] = compute_metrics(emg_true[:, i], emg_pred[:, i])

    return results, r_pred, r_true, emg_pred, emg_true


# ╔══════════════════════════════════════════════════════════════╗
# ║                 LOSO CROSS-VALIDATION                       ║
# ╚══════════════════════════════════════════════════════════════╝

def run_loso(subjects, model_type='tcn'):
    """Leave-One-Subject-Out 交叉验证."""
    all_subjects = sorted(subjects.keys())
    loso_results = []

    print(f"\n{'='*75}")
    print(f"  LOSO Cross-Validation ({len(all_subjects)} rounds) — {model_type.upper()}")
    print(f"{'='*75}")
    header = (f"  {'Round':<8s} {'Val':<6s} {'Bic r':>8s} {'Bic R²':>8s} "
              f"{'Tri r':>8s} {'Tri R²':>8s} {'Time':>8s}")
    print(header)
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for val_s in all_subjects:
        train_subs = [s for s in all_subjects if s != val_s]
        t_start = time.time()

        # ── 组装训练集 (串联所有训练受试者的窗口) ──
        train_Xm = np.concatenate([subjects[s]['X_motion'] for s in train_subs])
        train_Xc = np.concatenate([subjects[s]['X_calib'] for s in train_subs])
        train_Y = np.concatenate([subjects[s]['Y'] for s in train_subs])

        # ── Scaler: 只在训练集上 fit ──
        # 对运动特征做标准化 (按窗口展平)
        n_train, T, D = train_Xm.shape
        motion_scaler = StandardScaler()
        motion_scaler.fit(train_Xm.reshape(-1, D))
        train_Xm_s = motion_scaler.transform(train_Xm.reshape(-1, D)).reshape(n_train, T, D).astype(np.float32)

        calib_scaler = StandardScaler()
        calib_scaler.fit(train_Xc)
        train_Xc_s = calib_scaler.transform(train_Xc).astype(np.float32)

        train_ds = WindowDataset(train_Xm_s, train_Xc_s, train_Y)

        # ── 组装验证集 ──
        val_data = subjects[val_s]
        n_val = len(val_data['X_motion'])
        val_Xm_s = motion_scaler.transform(
            val_data['X_motion'].reshape(-1, D)
        ).reshape(n_val, T, D).astype(np.float32)
        val_Xc_s = calib_scaler.transform(val_data['X_calib']).astype(np.float32)
        val_ds = WindowDataset(val_Xm_s, val_Xc_s, val_data['Y'])

        # ── 训练 ──
        model, history = train_one_round(train_ds, val_ds, model_type)

        # ── 评估 ──
        round_metrics, r_pred, r_true, emg_pred, emg_true = evaluate_model(
            model, val_ds, val_data['voltages'], val_data['calib_vec'],
        )
        round_metrics['val_subject'] = val_s
        round_metrics['train_subs'] = train_subs
        round_metrics['n_train_windows'] = len(train_ds)
        round_metrics['n_val_windows'] = len(val_ds)
        loso_results.append(round_metrics)

        # ── 打印 ──
        br = round_metrics['Biceps_emg']['Pearson_r']
        bR = round_metrics['Biceps_emg']['R2']
        tr = round_metrics['Triceps_emg']['Pearson_r']
        tR = round_metrics['Triceps_emg']['R2']
        elapsed = time.time() - t_start
        print(f"  Round {val_s:<3d}  S{val_s:<5d} {br:>8.4f} {bR:>8.4f} "
              f"{tr:>8.4f} {tR:>8.4f} {elapsed:>7.1f}s")

        # 清理 GPU
        del model, train_ds, val_ds
        torch.cuda.empty_cache()

    # ── 汇总 ──
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    avg = {}
    metric_keys = [k for k in loso_results[0].keys()
                   if k not in ('val_subject', 'train_subs', 'n_train_windows', 'n_val_windows')]
    for key in metric_keys:
        avg[key] = {}
        for m in loso_results[0][key]:
            vals = [r[key][m] for r in loso_results]
            avg[key][m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    # 打印汇总
    print(f"\n  LOSO Summary — EMG voltage prediction (mean ± std):")
    print(f"  {'Muscle':<10s} {'R²':>14s} {'Pearson r':>14s} {'RMSE(uV)':>12s} {'NRMSE':>10s}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*12} {'-'*10}")
    for m_name in ['Biceps_emg', 'Triceps_emg']:
        label = m_name.replace('_emg', '')
        r2 = avg[m_name]['R2']
        pr = avg[m_name]['Pearson_r']
        rmse = avg[m_name]['RMSE']
        nrmse = avg[m_name]['NRMSE']
        print(f"  {label:<10s} {r2['mean']:>+.4f}±{r2['std']:.4f}  "
              f"{pr['mean']:>+.4f}±{pr['std']:.4f}  "
              f"{rmse['mean']:>8.1f}±{rmse['std']:.1f}  "
              f"{nrmse['mean']:>.4f}±{nrmse['std']:.4f}")

    return loso_results, avg


# ╔══════════════════════════════════════════════════════════════╗
# ║                   FINAL MODEL TRAINING                      ║
# ╚══════════════════════════════════════════════════════════════╝

def train_final(subjects, output_dir, model_type='tcn'):
    """全量训练最终部署模型."""
    print(f"\n{'='*60}")
    print(f"  Training FINAL {model_type.upper()} model (all subjects)")
    print(f"{'='*60}")

    all_subs = sorted(subjects.keys())
    train_Xm = np.concatenate([subjects[s]['X_motion'] for s in all_subs])
    train_Xc = np.concatenate([subjects[s]['X_calib'] for s in all_subs])
    train_Y = np.concatenate([subjects[s]['Y'] for s in all_subs])

    n_total, T, D = train_Xm.shape

    # Scaler
    motion_scaler = StandardScaler()
    motion_scaler.fit(train_Xm.reshape(-1, D))
    train_Xm_s = motion_scaler.transform(train_Xm.reshape(-1, D)).reshape(n_total, T, D).astype(np.float32)

    calib_scaler = StandardScaler()
    calib_scaler.fit(train_Xc)
    train_Xc_s = calib_scaler.transform(train_Xc).astype(np.float32)

    train_ds = WindowDataset(train_Xm_s, train_Xc_s, train_Y)

    # 训练
    model, history = train_one_round(train_ds, None, model_type)

    # ── 保存模型 (unwrap 避免 torch.compile 的 _orig_mod 前缀) ──
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'anchorcalib_{model_type}.pt')
    raw_model = unwrap_model(model)
    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'config': {
            'model_type': model_type,
            'motion_dim': MOTION_DIM,
            'calib_dim': CALIB_DIM,
            'window_size': WINDOW_SIZE,
            'tcn_channels': list(TCN_CHANNELS),
            'calib_hidden': list(CALIB_HIDDEN),
            'fusion_hidden': list(FUSION_HIDDEN),
            'output': 'muscle_activation_ratio [biceps, triceps]',
            'recovery': 'V_emg = ratio × (V_90 - V_rest) + V_rest',
        }
    }, model_path)
    print(f"  Model saved: {model_path}")

    # ── 保存 scaler ──
    joblib.dump(motion_scaler, os.path.join(output_dir, 'motion_scaler.pkl'))
    joblib.dump(calib_scaler, os.path.join(output_dir, 'calib_scaler.pkl'))
    print(f"  Scalers saved: motion_scaler.pkl, calib_scaler.pkl")

    # ── 保存校准配置 ──
    calib_config = {
        'feature_names': [
            'biceps_rest', 'biceps_90', 'biceps_peak', 'biceps_auc', 'biceps_slope', 'biceps_cv90',
            'triceps_rest', 'triceps_90', 'triceps_peak', 'triceps_auc', 'triceps_slope', 'triceps_cv90',
            'height', 'weight', 'BMI', 'gender',
        ],
        'recovery_formula': 'V_emg = ratio × (V_90 - V_rest) + V_rest',
        'motion_features': [
            'angle_norm', 'angular_vel', 'angular_acc', 'sin_angle', 'cos_angle',
            'phase_rest', 'phase_flexion', 'phase_hold', 'phase_extension', 'phase_training',
        ],
        'n_phases': N_PHASES,
        'angle_max': ANGLE_MAX,
        'window_size': WINDOW_SIZE,
        'window_stride': WINDOW_STRIDE,
        'fs': FS,
        'adc_vref': ADC_VREF if USE_HARDWARE_EMG else None,
        'adc_max': ADC_MAX if USE_HARDWARE_EMG else None,
    }
    with open(os.path.join(output_dir, 'calibration_config.json'), 'w', encoding='utf-8') as f:
        json.dump(calib_config, f, indent=2, ensure_ascii=False)
    print(f"  Config saved: calibration_config.json")

    return model, motion_scaler, calib_scaler


# ╔══════════════════════════════════════════════════════════════╗
# ║                      ONNX EXPORT                            ║
# ╚══════════════════════════════════════════════════════════════╝

def export_onnx(model, output_dir, model_type='tcn'):
    """导出 ONNX 模型供 RDK X5 部署."""
    print(f"\n  Exporting ONNX model for RDK X5...")

    # 取回 compile 前的原始模型, 避免 _orig_mod 前缀和导出兼容性问题
    raw_model = unwrap_model(model)
    raw_model.eval()

    # ONNX 导出在 CPU 上更稳定, 记录原设备以便恢复
    orig_device = next(raw_model.parameters()).device
    raw_model = raw_model.to('cpu')

    # 构造示例输入
    dummy_motion = torch.randn(1, WINDOW_SIZE, MOTION_DIM)
    dummy_calib = torch.randn(1, CALIB_DIM)

    onnx_path = os.path.join(output_dir, f'anchorcalib_{model_type}.onnx')

    try:
        torch.onnx.export(
            raw_model,
            (dummy_motion, dummy_calib),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['motion_sequence', 'calibration_vector'],
            output_names=['muscle_activation_ratio'],
            dynamic_axes={
                'motion_sequence': {0: 'batch_size'},
                'calibration_vector': {0: 'batch_size'},
                'muscle_activation_ratio': {0: 'batch_size'},
            },
        )
        print(f"  ONNX exported: {onnx_path}")
        print(f"    Input 1 'motion_sequence':    [B, {WINDOW_SIZE}, {MOTION_DIM}]")
        print(f"    Input 2 'calibration_vector': [B, {CALIB_DIM}]")
        print(f"    Output  'muscle_activation_ratio': [B, 2]")
    except Exception as e:
        print(f"  [WARN] ONNX export failed: {e}")
        print(f"  (Try with --no-compile if this persists.)")
    finally:
        # 恢复模型到原设备
        raw_model = raw_model.to(orig_device)


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    global USE_COMPILE, USE_AMP, WINDOW_SIZE, WINDOW_STRIDE
    global BATCH_SIZE, EPOCHS, LR_MAX, USE_HARDWARE_EMG, CALIB_DURATION_SEC

    parser = argparse.ArgumentParser(description='AnchorCalib-TCN Training')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR, help='.mat data directory')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--model', default='tcn', choices=['tcn', 'mlp'],
                        help='Model type (tcn=TCN, mlp=MLP baseline)')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LR_MAX, help='Peak learning rate')
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE, help='Window size')
    parser.add_argument('--stride', type=int, default=WINDOW_STRIDE, help='Window stride')
    parser.add_argument('--hardware-emg', action='store_true',
                        help='Use hardware-processed ADC envelope (skip software EMG filtering)')
    parser.add_argument('--calib-duration', type=float, default=CALIB_DURATION_SEC,
                        help='Seconds of beginning data used for calibration extraction (0=all data)')
    parser.add_argument('--skip-losov', action='store_true', help='Skip LOSO, train final only')
    parser.add_argument('--skip-onnx', action='store_true', help='Skip ONNX export')
    args = parser.parse_args()

    # 应用命令行覆盖全局配置
    if args.no_compile:
        USE_COMPILE = False
    if args.no_amp:
        USE_AMP = False
    if args.hardware_emg:
        USE_HARDWARE_EMG = True
    WINDOW_SIZE = args.window_size
    WINDOW_STRIDE = args.stride
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR_MAX = args.lr
    CALIB_DURATION_SEC = args.calib_duration

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # ── Header ──
    print("=" * 65)
    print("  AnchorCalib-TCN: Calibration-Guided EMG Prediction")
    print(f"  PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  AMP: {USE_AMP} | Compile: {USE_COMPILE} | "
              f"Batch: {args.batch_size} | Epochs: {args.epochs}")
    print(f"  Window: {WINDOW_SIZE}@100Hz={WINDOW_SIZE/100:.2f}s | Stride: {WINDOW_STRIDE}")
    print(f"  Model: {args.model.upper()} | Hardware EMG: {USE_HARDWARE_EMG}")
    print("=" * 65)

    # ── 1. 加载数据 ──
    print("\n[1/4] Loading data...")
    subjects = load_all_subjects(args.data_dir)

    if len(subjects) < 3:
        print(f"  [ERROR] Need at least 3 subjects, got {len(subjects)}. Check --data_dir.")
        sys.exit(1)

    # ── 2. LOSO 交叉验证 ──
    if not args.skip_losov:
        print(f"\n[2/4] LOSO Cross-Validation ({args.model.upper()})...")
        loso_results, loso_avg = run_loso(subjects, args.model)

        # 保存 LOSO 结果
        with open(os.path.join(args.output_dir, 'loso_results.json'), 'w', encoding='utf-8') as f:
            json.dump(loso_results, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(args.output_dir, 'loso_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(loso_avg, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  LOSO results saved.")
    else:
        print(f"\n[2/4] LOSO skipped (--skip-losov).")

    # ── 3. 全量训练最终模型 ──
    print(f"\n[3/4] Training final deployment model...")
    final_model, motion_scaler, calib_scaler = train_final(subjects, args.output_dir, args.model)

    # ── 4. ONNX 导出 ──
    if not args.skip_onnx:
        print(f"\n[4/4] Exporting ONNX...")
        export_onnx(final_model, args.output_dir, args.model)
    else:
        print(f"\n[4/4] ONNX export skipped (--skip-onnx).")

    # ── Done ──
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*65}")
    print(f"  ALL DONE! Total: {elapsed:.1f} min")
    print(f"  Output: {args.output_dir}/")
    print(f"  Files: anchorcalib_{args.model}.pt, *_scaler.pkl, calibration_config.json")
    if not args.skip_onnx:
        print(f"         anchorcalib_{args.model}.onnx")
    if not args.skip_losov:
        print(f"         loso_results.json, loso_summary.json")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
