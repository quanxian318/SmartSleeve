"""
适配 Lucchetti et al. EMG+Kinematics 数据集 → 统一 .mat 格式
=============================================================
数据集: Nature Scientific Data (2025), FigShare collection 7720187
  10 HS (healthy) + 10 ST (post-stroke) subjects
  12 EMG channels @ 1000Hz, 19 angle channels @ 125Hz
  6 tasks per subject (reaching, grasping, lifting, etc.)

输入: external_data/lucchetti/HS01.mat ~ HS10.mat, ST01.mat ~ ST10.mat
输出: D:/emg_datasets/unified/L01_E1_A1.mat ~ L20_E1_A1.mat

通道映射 (来自论文):
  EMG ch0  = Biceps Brachii        → 肱二头肌
  EMG ch1  = Triceps Brachii       → 肱三头肌
  Angle ch3 = Elbow Flex-Ext       → 肘关节屈伸角度

注意: EMG 数据已标准化 (zero-mean, small variance)，非原始电压。
"""
import os, sys
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

# ── 配置 ──
INPUT_DIR = r"D:\shuzinuansheng\train_emg\train_emg\anchorcalib_tcn\external_data\lucchetti"
OUT_DIR = r"D:\emg_datasets\unified"
TARGET_FS = 2000  # 统一采样到 2000Hz

os.makedirs(OUT_DIR, exist_ok=True)


def upsample_1d(data, fs_from, fs_to):
    """线性插值升采样"""
    n = len(data)
    t_old = np.arange(n) / fs_from
    t_new = np.arange(0, t_old[-1], 1.0 / fs_to)
    f = interp1d(t_old, data, kind='linear', bounds_error=False,
                 fill_value=(data[0], data[-1]))
    return f(t_new).astype(np.float32)


def process_lucchetti_subject(mat_path, subj_label):
    """处理单个 Lucchetti 受试者"""
    m = scipy.io.loadmat(mat_path)
    s = m['s'][0, 0]

    # 身体参数
    gender = str(s['Gender'][0]) if s['Gender'].size > 0 else 'M'
    gender = 'f' if 'F' in gender.upper() else 'm'
    bmi = float(s['BMI'][0, 0])
    height = 170.0
    weight = bmi * ((height / 100) ** 2)

    # 检测数据结构: HS=DataULdom, ST=DataULnonpleg(non-affected) 或 DataULpleg(affected)
    field_names = list(s.dtype.names)
    data_field = None
    for fname in ['DataULdom', 'DataULnonpleg', 'DataULpleg']:
        if fname in field_names:
            data_field = fname
            break
    if data_field is None:
        raise ValueError(f'Unknown data field. Available: {field_names}')

    tasks = s[data_field][0]
    emg_fs = float(s['EmgFreq'][0, 0])  # 1000 Hz
    kin_fs = float(s['KinFreq'][0, 0])  # 125 Hz

    all_emg_b, all_emg_t, all_angle = [], [], []

    for ti in range(len(tasks)):
        task = tasks[ti]
        emg = task['EMG']      # (12, N_emg)
        ang = task['Angles']   # (19, N_ang)

        # 提取通道
        emg_b = emg[0, :].astype(np.float64)   # Biceps
        emg_t = emg[1, :].astype(np.float64)   # Triceps
        elbow_ang = ang[3, :].astype(np.float64)  # Elbow flex-ext

        # 过滤无效值
        emg_b = np.nan_to_num(emg_b, nan=0.0)
        emg_t = np.nan_to_num(emg_t, nan=0.0)
        elbow_ang = np.nan_to_num(elbow_ang, nan=elbow_ang[~np.isnan(elbow_ang)].mean() if np.any(~np.isnan(elbow_ang)) else 90.0)

        # 升采样角度: 125 → 1000 Hz (先跟EMG对齐)
        ang_upsampled = upsample_1d(elbow_ang, kin_fs, emg_fs)

        # 截断到相同长度
        min_len = min(len(emg_b), len(ang_upsampled))
        emg_b = emg_b[:min_len]
        emg_t = emg_t[:min_len]
        ang_upsampled = ang_upsampled[:min_len]

        all_emg_b.append(emg_b)
        all_emg_t.append(emg_t)
        all_angle.append(ang_upsampled)

    # 拼接所有任务
    emg_b = np.concatenate(all_emg_b)
    emg_t = np.concatenate(all_emg_t)
    angle = np.concatenate(all_angle)

    # 升采样到 2000Hz
    emg_b = upsample_1d(emg_b, emg_fs, TARGET_FS)
    emg_t = upsample_1d(emg_t, emg_fs, TARGET_FS)
    angle = upsample_1d(angle, emg_fs, TARGET_FS)

    # 规范化角度到 [0, 180]
    angle = np.clip(angle, 0, 180)

    # 构造 12 通道 EMG (biceps@10, triceps@11)
    n_samples = len(emg_b)
    emg = np.zeros((n_samples, 12), dtype=np.float32)
    emg[:, 10] = emg_b
    emg[:, 11] = emg_t

    print(f"    → {n_samples:,} samples, angle [{angle.min():.0f}°, {angle.max():.0f}°], "
          f"gender={gender}, BMI={bmi:.1f}")

    return {
        'emg': emg,
        'angle': angle.astype(np.float32).reshape(-1, 1),
        'gender': gender,
        'weight': float(weight),
        'height': float(height),
        'description': f'Lucchetti_{subj_label}',
        'source': 'lucchetti_2025',
    }


def main():
    # HS01-HS10
    hs_files = [f'HS{i:02d}.mat' for i in range(1, 11)]
    # ST01-ST10
    st_files = [f'ST{i:02d}.mat' for i in range(1, 11)]

    counter = 0
    for label_prefix, files in [('HS', hs_files), ('ST', st_files)]:
        for fname in files:
            mat_path = os.path.join(INPUT_DIR, fname)
            if not os.path.exists(mat_path):
                print(f"  [SKIP] {fname} not found")
                continue

            counter += 1
            subj_label = fname.replace('.mat', '')
            print(f"\n{counter:02d}. {fname}:")
            try:
                data = process_lucchetti_subject(mat_path, subj_label)
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback; traceback.print_exc()
                continue

            out_fname = f'L{counter:02d}_E1_A1.mat'
            out_path = os.path.join(OUT_DIR, out_fname)
            scipy.io.savemat(out_path, data, do_compression=False)

            check = scipy.io.loadmat(out_path)
            print(f"  [OK] Saved: {out_fname} | emg{check['emg'].shape} "
                  f"angle{check['angle'].shape} | {os.path.getsize(out_path)/1024/1024:.1f} MB")

    print(f"\nDone! {counter} Lucchetti subjects converted.")


if __name__ == '__main__':
    main()
