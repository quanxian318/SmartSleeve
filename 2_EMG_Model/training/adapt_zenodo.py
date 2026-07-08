"""
适配 Zenodo EMG Elbow Dataset → 统一 .mat 格式
=================================================
数据集: https://zenodo.org/records/7946782
10 subjects, biceps(Ch1) + triceps(Ch2) + elbow angle

原始格式: .txt 制表符分隔, 5列:
  col0: raw sEMG Ch1 (biceps)
  col1: raw sEMG Ch2 (triceps)
  col2: filtered sEMG Ch1
  col3: filtered sEMG Ch2
  col4: joint angle (degree)

统一输出 .mat:
  emg:    [samples, 12]  biceps@idx10, triceps@idx11 (与现有 S1-S10 兼容)
  angle:  [samples]      肘关节角度 (0-180°)
  gender: 'm' or 'f'
  weight: float (kg)
  height: float (cm)
"""
import os, sys
import numpy as np
import scipy.io

# ── 配置 ──
ZENODO_DIR = r"D:\emg_datasets\zenodo_emg_elbow"
OUT_DIR = r"D:\emg_datasets\unified"
FS_ASSUMED = 2000  # 假设原始采样率 2000Hz (与现有数据一致)

os.makedirs(OUT_DIR, exist_ok=True)


def read_subject_info(subj_dir):
    """读取 subject_info.txt"""
    info_path = os.path.join(subj_dir, "subject_info.txt")
    info = {}
    with open(info_path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
    return {
        'id': int(info.get('id', 0)),
        'age': int(info.get('age', 25)),
        'height': float(info.get('height', 175)),
        'weight': float(info.get('weight', 70)),
        'sex': int(info.get('sex', 1)),  # 1=male, 0=female
        'arm_length': float(info.get('arm_length', 0.3)),
    }


def load_trial(filepath):
    """加载单个 trial 的 .txt 文件, 返回 (emg_raw, angle)"""
    data = np.loadtxt(filepath, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 5)
    emg_b = data[:, 0]  # raw biceps
    emg_t = data[:, 1]  # raw triceps
    angle = data[:, 4]  # joint angle
    return emg_b, emg_t, angle


def normalize_angle(angle, target_max=180.0):
    """将角度映射到 [0, target_max]"""
    amin, amax = angle.min(), angle.max()
    if amax <= target_max * 1.1 and amin >= -5:
        # Already in reasonable range
        return np.clip(angle, 0, target_max)
    # Normalize
    return (angle - amin) / (amax - amin + 1e-9) * target_max


def process_zenodo_subject(subj_num):
    """处理单个 Zenodo 受试者, 返回 dict 用于保存 .mat"""
    subj_dir = os.path.join(ZENODO_DIR, f"subject {subj_num}")
    info = read_subject_info(subj_dir)

    all_emg_b, all_emg_t, all_angle = [], [], []

    for fname in sorted(os.listdir(subj_dir)):
        if not fname.endswith('.txt') or fname == 'subject_info.txt':
            continue

        # 只使用屈伸 (flex) 数据, 跳过旋前/旋后 (pronsup)
        if 'pronsup' in fname:
            continue

        filepath = os.path.join(subj_dir, fname)
        print(f"    {fname}")

        emg_b, emg_t, angle = load_trial(filepath)
        all_emg_b.append(emg_b)
        all_emg_t.append(emg_t)
        all_angle.append(angle)

    # 拼接所有 trial
    emg_b = np.concatenate(all_emg_b)
    emg_t = np.concatenate(all_emg_t)
    angle = np.concatenate(all_angle)

    # 规范化角度
    angle = normalize_angle(angle, 180.0)

    # 构造 12 通道 emg 矩阵 (兼容现有 S1-S10 的通道布局)
    # CH_BICEPS=10, CH_TRICEPS=11
    n_samples = len(emg_b)
    emg = np.zeros((n_samples, 12), dtype=np.float32)
    emg[:, 10] = emg_b
    emg[:, 11] = emg_t

    # 性别
    gender = 'm' if info['sex'] == 1 else 'f'

    print(f"    → {n_samples} samples, angle [{angle.min():.0f}°, {angle.max():.0f}°], "
          f"gender={gender}, {info['weight']}kg, {info['height']}cm")

    return {
        'emg': emg,
        'angle': angle.astype(np.float32).reshape(-1, 1),  # (N,1) 满足 find_data_matrix(shape[0]>100)
        'gender': gender,
        'weight': info['weight'],
        'height': info['height'],
        'description': f'Zenodo_EMG_Elbow_Subject_{subj_num:02d}',
        'source': 'zenodo_emg_elbow',
    }


def main():
    for s in range(1, 11):
        print(f"\nSubject {s:02d}:")
        try:
            data = process_zenodo_subject(s)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        # 保存为统一 .mat
        fname = f"Z{s:02d}_E1_A1.mat"
        out_path = os.path.join(OUT_DIR, fname)
        scipy.io.savemat(out_path, data, do_compression=False)

        # 验证
        check = scipy.io.loadmat(out_path)
        emg_shape = check['emg'].shape
        angle_shape = check['angle'].shape
        print(f"  [OK] Saved: {fname} | emg{emg_shape} angle{angle_shape} | "
              f"{os.path.getsize(out_path)/1024/1024:.1f} MB")

    print(f"\nDone! Output: {OUT_DIR}")
    print(f"Files: {sorted(os.listdir(OUT_DIR))}")


if __name__ == '__main__':
    main()
