"""缓存 .mat → .npy 加速后续测试"""
import sys, os, numpy as np
sys.path.insert(0, '/root/autodl-tmp/anchorcalib_tcn')
from train import load_subject, build_windows

for s in range(1, 11):
    f = f'data/S{s}_E3_A1.mat'
    feat, targ, volt, calib, meta = load_subject(f)
    Xm, Xc, Y = build_windows(feat, targ, calib)
    out = dict(X_motion=Xm, X_calib=Xc, Y=Y, calib_vec=calib, voltages=volt, meta=meta)
    np.save(f.replace('.mat', '.mat.npy'), out)
    print(f'S{s}: {len(Xm):,} windows cached')
print('Cache done!')
