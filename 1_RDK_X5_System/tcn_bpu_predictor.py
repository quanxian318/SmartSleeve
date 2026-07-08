#!/usr/bin/env python3
"""
BPU TCN Predictor — replacement for ONNX Runtime TCNPredictor
==============================================================
Same API as the original TCNPredictor, but uses BPU .bin model
instead of onnxruntime.

Usage (identical to original):
    predictor = TCNPredictor(bin_path, motion_scaler_path, calib_scaler_path, config_path)
    ratios = predictor.predict(angle, calib_vec)  # -> [biceps_ratio, triceps_ratio]
"""

import json, os, time, math, subprocess, shutil
import numpy as np
import joblib

ANGLE_MAX = 180.0
WINDOW_SIZE = 64
BPU_INPUT_SCALE = 0.00787402
BPU_OUTPUT_SCALE = 0.00787402

# Output file that hrt_model_exec generates
OUTPUT_DUMP = "/root/model_infer_output_0_emg_ratio.txt"
TMP_INPUT = "/dev/shm/bpu_emg_in.bin"


class TCNPredictor:
    """AnchorCalib-TCN BPU inference — drop-in replacement for ONNX version."""

    def __init__(self, bin_path, motion_scaler_path, calib_scaler_path, config_path):
        self.bin_path = bin_path
        self.ms = joblib.load(motion_scaler_path)
        self.cs = joblib.load(calib_scaler_path)
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.motion_dim = len(self.cfg['motion_features'])
        self.calib_dim = len(self.cfg['feature_names'])

        # 64-frame sliding window
        self.buffer = []
        self.prev_angle = None
        self.prev_vel = 0.0
        self.prev_time = None

        # Clean up stale dump
        if os.path.exists(OUTPUT_DUMP):
            os.remove(OUTPUT_DUMP)

        print(f"[TCN-BPU] Model={bin_path} motion={self.motion_dim}d calib={self.calib_dim}d")
        print(f"[TCN-BPU] Input scale={BPU_INPUT_SCALE}, output scale={BPU_OUTPUT_SCALE}")

    def _build_motion(self, angle):
        """Build motion feature vector: [angle_norm, vel, acc, sin, cos, phase(5)]"""
        angle = max(0.0, min(ANGLE_MAX, angle))
        a_norm = angle / ANGLE_MAX
        rad = math.radians(angle)
        sin_a = math.sin(rad)
        cos_a = math.cos(rad)

        now = time.time()
        if self.prev_angle is not None and self.prev_time is not None:
            dt = max(now - self.prev_time, 0.001)
            vel = (angle - self.prev_angle) / dt
            acc = (vel - self.prev_vel) / dt
        else:
            vel = 0.0
            acc = 0.0
        self.prev_angle = angle
        self.prev_vel = vel
        self.prev_time = now

        # Phase detection
        if abs(vel) < 3:
            phase = 0  # rest
        elif vel > 0:
            phase = 1  # extension
        else:
            phase = 3  # flexion
        phase_oh = np.eye(5, dtype=np.float32)[phase]

        return np.array([a_norm, vel, acc, sin_a, cos_a] + phase_oh.tolist(), dtype=np.float32)

    def predict(self, angle, calib_vec):
        """
        Args:
            angle:     float, current elbow joint angle (0-180 degrees)
            calib_vec: [16] float, calibration vector
        Returns:
            [2] float64 [biceps_ratio, triceps_ratio]
        """
        # 1. Build motion features and maintain sliding window
        feat = self._build_motion(angle)
        self.buffer.append(feat)
        if len(self.buffer) > WINDOW_SIZE:
            self.buffer.pop(0)

        if len(self.buffer) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(self.buffer), self.motion_dim), dtype=np.float32)
            window = np.vstack([pad, np.array(self.buffer, dtype=np.float32)])
        else:
            window = np.array(self.buffer, dtype=np.float32)

        # 2. StandardScaler normalization
        motion_scaled = self.ms.transform(
            window.reshape(-1, self.motion_dim)
        ).reshape(1, WINDOW_SIZE, self.motion_dim).astype(np.float32)

        calib_scaled = self.cs.transform(
            np.array(calib_vec, dtype=np.float64).reshape(1, -1)
        ).astype(np.float32)

        # 3. Merge into BPU input format [1, 26, 1, 64]
        motion_4d = motion_scaled.transpose(0, 2, 1)[:, :, np.newaxis, :]  # [1, 10, 1, 64]
        calib_4d = np.tile(calib_scaled[:, :, np.newaxis, np.newaxis],
                          (1, 1, 1, WINDOW_SIZE))                          # [1, 16, 1, 64]
        merged = np.concatenate([motion_4d, calib_4d], axis=1).astype(np.float32)

        # 4. Quantize to INT8
        int8_input = np.clip(np.round(merged / BPU_INPUT_SCALE), -128, 127).astype(np.int8)
        int8_input.tofile(TMP_INPUT)

        # 5. BPU inference
        if os.path.exists(OUTPUT_DUMP):
            os.remove(OUTPUT_DUMP)

        subprocess.run([
            "hrt_model_exec", "infer",
            "--model_file", self.bin_path,
            "--input_file", TMP_INPUT,
            "--enable_dump", "true",
            "--hybrid_dequantize_process", "true",
            "--dump_format", "txt",
        ], capture_output=True, timeout=10)

        # 6. Parse dequantized output
        if not os.path.exists(OUTPUT_DUMP):
            raise RuntimeError("BPU output dump not found")

        with open(OUTPUT_DUMP) as f:
            vals = f.read().strip().split()
        ratios = np.array([float(v) for v in vals], dtype=np.float64)

        return ratios.clip(0.0, None)  # same clip as original


# ============================== Test ==============================
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--bin', default='/root/anchorcalib_tcn_bpu_v2.bin')
    p.add_argument('--motion_scaler', default='/root/motion_scaler_63subj.pkl')
    p.add_argument('--calib_scaler', default='/root/calib_scaler_63subj.pkl')
    p.add_argument('--config', default='/root/calibration_config_63subj.json')
    args = p.parse_args()

    print("=" * 50)
    print("  BPU TCN Predictor — Test")
    print("=" * 50)

    model = TCNPredictor(args.bin, args.motion_scaler, args.calib_scaler, args.config)
    print("[OK] Model loaded")

    # Build dummy calib vector
    calib = np.array([200, 500, 600, 10000, 2.0, 0.1,
                      80, 120, 150, 2000, 1.5, 0.15,
                      170.0, 70.0, 22.0, 0.0], dtype=np.float64)

    # Simulate motion for 70+ frames
    print("Simulating 70 frames of motion...")
    for i in range(70):
        angle = 180.0 if i < 10 else 180.0 - (i - 10) * 3
        angle = max(30, angle)
        result = model.predict(angle, calib)

    print(f"Final output: biceps_ratio={result[0]:.4f}, triceps_ratio={result[1]:.4f}")
    print("[OK] BPU TCN Predictor working!")
