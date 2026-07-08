#!/usr/bin/env python3
"""
RDK X5 EMG BPU Inference — Complete Deployment
================================================
Preprocessing: StandardScaler -> INT8 quantization
BPU Inference:  hrt_model_exec infer (single frame)
Postprocessing: INT8 dequantization -> EMG ratios

Usage:
    python3 emg_deploy.py                          # Quick test
    python3 emg_deploy.py --input window.npy        # Single inference
    python3 emg_deploy.py --bench 200               # Benchmark
"""

import numpy as np
import subprocess
import os, sys, time, joblib

# Config
MODEL_PATH   = "/root/anchorcalib_tcn_bpu_v2.bin"
MOTION_SCALER = "/root/motion_scaler.pkl"
CALIB_SCALER  = "/root/calib_scaler.pkl"
INPUT_SCALE   = 0.00787402
OUTPUT_SCALE  = 0.00787402
MOTION_DIM    = 10
CALIB_DIM     = 16
WINDOW_SIZE   = 64
FRAME_SIZE    = 1664  # 1 * 26 * 1 * 64 bytes
OUTPUT_FILE   = "/root/model_infer_output_0_emg_ratio.txt"


class EMGPredictor:
    """RDK X5 EMG inference with preprocessing."""

    def __init__(self):
        self.motion_scaler = joblib.load(MOTION_SCALER)
        self.calib_scaler = joblib.load(CALIB_SCALER)
        self._warmup()

    def _warmup(self):
        """Pre-load model (hrt_model_exec loads each time, but first call is slowest)."""
        x = np.random.randn(1, 26, 1, 64).astype(np.float32)
        self._infer_raw(x)

    def _infer_raw(self, f32_input):
        """Raw INT8 inference."""
        int8_data = np.clip(np.round(f32_input / INPUT_SCALE), -128, 127).astype(np.int8)
        in_path = "/dev/shm/emg_in.bin"
        int8_data.tofile(in_path)

        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)

        subprocess.run([
            "hrt_model_exec", "infer",
            "--model_file", MODEL_PATH,
            "--input_file", in_path,
            "--enable_dump", "true",
            "--hybrid_dequantize_process", "true",
            "--dump_format", "txt",
        ], capture_output=True, timeout=10)

        if not os.path.exists(OUTPUT_FILE):
            raise RuntimeError("BPU output not generated")

        with open(OUTPUT_FILE) as f:
            vals = f.read().strip().split()
        return np.array([float(v) for v in vals], dtype=np.float32)

    def predict(self, motion_window, calib_vector):
        """
        Full inference pipeline.
        Args:
            motion_window: [64, 10] float32 — 64 frames of motion features
            calib_vector:  [16] float32 — calibration vector
        Returns:
            [2] float32 — [biceps_ratio, triceps_ratio]
        """
        # 1. StandardScaler normalization (per-frame for motion, per-vector for calib)
        motion_scaled = self.motion_scaler.transform(
            motion_window.reshape(-1, MOTION_DIM)   # [64, 10]
        ).reshape(1, WINDOW_SIZE, MOTION_DIM)        # [1, 64, 10]

        calib_scaled = self.calib_scaler.transform(
            calib_vector.reshape(1, -1)               # [1, 16]
        )

        # 2. Merge into [1, 26, 1, 64]
        motion_4d = motion_scaled.transpose(0, 2, 1)[:, :, np.newaxis, :]   # [1, 10, 1, 64]
        calib_4d = np.tile(calib_scaled[:, :, np.newaxis, np.newaxis], (1, 1, 1, WINDOW_SIZE))  # [1, 16, 1, 64]
        merged = np.concatenate([motion_4d, calib_4d], axis=1).astype(np.float32)  # [1, 26, 1, 64]

        # 3. BPU inference
        return self._infer_raw(merged)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--bench', type=int, default=0, help='Run N-frame benchmark')
    p.add_argument('--input', type=str, help='Input .npy file')
    args = p.parse_args()

    print("=" * 50)
    print("  RDK X5 BPU EMG Inference")
    print("=" * 50)

    model = EMGPredictor()
    print("[OK] Model + scalers loaded")

    # Quick test
    motion = np.random.randn(64, 10).astype(np.float32)
    calib = np.random.randn(16).astype(np.float32)
    result = model.predict(motion, calib)
    print("Test: biceps={:.4f}, triceps={:.4f}".format(float(result[0]), float(result[1])))

    if args.bench:
        print("\nBenchmark ({} frames)...".format(args.bench))
        times = []
        for i in range(args.bench):
            t0 = time.perf_counter()
            model.predict(motion, calib)
            times.append(time.perf_counter() - t0)
        times_ms = np.array(times) * 1000
        print("  mean: {:.1f} ms".format(np.mean(times_ms)))
        print("  min:  {:.1f} ms".format(np.min(times_ms)))
        print("  FPS:  {:.0f}".format(1000 / np.mean(times_ms)))


if __name__ == '__main__':
    main()
