#!/usr/bin/env python3
"""
emg_cross_validation.py v2 — TCN预测 vs 真实EMG 交叉验证

架构:
  /body_arm_angles → [TCN ONNX] → predicted_emg ─┐
                                                  ├→ 误差计算 → /emg_validation
  ESP32 UDP / 仿真 → [RealEMG]   → measured_emg ─┘

v2 变更: AnchorMLP → AnchorCalib-TCN ONNX
"""

import json, math, os, socket, threading, time, argparse, copy, random
from collections import defaultdict
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

ANGLE_MAX = 180.0
WINDOW_SIZE = 64

# 质量监测模块
try:
    from emg_quality_monitor import CompensationDetector, ElectrodeHealthMonitor
    _HAS_QUALITY_MONITOR = True
except ImportError:
    print("[CV] emg_quality_monitor.py 未找到, 代偿/电极检测禁用")
    _HAS_QUALITY_MONITOR = False


# ==================== TCN ONNX 推理 (替换旧 AnchorMLP) ====================

class TCNPredictor:
    def __init__(self, onnx_path, motion_scaler_path, calib_scaler_path, config_path):
        import onnxruntime as ort
        import joblib
        import warnings
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ms = joblib.load(motion_scaler_path)
            self.cs = joblib.load(calib_scaler_path)
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.motion_dim = len(self.cfg['motion_features'])
        self.calib_dim = len(self.cfg['feature_names'])
        self.buffer = []
        self.prev_angle = None
        self.prev_vel = 0.0
        self.prev_time = None
        print(f"[TCN] motion={self.motion_dim}d calib={self.calib_dim}d")

    def _build_motion(self, angle):
        angle = max(0.0, min(ANGLE_MAX, angle))
        a_norm = angle / ANGLE_MAX
        rad = math.radians(angle)
        now = time.time()
        if self.prev_angle is not None and self.prev_time is not None:
            dt = max(now - self.prev_time, 0.001)
            vel = (angle - self.prev_angle) / dt
            acc = (vel - self.prev_vel) / dt
        else:
            vel = 0.0; acc = 0.0
        self.prev_angle = angle; self.prev_vel = vel; self.prev_time = now
        if abs(vel) < 3: phase = 0
        elif vel > 0: phase = 1
        else: phase = 3
        phase_oh = np.eye(5, dtype=np.float32)[phase]
        return np.array([a_norm, vel, acc, math.sin(rad), math.cos(rad)] + phase_oh.tolist(), dtype=np.float32)

    def predict(self, angle, calib_vec):
        feat = self._build_motion(angle)
        self.buffer.append(feat)
        if len(self.buffer) > WINDOW_SIZE: self.buffer.pop(0)
        if len(self.buffer) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(self.buffer), self.motion_dim), dtype=np.float32)
            window = np.vstack([pad, np.array(self.buffer, dtype=np.float32)])
        else:
            window = np.array(self.buffer, dtype=np.float32)
        Xm = self.ms.transform(window.reshape(-1, self.motion_dim)).reshape(1, WINDOW_SIZE, self.motion_dim).astype(np.float32)
        Xc = self.cs.transform(np.array(calib_vec, dtype=np.float64).reshape(1, -1)).astype(np.float32)
        ratio = self.sess.run(None, {'motion': Xm, 'calib': Xc})[0][0]
        ratio = np.clip(ratio, 0, None)
        v_rest_b, v_90_b = calib_vec[0], calib_vec[1]
        v_rest_t, v_90_t = calib_vec[6], calib_vec[7]
        b_uv = max(0.0, float(ratio[0] * (v_90_b - v_rest_b) + v_rest_b))
        t_uv = max(0.0, float(ratio[1] * (v_90_t - v_rest_t) + v_rest_t))
        return b_uv, t_uv


# ==================== 真实EMG接收器 (不变) ====================

class RealEMGReceiver:
    def __init__(self, udp_port=5005, simulate=False, sim_noise=20, sim_bias=(0.9, 1.15)):
        self.simulate = simulate; self.sim_noise = sim_noise; self.sim_bias = sim_bias
        self.latest = {"left": {"biceps_uv": None, "triceps_uv": None, "timestamp": 0},
                       "right": {"biceps_uv": None, "triceps_uv": None, "timestamp": 0}}
        self.packet_count = 0
        if not simulate:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.settimeout(0.1)
            self.sock.bind(("0.0.0.0", udp_port))
            self.running = True
            self.thread = threading.Thread(target=self._recv_loop, daemon=True)

    def start(self):
        if not self.simulate: self.thread.start()

    def stop(self):
        if not self.simulate: self.running = False

    def _recv_loop(self):
        while self.running:
            try: data, addr = self.sock.recvfrom(4096)
            except socket.timeout: continue
            except Exception: continue
            try:
                text = data.decode("utf-8", errors="ignore").strip()
                parts = text.split(",")
                values = [int(x) for x in parts[2:]]
                ts = int(parts[1])
                if len(values) >= 2:
                    self.latest["left"]["biceps_uv"] = values[0]
                    self.latest["left"]["triceps_uv"] = values[1]
                    self.latest["left"]["timestamp"] = ts
                if len(values) >= 4:
                    self.latest["right"]["biceps_uv"] = values[2]
                    self.latest["right"]["triceps_uv"] = values[3]
                    self.latest["right"]["timestamp"] = ts
                else:
                    self.latest["right"] = copy.deepcopy(self.latest["left"])
                self.packet_count += 1
            except Exception: pass

    def get_latest(self): return copy.deepcopy(self.latest)

    def simulate_from_predicted(self, pred_b, pred_t, angle):
        real_b = pred_b * self.sim_bias[0] + 15 * math.sin(math.radians(angle * 0.7))
        real_t = pred_t * self.sim_bias[1] + 10 * math.cos(math.radians(angle * 0.5))
        real_b += random.gauss(0, self.sim_noise)
        real_t += random.gauss(0, self.sim_noise)
        if random.random() < 0.02:
            if random.random() < 0.5: real_b *= random.uniform(0.3, 0.5)
            else: real_t += random.uniform(100, 200)
        return max(0, real_b), max(0, real_t)


# ==================== 交叉验证指标 (不变) ====================

class ValidationMetrics:
    def __init__(self, bin_size=5):
        self.bin_size = bin_size; self.reset()

    def reset(self):
        self.bins = defaultdict(lambda: {"pred_b": [], "pred_t": [], "real_b": [], "real_t": []})
        self.total_samples = 0

    def add(self, angle, pred_b, pred_t, real_b, real_t):
        bin_key = int(round(angle / self.bin_size) * self.bin_size)
        self.bins[bin_key]["pred_b"].append(pred_b)
        self.bins[bin_key]["pred_t"].append(pred_t)
        self.bins[bin_key]["real_b"].append(real_b)
        self.bins[bin_key]["real_t"].append(real_t)
        self.total_samples += 1

    @staticmethod
    def _compute(pred_list, real_list):
        if len(pred_list) < 3: return None
        pred = np.array(pred_list); real = np.array(real_list)
        errors = pred - real
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))
        ss_res = np.sum(errors ** 2); ss_tot = np.sum((real - np.mean(real)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        pred_mean = np.mean(pred); real_mean = np.mean(real)
        numerator = np.sum((pred - pred_mean) * (real - real_mean))
        denominator = np.sqrt(np.sum((pred - pred_mean) ** 2) * np.sum((real - real_mean) ** 2))
        pearson_r = float(numerator / denominator) if denominator > 1e-10 else 0.0
        mape = float(np.mean(np.abs(errors / (real + 1e-6))) * 100)
        return {"n_samples": len(pred_list), "pred_mean": float(np.mean(pred)),
                "real_mean": float(np.mean(real)), "mae": mae, "rmse": rmse,
                "r2": r2, "pearson_r": pearson_r, "mape_pct": mape, "bias": bias}

    def compute_by_bin(self):
        results = {}
        for bin_key in sorted(self.bins.keys()):
            data = self.bins[bin_key]
            b_metrics = self._compute(data["pred_b"], data["real_b"])
            t_metrics = self._compute(data["pred_t"], data["real_t"])
            if b_metrics or t_metrics:
                results[str(bin_key)] = {"angle_range": f"{bin_key}°", "biceps": b_metrics, "triceps": t_metrics}
        return results

    def compute_overall(self):
        all_pred_b, all_real_b = [], []
        all_pred_t, all_real_t = [], []
        for data in self.bins.values():
            all_pred_b.extend(data["pred_b"]); all_real_b.extend(data["real_b"])
            all_pred_t.extend(data["pred_t"]); all_real_t.extend(data["real_t"])
        return {"biceps": self._compute(all_pred_b, all_real_b),
                "triceps": self._compute(all_pred_t, all_real_t),
                "total_paired_samples": self.total_samples, "covered_angle_bins": len(self.bins)}

    def diagnose(self):
        overall = self.compute_overall()
        diagnoses = []
        for muscle, key in [("二头肌", "biceps"), ("三头肌", "triceps")]:
            m = overall.get(key)
            if m is None: continue
            if m["r2"] < 0.3: diagnoses.append(f"R {muscle} R2={m['r2']:.2f} — 模型几乎不能解释真实EMG变化")
            elif m["r2"] < 0.7: diagnoses.append(f"Y {muscle} R2={m['r2']:.2f} — 部分解释，有较大误差")
            else: diagnoses.append(f"G {muscle} R2={m['r2']:.2f} — 高度相关")
            if abs(m["bias"]) > 50:
                direction = "偏高" if m["bias"] > 0 else "偏低"
                diagnoses.append(f"C {muscle} 系统性{direction} {abs(m['bias']):.0f}uV — 建议修正校准")
            if m["mape_pct"] > 30:
                diagnoses.append(f"W {muscle} MAPE={m['mape_pct']:.1f}% — 误差较大")
        return diagnoses


# ==================== ROS2 交叉验证节点 (v2: TCN) ====================

class CrossValidationNode(Node):
    def __init__(self, args):
        super().__init__("emg_cross_validation")

        # TCN 预测器
        onnx = os.path.expanduser(args.tcn_onnx)
        ms = os.path.expanduser(args.motion_scaler)
        cs = os.path.expanduser(args.calib_scaler)
        cfg = os.path.expanduser(args.calib_config)
        self.predictor = TCNPredictor(onnx, ms, cs, cfg)

        if args.calib_vec:
            self.calib_vec = np.array(args.calib_vec, dtype=np.float64)
        else:
            self.calib_vec = np.array([
                args.b_rest, args.b_90, args.b_90, 5000, 10, 0.1,
                args.t_rest, args.t_90, args.t_90, 2000, 5, 0.1,
                args.height, args.weight, args.bmi, args.gender
            ], dtype=np.float64)

        self.b90 = self.calib_vec[1]; self.t90 = self.calib_vec[7]
        self.bmi = args.bmi; self.height = args.height
        self.weight = args.weight; self.gender = args.gender

        # 校准持久化
        self.calib_save_path = os.path.expanduser(args.calib_save)
        self._last_save_time = 0
        self._save_interval = args.calib_save_interval
        self._calib_mtime = 0  # 文件修改时间, 用于自动重载

        # 尝试加载已保存的校准
        if args.load_calib and os.path.exists(self.calib_save_path):
            self._calib_mtime = os.path.getmtime(self.calib_save_path)
            try:
                with open(self.calib_save_path) as f:
                    saved = json.load(f)
                cv = saved.get('calib_vec', [])
                if len(cv) == 16:
                    self.calib_vec = np.array(cv, dtype=np.float64)
                    self.b90 = self.calib_vec[1]; self.t90 = self.calib_vec[7]
                    self.get_logger().info(
                        f'已加载校准: b_rest={self.calib_vec[0]:.0f} b_90={self.b90:.0f} '
                        f't_rest={self.calib_vec[6]:.0f} t_90={self.t90:.0f} '
                        f'(保存时间: {saved.get("saved_at", "?")})'
                    )
            except Exception as e:
                self.get_logger().warn(f'加载校准失败: {e}')

        self.real_emg = RealEMGReceiver(udp_port=args.udp_port, simulate=args.simulate_real,
                                         sim_noise=args.sim_noise, sim_bias=(args.sim_bias_b, args.sim_bias_t))
        self.real_emg.start()

        self.metrics = ValidationMetrics(bin_size=args.bin_size)
        self.calibration_window = []
        self.auto_calibrate = args.auto_calibrate
        self.emit_interval = args.emit_interval

        self.sub_angles = self.create_subscription(String, "/body_arm_angles", self.angle_callback, 10)
        self.pub_validation = self.create_publisher(String, "/emg_validation", 10)
        self.pub_calibration = self.create_publisher(String, "/emg_calibration", 10)
        self.pub_alerts = self.create_publisher(String, "/emg_alerts", 10)
        self.timer = self.create_timer(self.emit_interval, self.emit_metrics)

        # 质量监测器
        if _HAS_QUALITY_MONITOR:
            self.comp_detector = CompensationDetector(window_size=50, alert_threshold=40)
            self.elec_monitor = ElectrodeHealthMonitor(window_size=100)
            self.get_logger().info("代偿检测 + 电极监测 已启用")
        else:
            self.comp_detector = None
            self.elec_monitor = None

        # 诊断计数器
        self._callback_count = 0      # angle_callback 被调用次数
        self._sample_count = 0        # 实际添加的样本数
        self._last_callback_log = time.time()

        mode_str = "仿真" if args.simulate_real else f"UDP:{args.udp_port}"
        self.get_logger().info(f"EMG交叉验证 v2 [TCN] | real={mode_str}")
        self.get_logger().info(f"校准: rest_b={self.calib_vec[0]:.0f} v90_b={self.calib_vec[1]:.0f}")

    def angle_callback(self, msg):
        try: data = json.loads(msg.data)
        except Exception: return
        real = self.real_emg.get_latest()
        calib = self.calib_vec

        for side in ["left", "right"]:
            valid = data.get(f"{side}_valid", False)
            angle = data.get(f"{side}_elbow_angle")
            if not valid or angle is None: continue

            pred_b, pred_t = self.predictor.predict(angle, calib)

            if self.real_emg.simulate:
                real_b, real_t = self.real_emg.simulate_from_predicted(pred_b, pred_t, angle)
            else:
                real_b = real[side]["biceps_uv"]; real_t = real[side]["triceps_uv"]
                if real[side]["timestamp"] > 0:
                    age = time.time() - real[side]["timestamp"] / 1000.0
                    if age > 2.0: continue
                if real_b is None or real_t is None: continue

            self.metrics.add(angle, pred_b, pred_t, real_b, real_t)

            # 喂入质量监测器
            if self.comp_detector is not None:
                self.comp_detector.add(angle, pred_b, pred_t, real_b, real_t)
            if self.elec_monitor is not None:
                self.elec_monitor.add(real_b, real_t, angle)

            if self.auto_calibrate:
                self.calibration_window.append({"angle": angle, "pred_b": pred_b, "pred_t": pred_t,
                                                 "real_b": real_b, "real_t": real_t})

        # 诊断: 每60秒输出一次回调统计
        self._callback_count += 1
        prev_samples = self._sample_count
        self._sample_count = self.metrics.total_samples
        now = time.time()
        if now - self._last_callback_log >= 60:
            delta = self._sample_count - getattr(self, '_last_sample_count', 0)
            self._last_sample_count = self._sample_count
            self._last_callback_log = now
            self.get_logger().info(
                f'[诊断] angle_callback调用={self._callback_count}次 '
                f'总样本={self._sample_count} 本分钟新增={delta} '
                f'验证窗口={len(self.calibration_window)}')

    def emit_metrics(self, event=None):
        # 自动检测语音校准更新并重载
        self._reload_calib_if_updated()

        total = self.metrics.total_samples
        if total < 10:
            msg = String()
            msg.data = json.dumps({"status": "collecting", "total_samples": total,
                                   "message": f"样本不足 ({total}/10)"}, ensure_ascii=False)
            self.pub_validation.publish(msg)
            return

        overall = self.metrics.compute_overall()
        by_bin = self.metrics.compute_by_bin()
        diagnoses = self.metrics.diagnose()

        calib = None
        if self.auto_calibrate and len(self.calibration_window) >= 20:
            calib = self._compute_calibration()
            if calib: self._apply_calibration(calib)

        report = {"timestamp": time.time(), "status": "validating",
                  "config": {"bin_size": self.metrics.bin_size, "b90": self.b90, "t90": self.t90},
                  "overall": overall, "by_angle_bin": by_bin, "diagnosis": diagnoses}
        if calib: report["calibration"] = calib

        # ---- 质量监测 ----
        quality_alerts = []
        if self.comp_detector is not None:
            comp = self.comp_detector.analyze()
            report['compensation'] = comp
            if comp.get('alert'):
                quality_alerts.append({
                    'type': 'compensation',
                    'level': comp['compensation_level'],
                    'score': comp['compensation_score'],
                    'details': comp['details'][:2],
                })

        if self.elec_monitor is not None:
            elec = self.elec_monitor.analyze()
            report['electrode_health'] = elec
            if elec.get('alert'):
                quality_alerts.append({
                    'type': 'electrode',
                    'biceps_quality': elec['biceps_quality'],
                    'triceps_quality': elec['triceps_quality'],
                    'dropout_b': elec['dropout_b'],
                    'dropout_t': elec['dropout_t'],
                    'details': elec['details'][:2],
                })

        msg = String(); msg.data = json.dumps(report, ensure_ascii=False)
        self.pub_validation.publish(msg)

        # 发布紧急告警
        if quality_alerts:
            alert_msg = String()
            alert_msg.data = json.dumps({
                'timestamp': time.time(),
                'alerts': quality_alerts,
            }, ensure_ascii=False)
            self.pub_alerts.publish(alert_msg)

        b = overall.get("biceps", {}) or {}; t = overall.get("triceps", {}) or {}
        q_info = ""
        if self.comp_detector is not None:
            cs = report.get('compensation', {}).get('compensation_score', 0)
            q_info += f" | 代偿={cs:.0f}"
        if self.elec_monitor is not None:
            eq = report.get('electrode_health', {})
            qb = eq.get('biceps_quality', 100)
            qt = eq.get('triceps_quality', 100)
            q_info += f" | 电极={qb:.0f}/{qt:.0f}"
        # 检测数据流停滞
        last_n = getattr(self, '_last_emit_n', 0)
        stale_rounds = getattr(self, '_stale_rounds', 0)
        if total == last_n and total >= 10:
            stale_rounds += 1
        else:
            stale_rounds = 0
        self._stale_rounds = stale_rounds
        self._last_emit_n = total

        stall_warn = ''
        if stale_rounds >= 3:
            stall_warn = ' [⚠数据停滞! 检查摄像头是否对准人]'

        self.get_logger().info(
            f"[交叉验证] n={total} | 二头 MAE={b.get('mae',0):.1f}uV R2={b.get('r2',0):.2f} | "
            f"三头 MAE={t.get('mae',0):.1f}uV R2={t.get('r2',0):.2f}{q_info}{stall_warn}")

    def _compute_calibration(self):
        window = self.calibration_window[-50:]
        self.calibration_window = self.calibration_window[-100:]
        ratios_b, ratios_t = [], []
        for s in window:
            if s["pred_b"] > 10 and s["real_b"] > 10:
                ratios_b.append(s["real_b"] / s["pred_b"])
            if s["pred_t"] > 10 and s["real_t"] > 10:
                ratios_t.append(s["real_t"] / s["pred_t"])
        if len(ratios_b) < 10 or len(ratios_t) < 10: return None
        mr_b = float(np.median(ratios_b)); mr_t = float(np.median(ratios_t))
        return {"median_ratio_biceps": round(mr_b,3), "median_ratio_triceps": round(mr_t,3),
                "suggested_b90": round(self.b90 * mr_b,1), "suggested_t90": round(self.t90 * mr_t,1),
                "current_b90": self.b90, "current_t90": self.t90, "n_samples": len(ratios_b)}

    def _apply_calibration(self, calib):
        # 仿真模式下不调整校准值 (避免反馈回路污染)
        if self.real_emg.simulate:
            return
        alpha = 0.3
        old_b, old_t = self.b90, self.t90
        self.b90 = round(self.b90 * (1-alpha) + calib["suggested_b90"] * alpha, 1)
        self.t90 = round(self.t90 * (1-alpha) + calib["suggested_t90"] * alpha, 1)
        self.calib_vec[1] = self.b90; self.calib_vec[7] = self.t90
        calib_msg = String()
        calib_msg.data = json.dumps({"old_b90": old_b, "old_t90": old_t,
                                      "new_b90": self.b90, "new_t90": self.t90}, ensure_ascii=False)
        self.pub_calibration.publish(calib_msg)
        self.get_logger().info(f"[校准] b90:{old_b}→{self.b90} t90:{old_t}→{self.t90}")
        self._save_calib()

    def _reload_calib_if_updated(self):
        """检测语音校准文件更新, 自动重载 (无需重启)"""
        try:
            if not os.path.exists(self.calib_save_path):
                return
            mtime = os.path.getmtime(self.calib_save_path)
            if mtime == self._calib_mtime:
                return
            self._calib_mtime = mtime
            with open(self.calib_save_path) as f:
                saved = json.load(f)
            cv = saved.get('calib_vec', [])
            if len(cv) == 16:
                old_b, old_t = self.b90, self.t90
                self.calib_vec = np.array(cv, dtype=np.float64)
                self.b90 = self.calib_vec[1]; self.t90 = self.calib_vec[7]
                src = saved.get('source', 'file')
                self.get_logger().info(
                    f'[校准] 自动重载 (来源:{src}) b90:{old_b}→{self.b90} t90:{old_t}→{self.t90}')
        except Exception as e:
            self.get_logger().warn(f'[校准] 重载失败: {e}')

    def _save_calib(self):
        """持久化校准值到文件 (仅真实EMG模式, 仿真模式跳过以免污染校准)"""
        if self.real_emg.simulate:
            return  # 仿真数据不保存校准, 避免污染语音校准结果
        now = time.time()
        if now - self._last_save_time < self._save_interval:
            return
        self._last_save_time = now
        try:
            data = {
                'calib_vec': self.calib_vec.tolist(),
                'b_rest': float(self.calib_vec[0]),
                'b_90': self.b90,
                't_rest': float(self.calib_vec[6]),
                't_90': self.t90,
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': self.metrics.total_samples,
            }
            with open(self.calib_save_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().warn(f'保存校准失败: {e}')

    def destroy(self):
        self.real_emg.stop()
        if self.metrics.total_samples >= 10:
            self.get_logger().info("生成最终报告...")
            self.emit_metrics()
        super().destroy()


def main():
    parser = argparse.ArgumentParser(description="EMG交叉验证 v2 (TCN)")

    # TCN 模型
    parser.add_argument("--tcn_onnx", default="~/anchorcalib_tcn_63subj.onnx")
    parser.add_argument("--motion_scaler", default="~/motion_scaler_63subj.pkl")
    parser.add_argument("--calib_scaler", default="~/calib_scaler_63subj.pkl")
    parser.add_argument("--calib_config", default="~/calibration_config_63subj.json")

    # 数据源
    parser.add_argument("--simulate_real", action="store_true")
    parser.add_argument("--udp_port", type=int, default=5005)
    parser.add_argument("--sim_noise", type=float, default=20.0)
    parser.add_argument("--sim_bias_b", type=float, default=0.88)
    parser.add_argument("--sim_bias_t", type=float, default=1.18)

    # 校准 + 体征
    parser.add_argument("--b_rest", type=float, default=200.0)
    parser.add_argument("--b_90", type=float, default=500.0)
    parser.add_argument("--t_rest", type=float, default=80.0)
    parser.add_argument("--t_90", type=float, default=120.0)
    parser.add_argument("--bmi", type=float, default=22.0)
    parser.add_argument("--height", type=float, default=170.0)
    parser.add_argument("--weight", type=float, default=70.0)
    parser.add_argument("--gender", type=int, default=0)
    parser.add_argument("--calib_vec", type=float, nargs=16, default=None)

    # 验证参数
    parser.add_argument("--bin_size", type=int, default=5)
    parser.add_argument("--emit_interval", type=float, default=5.0)
    parser.add_argument("--auto_calibrate", action="store_true")
    parser.add_argument("--load_calib", action="store_true", default=True,
                        help="启动时加载已保存的校准值")
    parser.add_argument("--calib_save", default="~/calibration_save.json",
                        help="校准持久化文件路径")
    parser.add_argument("--calib_save_interval", type=float, default=30.0,
                        help="校准保存间隔(秒)")

    args = parser.parse_args()
    rclpy.init()
    node = CrossValidationNode(args)
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info("Ctrl+C退出")
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
