#!/usr/bin/env python3
"""
ros2_emg_bridge.py v2 — AnchorCalib-TCN → /virtual_emg

三种模式:
  --simulate    模拟模式：正弦角度 + 公式仿真
  --ml_predict  TCN预测模式：ONNX 推理 (替换旧 AnchorMLP)
  (默认)        UDP模式：接收 ESP32 实时 EMG
"""

import json, os, socket, threading, time, math, random, argparse, copy
import numpy as np
from tcn_bpu_predictor import TCNPredictor  # BPU version
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

ANGLE_MAX = 180.0
WINDOW_SIZE = 64


# ==================== UDP 接收器 (不变) ====================

class EMGUDPReceiver:
    def __init__(self, port=5005, timeout=0.1):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(timeout)
        self.sock.bind(("0.0.0.0", port))
        self.latest_emg = {
            "left":  {"biceps_uv": None, "triceps_uv": None, "timestamp": 0},
            "right": {"biceps_uv": None, "triceps_uv": None, "timestamp": 0}
        }
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.packet_count = 0

    def start(self): self.thread.start()
    def stop(self): self.running = False

    def _recv_loop(self):
        while self.running:
            try: data, addr = self.sock.recvfrom(4096)
            except socket.timeout: continue
            except Exception: continue
            try:
                text = data.decode("utf-8", errors="ignore").strip()
                parts = text.split(",")
                ts = int(parts[1])
                values = [int(x) for x in parts[2:]]
                if len(values) >= 2:
                    self.latest_emg["left"]["biceps_uv"] = values[0]
                    self.latest_emg["left"]["triceps_uv"] = values[1]
                    self.latest_emg["left"]["timestamp"] = ts
                if len(values) >= 4:
                    self.latest_emg["right"]["biceps_uv"] = values[2]
                    self.latest_emg["right"]["triceps_uv"] = values[3]
                    self.latest_emg["right"]["timestamp"] = ts
                else:
                    self.latest_emg["right"] = copy.deepcopy(self.latest_emg["left"])
                self.packet_count += 1
            except Exception: pass

    def get_latest_emg(self): return copy.deepcopy(self.latest_emg)


# ==================== TCN ONNX 推理引擎 (新) ====================

#OLD#class TCNPredictor:
#OLD#    """AnchorCalib-TCN ONNX 推理: 64帧窗口 → 2路ratio → 电压"""
#OLD#
#OLD#    def __init__(self, bin_path, motion_scaler_path, calib_scaler_path, config_path):
#OLD## REMOVED: old onnxruntime
#OLD#        import joblib
#OLD#        # BPU version - model loaded in TCNPredictor.__init__
#OLD#        self.ms = joblib.load(motion_scaler_path)
#OLD#        self.cs = joblib.load(calib_scaler_path)
#OLD#        with open(config_path) as f:
#OLD#            self.cfg = json.load(f)
#OLD#
#OLD#        self.motion_dim = len(self.cfg['motion_features'])
#OLD#        self.calib_dim = len(self.cfg['feature_names'])
#OLD#
#OLD#        # 64帧滑动窗口
#OLD#        self.buffer = []
#OLD#        self.prev_angle = None
#OLD#        self.prev_vel = 0.0
#OLD#        self.prev_time = None
#OLD#
#OLD#        print(f"[TCN] ONNX加载完成 motion={self.motion_dim}d calib={self.calib_dim}d")
#OLD#
#OLD#    def _build_motion(self, angle):
#OLD#        """构建运动特征: [angle_norm, vel, acc, sin, cos, phase×5]"""
#OLD#        angle = max(0.0, min(ANGLE_MAX, angle))
#OLD#        a_norm = angle / ANGLE_MAX
#OLD#        rad = math.radians(angle)
#OLD#        sin_a = math.sin(rad)
#OLD#        cos_a = math.cos(rad)
#OLD#
#OLD#        now = time.time()
#OLD#        if self.prev_angle is not None and self.prev_time is not None:
#OLD#            dt = max(now - self.prev_time, 0.001)
#OLD#            vel = (angle - self.prev_angle) / dt
#OLD#            acc = (vel - self.prev_vel) / dt
#OLD#        else:
#OLD#            vel = 0.0; acc = 0.0
#OLD#        self.prev_angle = angle
#OLD#        self.prev_vel = vel
#OLD#        self.prev_time = now
#OLD#
#OLD#        if abs(vel) < 3:       phase = 0
#OLD#        elif vel > 0:           phase = 1
#OLD#        else:                   phase = 3
#OLD#        phase_oh = np.eye(5, dtype=np.float32)[phase]
#OLD#
#OLD#        return np.array([a_norm, vel, acc, sin_a, cos_a] + phase_oh.tolist(), dtype=np.float32)
#OLD#
#OLD#    def predict(self, angle, calib_vec):
#OLD#        """calib_vec: [b_rest,b_90,b_peak,b_auc,b_slope,b_cv90,
#OLD#                        t_rest,t_90,t_peak,t_auc,t_slope,t_cv90,
#OLD#                        height,weight,bmi,gender]"""
#OLD#
#OLD#        feat = self._build_motion(angle)
#OLD#        self.buffer.append(feat)
#OLD#        if len(self.buffer) > WINDOW_SIZE:
#OLD#            self.buffer.pop(0)
#OLD#
#OLD#        if len(self.buffer) < WINDOW_SIZE:
#OLD#            pad = np.zeros((WINDOW_SIZE - len(self.buffer), self.motion_dim), dtype=np.float32)
#OLD#            window = np.vstack([pad, np.array(self.buffer, dtype=np.float32)])
#OLD#        else:
#OLD#            window = np.array(self.buffer, dtype=np.float32)
#OLD#
#OLD#        Xm = self.ms.transform(window.reshape(-1, self.motion_dim)).reshape(1, WINDOW_SIZE, self.motion_dim).astype(np.float32)
#OLD#        Xc = self.cs.transform(np.array(calib_vec, dtype=np.float64).reshape(1, -1)).astype(np.float32)
#OLD#
#OLD#        ratio = self.sess.run(None, {'motion': Xm, 'calib': Xc})[0][0]
#OLD#        ratio = np.clip(ratio, 0, None)
#OLD#
#OLD#        v_rest_b, v_90_b = calib_vec[0], calib_vec[1]
#OLD#        v_rest_t, v_90_t = calib_vec[6], calib_vec[7]
#OLD#
#OLD#        b_uv = max(0.0, float(ratio[0] * (v_90_b - v_rest_b) + v_rest_b))
#OLD#        t_uv = max(0.0, float(ratio[1] * (v_90_t - v_rest_t) + v_rest_t))
#OLD#        return b_uv, t_uv
#OLD#
#OLD#
#OLD## ==================== 模拟 EMG 生成器 (不变) ====================
#OLD#
class EMGSimulator:
    BICEPS_MIN, BICEPS_MAX = 150, 850
    TRICEPS_MIN, TRICEPS_MAX = 100, 500

    def __init__(self, noise_std=15):
        self.noise_std = noise_std
        self.time = 0.0

    def simulate(self, angle):
        if angle is None: return 0, 0
        ratio = max(0.0, min(1.0, (180.0 - angle) / 150.0))
        b = self.BICEPS_MIN + (self.BICEPS_MAX - self.BICEPS_MIN) * ratio
        t = self.TRICEPS_MAX - (self.TRICEPS_MAX - self.TRICEPS_MIN) * ratio
        b += random.gauss(0, self.noise_std)
        t += random.gauss(0, self.noise_std)
        return int(max(0, b)), int(max(0, t))

    def simulate_angle(self, period=6.0, amplitude=65.0, center=105.0):
        self.time += 0.1
        return round(center + amplitude * math.sin(2 * math.pi * self.time / period), 1)


# ==================== ROS2 桥接节点 ====================

class EMGBridgeNode(Node):
    def __init__(self, args):
        super().__init__("ros2_emg_bridge")
        self.mode = "simulate" if args.simulate else ("ml" if args.ml_predict else "udp")
        self.last_pub_time = time.time()

        if self.mode == "simulate":
            self.simulator = EMGSimulator(noise_std=args.sim_noise)
            self.sim_timer = self.create_timer(0.1, self.sim_timer_callback)
            self.get_logger().info("EMG Bridge [模拟模式] 10Hz")
            return

        if self.mode == "ml":
            bin_path = args.tcn_bin or os.path.expanduser("~/anchorcalib_tcn_bpu_v2.bin")
            ms = args.motion_scaler or os.path.expanduser("~/motion_scaler_63subj.pkl")
            cs = args.calib_scaler or os.path.expanduser("~/calib_scaler_63subj.pkl")
            cfg = args.calib_config or os.path.expanduser("~/calibration_config_63subj.json")

            self.predictor = TCNPredictor(args.tcn_bin, ms, cs, cfg)
            self.calib_vec = np.array(args.calib_vec, dtype=np.float64) if args.calib_vec else self._default_calib(args)

            # 加载持久化校准 (优先级高于默认值)
            if args.load_calib:
                calib_file = os.path.expanduser(args.calib_save or "~/calibration_save.json")
                if os.path.exists(calib_file):
                    try:
                        with open(calib_file) as f:
                            saved = json.load(f)
                        cv = saved.get('calib_vec', [])
                        if len(cv) == 16:
                            self.calib_vec = np.array(cv, dtype=np.float64)
                            self.get_logger().info(f'已加载校准: b_rest={self.calib_vec[0]:.0f} b_90={self.calib_vec[1]:.0f} t_rest={self.calib_vec[6]:.0f} t_90={self.calib_vec[7]:.0f}')
                    except Exception as e:
                        self.get_logger().warn(f'加载校准失败: {e}')

            self.sub = self.create_subscription(String, "/body_arm_angles", self.ml_angle_callback, 10)
            self.get_logger().info("EMG Bridge [TCN ONNX] AnchorCalib-TCN")
            self.get_logger().info(f"订阅: /body_arm_angles → 发布: /virtual_emg")
            self.get_logger().info(f"校准: rest_b={self.calib_vec[0]:.0f} v90_b={self.calib_vec[1]:.0f}")
            return

        # UDP mode
        self.sub = self.create_subscription(String, "/body_arm_angles", self.angle_callback, 10)
        self.emg = EMGUDPReceiver(port=args.udp_port)
        self.emg.start()
        self.emg_stale_timeout = args.emg_stale_timeout
        self.get_logger().info(f"EMG Bridge [UDP] port={args.udp_port}")

    def _default_calib(self, args):
        """从命令行参数构造默认校准向量"""
        return np.array([
            args.b_rest, args.b_90, args.b_90, 5000, 10, 0.1,   # biceps
            args.t_rest, args.t_90, args.t_90, 2000,  5, 0.1,   # triceps
            args.height, args.weight, args.bmi, args.gender
        ], dtype=np.float64)

    # ---- 模拟 ----
    def sim_timer_callback(self):
        angle = self.simulator.simulate_angle()
        b, t = self.simulator.simulate(angle)
        self._publish({"left":  {"valid": True, "angle": angle, "biceps_uv": b, "triceps_uv": t},
                        "right": {"valid": True, "angle": angle, "biceps_uv": int(b*0.95), "triceps_uv": int(t*0.95)}})
        now = time.time()
        if now - self.last_pub_time >= 3.0:
            self.get_logger().info(f"角度={angle:.0f}° → 二头={b} 三头={t}")
            self.last_pub_time = now

    # ---- TCN ML预测 ----
    def ml_angle_callback(self, msg):
        try: data = json.loads(msg.data)
        except Exception: return

        result = {}
        for side in ["left", "right"]:
            valid = data.get(f"{side}_valid", False)
            angle = data.get(f"{side}_elbow_angle")
            if valid and angle is not None:
                b, t = self.predictor.predict(angle, self.calib_vec)
            else:
                b, t = 0.0, 0.0
            result[side] = {"valid": valid and angle is not None, "angle": angle,
                            "biceps_uv": round(b, 1), "triceps_uv": round(t, 1)}
        self._publish(result)

    # ---- UDP ----
    def angle_callback(self, msg):
        try: angle_data = json.loads(msg.data)
        except Exception: return
        emg = self.emg.get_latest_emg()
        result = {}
        for side in ["left", "right"]:
            valid = angle_data.get(f"{side}_valid", False)
            angle = angle_data.get(f"{side}_elbow_angle")
            b_uv = emg[side]["biceps_uv"]; t_uv = emg[side]["triceps_uv"]
            emg_ts = emg[side]["timestamp"]
            emg_fresh = True
            if self.emg_stale_timeout > 0 and emg_ts > 0:
                emg_fresh = (time.time() - emg_ts / 1000.0) < self.emg_stale_timeout
            result[side] = {"valid": valid and emg_fresh and (b_uv is not None),
                            "angle": angle, "biceps_uv": b_uv or 0, "triceps_uv": t_uv or 0}
        self._publish(result)

    def _publish(self, data):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.pub.publish(msg)

    @property
    def pub(self):
        if not hasattr(self, "_pub"):
            self._pub = self.create_publisher(String, "/virtual_emg", 10)
        return self._pub

    def destroy(self):
        if self.mode == "udp": self.emg.stop()
        self.get_logger().info("EMG Bridge 已停止")
        super().destroy()


def main():
    parser = argparse.ArgumentParser(description="AnchorCalib-TCN EMG Bridge v2")

    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--ml_predict", action="store_true")
    parser.add_argument("--udp_port", type=int, default=5005)
    parser.add_argument("--emg_stale_timeout", type=float, default=2.0)
    parser.add_argument("--sim_noise", type=float, default=15.0)

    # TCN 模型路径
    parser.add_argument("--tcn_bin", type=str, default="~/anchorcalib_tcn_bpu_v2.bin")
    parser.add_argument("--motion_scaler", type=str, default="~/motion_scaler_63subj.pkl")
    parser.add_argument("--calib_scaler", type=str, default="~/calib_scaler_63subj.pkl")
    parser.add_argument("--calib_config", type=str, default="~/calibration_config_63subj.json")

    # 用户体征 + 校准
    parser.add_argument("--b_rest", type=float, default=200.0)
    parser.add_argument("--b_90", type=float, default=500.0)
    parser.add_argument("--t_rest", type=float, default=80.0)
    parser.add_argument("--t_90", type=float, default=120.0)
    parser.add_argument("--bmi", type=float, default=22.0)
    parser.add_argument("--height", type=float, default=170.0)
    parser.add_argument("--weight", type=float, default=70.0)
    parser.add_argument("--gender", type=int, default=0)
    parser.add_argument("--calib_vec", type=float, nargs=16, default=None,
                        help="完整16维校准向量")
    parser.add_argument("--load_calib", action="store_true", default=True,
                        help="加载持久化校准文件")
    parser.add_argument("--calib_save", default="~/calibration_save.json",
                        help="校准持久化文件路径")

    args = parser.parse_args()
    rclpy.init()
    node = EMGBridgeNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
