#!/usr/bin/env python3
"""
语音交互 v6 — 系统状态感知 + 真实功能控制

架构:
  主线程:  语音交互循环 (录音→识别→意图匹配→执行→TTS)
  ROS2线程: 订阅 /body_arm_angles, /virtual_emg, /emg_validation → SystemState

模式:
  --push-to-talk  (默认) 按 Enter 开始说话
  --vad           语音活动检测自动触发 (免按键)
  --daemon        VAD + 后台持续运行 (配合 start_all.sh)

新增功能 vs v5:
  - 读取真实角度/EMG/交叉验证结果 (不再是写死的假数据)
  - 训练模式: 开始/停止/报告 (实时计次、计时、统计)
  - 校准引导: 语音指导用户完成 rest→90° 校准流程
  - 系统诊断: 读取交叉验证报告, 评估模型准确度
  - 免按键VAD模式: 持续监听, 检测到说话自动识别

用法:
  python3 voice_demo_v6.py                          # 按键模式
  python3 voice_demo_v6.py --vad                    # VAD模式
  python3 voice_demo_v6.py --daemon                 # 后台守护模式
"""

import subprocess, os, sys, time, wave, struct, json, math
import threading
import argparse
import random
from collections import deque

import numpy as np
from faster_whisper import WhisperModel

# ============================== 配置 ==============================

MODEL_PATH = '/opt/whisper-models/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66'
MIC_DEV    = 'plughw:0,0'     # Jieli 无线麦
RATE       = 48000             # Jieli 原生采样率 (plughw 自动SRC)
RECORD_SEC = 5                 # 最长录音秒数
VAD_ENERGY_THRESHOLD = 300     # VAD 能量阈值 (RMS)
VAD_SILENCE_SEC = 1.2          # 连续静音多久视为说话结束
VAD_CHUNK_SEC = 0.2            # VAD 检测粒度

# ============================== TTS ==============================

class TTSSpeaker:
    """轻量 TTS — 避免依赖外部 tts_speaker.py (该文件可能不在同目录)"""

    def __init__(self):
        import shutil
        self._has_edge = shutil.which('edge-tts') is not None
        self._has_espeak = shutil.which('espeak-ng') is not None
        self._has_ffmpeg = shutil.which('ffmpeg') is not None
        self.engine = 'edge' if self._has_edge else ('espeak' if self._has_espeak else None)

    def speak(self, text):
        """同步播放, 阻塞到完成"""
        if not text: return
        print(f'  🔊 {text}')
        if self.engine == 'edge' and self._has_edge:
            self._speak_edge(text)
        elif self._has_espeak:
            self._speak_espeak(text)

    def _speak_edge(self, text):
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                mp3_path = f.name
            wav_path = mp3_path.replace('.mp3', '.wav')
            subprocess.run([
                'edge-tts', '--text', text,
                '--voice', 'zh-CN-XiaoxiaoNeural',
                '--rate', '+0%', '--volume', '+0%',
                '--write-media', mp3_path
            ], capture_output=True, timeout=20)
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                subprocess.run(['ffmpeg', '-i', mp3_path, '-y', wav_path],
                               capture_output=True, timeout=10)
                if os.path.exists(wav_path):
                    subprocess.run(['aplay', '-q', wav_path], timeout=20)
            for p in [mp3_path, wav_path]:
                if os.path.exists(p): os.unlink(p)
        except Exception as e:
            print(f'  [TTS edge] 失败: {e}')
            if self._has_espeak: self._speak_espeak(text)

    def _speak_espeak(self, text):
        try:
            subprocess.run(['espeak-ng', '-v', 'cmn', '-s', '160', '-p', '50', '--', text], timeout=15)
        except Exception as e:
            print(f'  [TTS espeak] 失败: {e}')


# ============================== 系统状态 (线程安全) ==============================

class SystemState:
    """ROS2 订阅的实时数据, 线程安全"""

    def __init__(self):
        self._lock = threading.Lock()

        # 传感器数据
        self.left_angle = None
        self.right_angle = None
        self.left_valid = False
        self.right_valid = False
        self.angle_timestamp = 0

        # EMG 预测值
        self.emg_biceps = None
        self.emg_triceps = None
        self.emg_timestamp = 0

        # 交叉验证
        self.validation_report = None
        self.validation_timestamp = 0

        # 告警
        self.latest_alerts = []
        self.alerts_timestamp = 0

        # 训练状态
        self.is_training = False
        self.training_start_time = 0
        self.training_rep_count = 0
        self.training_angles = []        # [(ts, angle), ...]
        self.training_emg_b = []
        self.training_emg_t = []
        self._rep_state = 'waiting'       # waiting | bending | extending
        self._rep_peak_angle = 0
        self._rep_min_angle = 180

        # 校准流程状态
        self.calib_phase = 'idle'         # idle → relax → recording_rest → bend → recording_90 → done
        self.calib_rest_b = None
        self.calib_rest_t = None
        self.calib_v90_b = None
        self.calib_v90_t = None
        self.calib_phase_start = 0

    # ---- 传感器更新 (ROS2回调) ----
    def update_angle(self, side, angle, valid):
        with self._lock:
            if side == 'left':
                self.left_angle = angle; self.left_valid = valid
            else:
                self.right_angle = angle; self.right_valid = valid
            self.angle_timestamp = time.time()

            # 训练模式: 计次
            if self.is_training and valid and angle is not None:
                self.training_angles.append((time.time(), angle))
                self._count_rep(angle)

    def update_emg(self, biceps, triceps):
        with self._lock:
            self.emg_biceps = biceps
            self.emg_triceps = triceps
            self.emg_timestamp = time.time()

            if self.is_training:
                self.training_emg_b.append(biceps)
                self.training_emg_t.append(triceps)

    def update_validation(self, report):
        with self._lock:
            self.validation_report = report
            self.validation_timestamp = time.time()

    # ---- 训练计次 ----
    def _count_rep(self, angle):
        """状态机计次: waiting → bending(角度上升>60°) → extending(角度下降) = 1 rep"""
        if self._rep_state == 'waiting':
            if angle > 60:
                self._rep_state = 'bending'
                self._rep_peak_angle = angle

        elif self._rep_state == 'bending':
            if angle > self._rep_peak_angle:
                self._rep_peak_angle = angle
            elif angle < self._rep_peak_angle - 15:   # 开始回落
                self._rep_state = 'extending'
                self._rep_min_angle = angle

        elif self._rep_state == 'extending':
            if angle < self._rep_min_angle:
                self._rep_min_angle = angle
            elif angle > self._rep_min_angle + 10:    # 再次上升 = 新的一次
                self.training_rep_count += 1
                self._rep_state = 'bending'
                self._rep_peak_angle = angle

    def start_training(self):
        with self._lock:
            self.is_training = True
            self.training_start_time = time.time()
            self.training_rep_count = 0
            self.training_angles = []
            self.training_emg_b = []
            self.training_emg_t = []
            self._rep_state = 'waiting'
            self._rep_peak_angle = 0
            self._rep_min_angle = 180

    def stop_training(self):
        with self._lock:
            self.is_training = False
            duration = time.time() - self.training_start_time if self.training_start_time else 0
            angles = [a for _, a in self.training_angles] if self.training_angles else [0]
            emg_b = self.training_emg_b if self.training_emg_b else [0]
            emg_t = self.training_emg_t if self.training_emg_t else [0]
            return {
                'duration_sec': round(duration, 1),
                'rep_count': self.training_rep_count,
                'angle_samples': len(angles),
                'max_angle': round(max(angles), 1),
                'avg_angle': round(sum(angles) / len(angles), 1),
                'avg_emg_biceps': round(sum(emg_b) / len(emg_b), 1),
                'avg_emg_triceps': round(sum(emg_t) / len(emg_t), 1),
            }

    # ---- 读取 (线程安全快照) ----
    def get_angle(self):
        with self._lock:
            return self.left_angle, self.left_valid

    def get_emg(self):
        with self._lock:
            return self.emg_biceps, self.emg_triceps, (time.time() - self.emg_timestamp if self.emg_timestamp else 999)

    def update_alerts(self, alerts):
        with self._lock:
            self.latest_alerts = alerts
            self.alerts_timestamp = time.time()

    def get_alerts(self):
        with self._lock:
            age = time.time() - self.alerts_timestamp if self.alerts_timestamp else 999
            return list(self.latest_alerts), age

    def get_validation(self):
        with self._lock:
            return self.validation_report, (time.time() - self.validation_timestamp if self.validation_timestamp else 999)

    def get_training_live(self):
        with self._lock:
            if not self.is_training: return None
            return {
                'duration': round(time.time() - self.training_start_time, 1),
                'reps': self.training_rep_count,
                'angle': self.left_angle,
            }

    # ---- 校准流程 ----
    def start_calibration(self):
        with self._lock:
            self.calib_phase = 'relax'
            self.calib_phase_start = time.time()
            self.calib_rest_b = None
            self.calib_rest_t = None
            self.calib_v90_b = None
            self.calib_v90_t = None

    def calib_tick(self):
        """每轮检查校准状态, 返回 (phase, prompt, is_done)"""
        with self._lock:
            phase = self.calib_phase
            elapsed = time.time() - self.calib_phase_start

        if phase == 'idle':
            return 'idle', None, False

        elif phase == 'relax':
            if elapsed > 3.0:
                # 记录 rest 值
                with self._lock:
                    if self.emg_biceps is not None:
                        self.calib_rest_b = self.emg_biceps
                        self.calib_rest_t = self.emg_triceps or 0
                    self.calib_phase = 'bend'
                    self.calib_phase_start = time.time()
                return 'relax', None, False
            return 'relax', '请完全放松手臂，自然下垂，保持静止', False

        elif phase == 'bend':
            if elapsed > 5.0:
                # 超时, 仍然尝试记录
                with self._lock:
                    angle = self.left_angle
                    if angle is not None and angle > 45:
                        self.calib_v90_b = self.emg_biceps
                        self.calib_v90_t = self.emg_triceps
                    self.calib_phase = 'done'
                    self.calib_phase_start = time.time()
                return 'bend', None, False
            return 'bend', '请缓慢弯曲手肘至九十度，并保持住', False

        elif phase == 'done':
            return 'done', None, True

        return 'idle', None, False

    def get_calib_result(self):
        with self._lock:
            return {
                'rest_b': self.calib_rest_b,
                'rest_t': self.calib_rest_t,
                'v90_b': self.calib_v90_b,
                'v90_t': self.calib_v90_t,
            }


# ============================== ROS2 订阅节点 (后台线程) ==============================

class VoiceStateNode:
    """最小 ROS2 节点: 仅订阅, 更新 SystemState"""

    def __init__(self, state: SystemState):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        self._state = state
        self._node = Node('voice_state_listener')
        self._node.create_subscription(String, '/body_arm_angles', self._on_angle, 10)
        self._node.create_subscription(String, '/virtual_emg', self._on_emg, 10)
        self._node.create_subscription(String, '/emg_validation', self._on_validation, 10)
        self._node.create_subscription(String, '/emg_alerts', self._on_alerts, 10)
        self._node.get_logger().info('语音状态监听已就绪 (含告警订阅)')

    @property
    def node(self): return self._node

    def _on_angle(self, msg):
        try:
            data = json.loads(msg.data)
            for side in ['left', 'right']:
                valid = data.get(f'{side}_valid', False)
                angle = data.get(f'{side}_elbow_angle')
                self._state.update_angle(side, angle, valid)
        except Exception:
            pass

    def _on_emg(self, msg):
        try:
            data = json.loads(msg.data)
            left = data.get('left', {})
            self._state.update_emg(
                left.get('biceps_uv'),
                left.get('triceps_uv')
            )
        except Exception:
            pass

    def _on_validation(self, msg):
        try:
            report = json.loads(msg.data)
            self._state.update_validation(report)
        except Exception:
            pass

    def _on_alerts(self, msg):
        try:
            data = json.loads(msg.data)
            alerts = data.get('alerts', [])
            self._state.update_alerts(alerts)
        except Exception:
            pass


def ros2_spin_thread(state: SystemState):
    """后台线程: ROS2 spin"""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=['--ros-args', '--log-level', 'warn'])
    bridge = VoiceStateNode(state)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge.node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(bridge.node)
        bridge.node.destroy_node()
        rclpy.shutdown()


# ============================== 音频采集 ==============================

def record_wav(wav_path, duration=RECORD_SEC):
    """从 Jieli 无线麦录音"""
    subprocess.run(['arecord', '-q',
                    '-D', MIC_DEV,
                    '-d', str(int(duration)),
                    '-f', 'S16_LE',
                    '-r', str(RATE),
                    '-c', '1',
                    wav_path],
                   timeout=int(duration) + 5)


def preprocess_audio(wav_path):
    """去直流 → 16kHz WAV"""
    with wave.open(wav_path, 'rb') as w:
        params = w.getparams()
        frames = w.readframes(params.nframes)
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    peak_raw = float(abs(data).max())
    data_clean = data - data.mean()
    zcr = float(np.sum(np.abs(np.diff(np.sign(data_clean))) > 0) / len(data_clean))
    # 归一化
    data_clean = data_clean / (abs(data_clean).max() + 1e-8)
    out_path = wav_path.replace('.wav', '_norm.wav')
    data_out = (data_clean * 32767).astype(np.int16)
    with wave.open(out_path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(data_out.tobytes())
    return out_path, peak_raw, zcr


def record_vad(wav_path, max_duration=RECORD_SEC, timeout=30):
    """VAD 模式录音: 检测到声音开始录, 静音>VAD_SILENCE_SEC停止"""
    CHUNK = int(RATE * VAD_CHUNK_SEC)  # 采样点数/chunk
    started = False
    silent_chunks = 0
    silence_needed = int(VAD_SILENCE_SEC / VAD_CHUNK_SEC)
    all_frames = []
    total_chunks = 0
    max_chunks = int(max_duration / VAD_CHUNK_SEC)
    timeout_chunks = int(timeout / VAD_CHUNK_SEC)

    # 清空缓冲区
    subprocess.run(['arecord', '-q', '-D', MIC_DEV, '-d', '0.1',
                    '-f', 'S16_LE', '-r', str(RATE), '-c', '1', '/dev/null'],
                   timeout=2)

    while total_chunks < timeout_chunks:
        chunk_file = f'/tmp/vad_chunk_{total_chunks}.wav'
        subprocess.run(['arecord', '-q', '-D', MIC_DEV,
                        '-d', str(VAD_CHUNK_SEC),
                        '-f', 'S16_LE', '-r', str(RATE), '-c', '1',
                        chunk_file],
                       timeout=int(VAD_CHUNK_SEC) + 2)

        if not os.path.exists(chunk_file): break
        with wave.open(chunk_file, 'rb') as w:
            frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        rms = float(np.sqrt(np.mean((data - data.mean()) ** 2)))
        os.unlink(chunk_file)

        total_chunks += 1

        if not started:
            if rms > VAD_ENERGY_THRESHOLD:
                started = True
                silent_chunks = 0
                all_frames.append(data)
                print('  🎤 检测到语音...', end='', flush=True)
            continue

        # 已开始录音
        print('.' if rms > VAD_ENERGY_THRESHOLD else '_', end='', flush=True)
        all_frames.append(data)

        if rms < VAD_ENERGY_THRESHOLD:
            silent_chunks += 1
            if silent_chunks >= silence_needed:
                print(' ✓')
                break
        else:
            silent_chunks = 0

        if len(all_frames) >= max_chunks:
            print(' (最长)')
            break

    if not started:
        return False, 0, 0

    # 拼接并写入 WAV
    combined = np.concatenate(all_frames)
    peak_raw = float(abs(combined).max())
    combined_clean = combined - combined.mean()
    zcr = float(np.sum(np.abs(np.diff(np.sign(combined_clean))) > 0) / len(combined_clean))

    # 写 48kHz 原始
    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(combined.astype(np.int16).tobytes())

    return True, peak_raw, zcr


# ============================== 语音助手主逻辑 ==============================

class VoiceAssistant:
    """语音助手: 命令匹配 + 系统交互"""

    def __init__(self, state: SystemState, tts: TTSSpeaker, use_vad=False):
        self.state = state
        self.tts = tts
        self.use_vad = use_vad
        self.running = True

        # 校准状态机
        self._calib_active = False
        self._calib_last_prompt = None

        # 上一次状态播报 (避免重复)
        self._last_status_time = 0

    # ---- 意图匹配 ----
    def match_command(self, text):
        """返回 (command_id, args) 或 None"""
        text = text.strip().lower().replace(' ', '')

        # 多关键词匹配, 按优先级
        patterns = [
            (['再见', '退出', '拜拜'], 'goodbye'),
            (['谢谢', '多谢', '感谢'], 'thanks'),
            (['帮助', '你能做什么', '有什么功能', '指令', '命令'], 'help'),
            (['停止训练', '结束训练', '停止记录', '结束记录'], 'stop_training'),
            (['开始训练', '启动训练', '开始记录', '启动记录', '训练开始'], 'start_training'),
            (['训练报告', '训练统计', '训练结果', '训练总结'], 'training_report'),
            (['开始校准', '校准', '重新校准', '自动校准'], 'start_calib'),
            (['肌肉状态', '肌肉数据', '肌电数据', '肌电', 'emg'], 'muscle_status'),
            (['当前角度', '角度多少', '手臂角度', '关节角度', '现在多少度'], 'current_angle'),
            (['诊断', '健康检查', '系统检测', '自检'], 'diagnosis'),
            (['检查状态', '系统状态', '运行状态', '状态'], 'status'),
            (['你好', '嗨', '哈喽', 'hello', 'hi'], 'greeting'),
        ]

        for keywords, cmd_id in patterns:
            for kw in keywords:
                if kw in text:
                    return cmd_id

        return None

    # ---- 命令处理 ----
    def handle(self, cmd_id):
        """执行命令, 返回 TTS 响应文本"""
        if cmd_id == 'greeting':
            return self._cmd_greeting()

        elif cmd_id == 'status':
            return self._cmd_status()

        elif cmd_id == 'current_angle':
            return self._cmd_angle()

        elif cmd_id == 'muscle_status':
            return self._cmd_muscle()

        elif cmd_id == 'diagnosis':
            return self._cmd_diagnosis()

        elif cmd_id == 'start_training':
            return self._cmd_start_training()

        elif cmd_id == 'stop_training':
            return self._cmd_stop_training()

        elif cmd_id == 'training_report':
            return self._cmd_training_report()

        elif cmd_id == 'start_calib':
            return self._cmd_start_calib()

        elif cmd_id == 'help':
            return self._cmd_help()

        elif cmd_id == 'goodbye':
            self.running = False
            return '再见，祝您康复顺利！'

        elif cmd_id == 'thanks':
            return '不客气，这是我应该做的。祝您康复顺利！'

        return None

    # ---- 具体实现 ----

    def _cmd_greeting(self):
        angle, valid = self.state.get_angle()
        emg_b, emg_t, age = self.state.get_emg()

        parts = ['你好，我是智能健康袖套语音助手。']
        if valid and angle is not None:
            parts.append(f'当前肘关节角度{angle:.0f}度。')
        if emg_b is not None and age < 5:
            parts.append(f'二头肌肌电{emg_b:.0f}微伏。')
        live = self.state.get_training_live()
        if live:
            parts.append(f'训练已进行{live["duration"]:.0f}秒，完成{live["reps"]}次。')
        parts.append('请说"帮助"查看可用指令。')
        return ' '.join(parts)

    def _cmd_status(self):
        angle, valid = self.state.get_angle()
        emg_b, emg_t, emg_age = self.state.get_emg()
        val_rpt, val_age = self.state.get_validation()

        parts = []
        # 角度
        if valid and angle is not None:
            parts.append(f'肘关节角度{angle:.0f}度。')
        else:
            parts.append('角度传感器暂未连接。')

        # EMG
        if emg_b is not None and emg_age < 5:
            parts.append(f'二头肌{emg_b:.0f}微伏，三头肌{emg_t:.0f}微伏。')
        else:
            parts.append('肌电数据暂未更新。')

        # 交叉验证
        if val_rpt and val_age < 30:
            overall = val_rpt.get('overall', {})
            b = overall.get('biceps', {}) or {}
            t = overall.get('triceps', {}) or {}
            if b.get('r2') is not None:
                parts.append(f'模型准确度：二头肌R方{b["r2"]:.2f}，三头肌R方{t.get("r2", 0):.2f}。')
        else:
            parts.append('交叉验证报告尚未生成，请先运行训练采集数据。')

        # 训练
        live = self.state.get_training_live()
        if live:
            parts.append(f'当前训练进行中，已{live["duration"]:.0f}秒，{live["reps"]}次。')

        # 质量告警
        alerts, alert_age = self.state.get_alerts()
        if alerts and alert_age < 30:
            for a in alerts[:2]:
                if a['type'] == 'compensation':
                    parts.append(f'注意：检测到{a["level"]}肌肉代偿。')
                elif a['type'] == 'electrode':
                    if a.get('dropout_b') or a.get('dropout_t'):
                        parts.append('警告：电极信号脱落，请检查电极贴合。')
                    else:
                        parts.append('提示：电极接触不良，请调整位置。')
        return ' '.join(parts)

    def _cmd_angle(self):
        angle, valid = self.state.get_angle()
        if valid and angle is not None:
            if angle < 30:
                desc = '手臂接近伸直'
            elif angle < 60:
                desc = '手臂轻微弯曲'
            elif angle < 100:
                desc = '手臂弯曲约九十度'
            else:
                desc = '手臂接近最大弯曲'
            return f'当前肘关节角度{angle:.0f}度，{desc}。'
        else:
            return '角度传感器暂未连接，请确认摄像头或骨架追踪已启动。'

    def _cmd_muscle(self):
        emg_b, emg_t, age = self.state.get_emg()
        if emg_b is None or age > 5:
            return '肌电数据暂未更新，请确认EMG桥接节点已启动。'

        parts = [f'二头肌{emg_b:.0f}微伏，三头肌{emg_t:.0f}微伏。']

        # 定性评估
        if emg_b < 100:
            parts.append('二头肌处于放松状态。')
        elif emg_b < 400:
            parts.append('二头肌轻度激活。')
        elif emg_b < 800:
            parts.append('二头肌中度收缩。')
        else:
            parts.append('二头肌强力收缩。')

        # 比对
        val_rpt, val_age = self.state.get_validation()
        if val_rpt and val_age < 60:
            overall = val_rpt.get('overall', {})
            b_bias = (overall.get('biceps', {}) or {}).get('bias', 0)
            t_bias = (overall.get('triceps', {}) or {}).get('bias', 0)
            if abs(b_bias) > 50:
                direction = '偏高' if b_bias > 0 else '偏低'
                parts.append(f'注意：二头肌预测值系统性{direction}，建议重新校准。')

        return ' '.join(parts)

    def _cmd_diagnosis(self):
        val_rpt, val_age = self.state.get_validation()
        if not val_rpt or val_age > 60:
            return '暂无交叉验证报告。请先运行交叉验证节点采集数据，至少需要十组样本。'

        overall = val_rpt.get('overall', {})
        diag = val_rpt.get('diagnosis', [])

        if not diag:
            return '交叉验证数据不足，请继续训练以生成诊断报告。'

        # 解析诊断
        parts = ['系统诊断报告：']
        for d in diag:
            if d.startswith('R '):
                parts.append(f'严重：{d[2:]}。')
            elif d.startswith('Y '):
                parts.append(f'注意：{d[2:]}。')
            elif d.startswith('G '):
                parts.append(f'良好：{d[2:]}。')
            elif d.startswith('C '):
                parts.append(f'校准建议：{d[2:]}。')
            elif d.startswith('W '):
                parts.append(f'警告：{d[2:]}。')
            else:
                parts.append(d)

        # 代偿分析
        comp = val_rpt.get('compensation', {})
        if comp:
            cs = comp.get('compensation_score', 0)
            cl = comp.get('compensation_level', 'unknown')
            if cl == 'severe':
                parts.append(f'肌肉代偿评分{cs:.0f}分，严重，二头肌发力严重不足。')
            elif cl == 'moderate':
                parts.append(f'肌肉代偿评分{cs:.0f}分，中度，请注意发力模式。')
            elif cl == 'mild':
                parts.append(f'肌肉代偿评分{cs:.0f}分，轻度。')

        # 电极健康
        elec = val_rpt.get('electrode_health', {})
        if elec:
            qb = elec.get('biceps_quality', 100)
            qt = elec.get('triceps_quality', 100)
            if qb < 50 or qt < 50:
                parts.append(f'电极接触质量：二头肌{qb:.0f}分，三头肌{qt:.0f}分。请检查电极贴合。')
            if elec.get('dropout_b'):
                parts.append('二头肌电极存在脱落。')
            if elec.get('dropout_t'):
                parts.append('三头肌电极存在脱落。')

        return ' '.join(parts)

    def _cmd_start_training(self):
        if self.state.is_training:
            return '训练已经在进行中，请说"停止训练"来结束。'

        self.state.start_training()
        return '康复训练已开始。请缓慢弯曲和伸直手肘，系统会自动记录次数和角度。说"停止训练"结束。'

    def _cmd_stop_training(self):
        if not self.state.is_training:
            return '当前没有进行中的训练。说"开始训练"启动。'

        report = self.state.stop_training()
        mins = int(report['duration_sec'] // 60)
        secs = int(report['duration_sec'] % 60)

        parts = [
            f'训练已停止。',
            f'用时{mins}分{secs}秒，',
            f'完成{report["rep_count"]}次屈伸，',
            f'最大角度{report["max_angle"]:.0f}度，',
            f'平均角度{report["avg_angle"]:.0f}度，',
            f'平均二头肌肌电{report["avg_emg_biceps"]:.0f}微伏。',
        ]
        return ' '.join(parts)

    def _cmd_training_report(self):
        if self.state.is_training:
            report = self.state.stop_training()
        else:
            # 没有活跃训练, 无数据
            return '没有训练数据。请先说"开始训练"启动训练。'

        if report is None:
            return '没有训练数据。请先说"开始训练"启动训练。'

        mins = int(report['duration_sec'] // 60)
        secs = int(report['duration_sec'] % 60)
        parts = [
            f'训练报告：',
            f'用时{mins}分{secs}秒，',
            f'完成{report["rep_count"]}次屈伸，',
            f'最大角度{report["max_angle"]:.0f}度，',
            f'平均角度{report["avg_angle"]:.0f}度，',
            f'平均二头肌肌电{report["avg_emg_biceps"]:.0f}微伏，',
            f'平均三头肌肌电{report["avg_emg_triceps"]:.0f}微伏。',
        ]
        return ' '.join(parts)

    def _cmd_start_calib(self):
        """启动校准引导流程"""
        if self._calib_active:
            return '校准已在执行中，请跟随语音提示完成动作。'

        self.state.start_calibration()
        self._calib_active = True
        self._calib_last_prompt = None
        return None  # 校准由主循环的 calib_tick 驱动

    def _calib_loop(self):
        """主循环中调用: 驱动校准状态机"""
        if not self._calib_active:
            return

        phase, prompt, is_done = self.state.calib_tick()

        if is_done:
            result = self.state.get_calib_result()
            self._calib_active = False
            self.state.calib_phase = 'idle'
            self.tts.speak(
                f'校准完成。'
                f'二头肌：放松{result["rest_b"]:.0f}微伏，九十度{result["v90_b"]:.0f}微伏。'
                f'三头肌：放松{result["rest_t"]:.0f}微伏，九十度{result["v90_t"]:.0f}微伏。'
                f'请记下这些数值用于系统配置。'
            )
            return

        if prompt and prompt != self._calib_last_prompt:
            self._calib_last_prompt = prompt
            self.tts.speak(prompt)

    def _cmd_help(self):
        return (
            '可用指令如下：'
            '你好 — 系统问候。'
            '检查状态 — 查看系统运行状态和实时数据。'
            '当前角度 — 读取肘关节角度。'
            '肌肉状态 — 查看肌电数据和激活程度。'
            '诊断 — 系统自检和模型准确度评估。'
            '开始训练 — 启动训练，自动计次计时。'
            '停止训练 — 停止训练并播报结果。'
            '训练报告 — 查看训练统计数据。'
            '开始校准 — 语音引导完成校准流程。'
            '帮助 — 列出所有指令。'
            '再见 — 退出语音助手。'
        )

    # ---- 主循环 ----

    def run_push_to_talk(self, model):
        """按键触发模式"""
        print('=' * 55)
        print('  智能健康袖套 — 语音助手 v6 [按键模式]')
        print(f'  麦克风: {MIC_DEV}')
        print('  指令: 你好 | 状态 | 角度 | 肌肉 | 诊断 | 训练 | 校准 | 帮助 | 再见')
        print('  按 Enter 开始说话，Ctrl+C 退出')
        print('=' * 55)

        self.tts.speak('语音助手已启动，请按回车键开始说话。')

        while self.running:
            try:
                # 先处理校准状态机 (非阻塞)
                self._calib_loop()

                # 等待用户按键 (0.5s超时用于校准状态机)
                try:
                    import select
                    if select.select([sys.stdin], [], [], 0.5)[0]:
                        sys.stdin.readline()
                    else:
                        continue  # 超时, 继续循环检查校准
                except Exception:
                    # Windows / 无 select 的 fallback
                    try:
                        input('\n[按 Enter 开始录音]')
                    except EOFError:
                        time.sleep(0.5)
                        continue

                # 倒计时
                for i in [3, 2, 1]:
                    print(f'{i}...', end='', flush=True)
                    time.sleep(0.7)
                print('说！')

                # 录音
                wav = '/tmp/voice_cmd.wav'
                record_wav(wav, RECORD_SEC)

                # 预处理
                wav_norm, peak, zcr = preprocess_audio(wav)
                print(f'  峰值={peak:.0f}  过零率={zcr:.3f}  ', end='')

                if zcr < 0.02:
                    print('⚠️ 信号异常')
                    self.tts.speak('未检测到语音信号，请检查麦克风是否连接。')
                    self._cleanup_wavs(wav, wav_norm)
                    continue

                # Whisper 识别
                print('识别中...')
                t0 = time.time()
                segments, info = model.transcribe(wav_norm, language='zh',
                                                  beam_size=5, vad_filter=True)
                text = ''.join(seg.text for seg in segments)
                elapsed = time.time() - t0

                if text.strip():
                    text = text.strip()
                    print(f'  [{elapsed:.1f}s] 识别: "{text}"')

                    cmd_id = self.match_command(text)
                    if cmd_id:
                        print(f'  >>> 执行: {cmd_id}')
                        response = self.handle(cmd_id)
                        if response:
                            self.tts.speak(response)
                    else:
                        print('  >>> 未匹配指令, 回显')
                        self.tts.speak(f'你说的是：{text}。如需帮助，请说"帮助"。')
                else:
                    print(f'  [{elapsed:.1f}s] 未检测到语音')
                    self.tts.speak('抱歉，没有听清，请再说一次。')

                self._cleanup_wavs(wav, wav_norm)

            except KeyboardInterrupt:
                print('\n退出')
                break
            except Exception as e:
                print(f'  错误: {e}')
                import traceback
                traceback.print_exc()
                continue

    def run_vad(self, model):
        """VAD 自动触发模式"""
        print('=' * 55)
        print('  智能健康袖套 — 语音助手 v6 [VAD模式]')
        print(f'  麦克风: {MIC_DEV}')
        print('  语音活动检测中... 直接说话即可')
        print('  Ctrl+C 退出')
        print('=' * 55)

        self.tts.speak('语音助手已启动，直接对我说话即可。')

        while self.running:
            try:
                # 处理校准状态机
                self._calib_loop()

                # VAD 录音
                print('  🎧 监听中...', end='', flush=True)
                wav = '/tmp/voice_cmd_vad.wav'
                detected, peak, zcr = record_vad(wav)

                if not detected:
                    print(' (静默)')
                    continue

                # 预处理
                wav_norm, peak2, zcr2 = preprocess_audio(wav)
                if zcr2 < 0.02:
                    print(f'  ⚠️ 信号异常 ZCR={zcr2:.3f}')
                    self._cleanup_wavs(wav, wav_norm)
                    continue

                # Whisper 识别
                print(f'  识别中...', end='', flush=True)
                t0 = time.time()
                segments, info = model.transcribe(wav_norm, language='zh',
                                                  beam_size=5, vad_filter=True)
                text = ''.join(seg.text for seg in segments)
                elapsed = time.time() - t0

                if text.strip():
                    text = text.strip()
                    print(f' [{elapsed:.1f}s] "{text}"')

                    cmd_id = self.match_command(text)
                    if cmd_id:
                        print(f'  >>> {cmd_id}')
                        response = self.handle(cmd_id)
                        if response:
                            self.tts.speak(response)
                    else:
                        # 未匹配: 不打断用户, 除非是清晰的短句
                        if len(text) > 3:
                            self.tts.speak(f'你说的是：{text}')
                else:
                    print(f' [{elapsed:.1f}s] 空')

                self._cleanup_wavs(wav, wav_norm)

            except KeyboardInterrupt:
                print('\n退出')
                break
            except Exception as e:
                print(f'  错误: {e}')
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
                continue

    def _cleanup_wavs(self, *paths):
        for p in paths:
            if p and os.path.exists(p):
                try: os.unlink(p)
                except Exception: pass


# ============================== 主入口 ==============================

def main():
    parser = argparse.ArgumentParser(description='智能健康袖套语音助手 v6')
    parser.add_argument('--vad', action='store_true', help='VAD自动触发模式')
    parser.add_argument('--daemon', action='store_true', help='后台守护模式 (等同于 --vad)')
    args = parser.parse_args()

    use_vad = args.vad or args.daemon

    # 1. 初始化共享状态
    state = SystemState()

    # 2. 启动 ROS2 后台线程
    print('[语音] 启动 ROS2 监听...')
    ros2_thread = threading.Thread(target=ros2_spin_thread, args=(state,), daemon=True)
    ros2_thread.start()
    time.sleep(1.5)  # 等待 ROS2 初始化

    # 3. 加载 Whisper
    print('[语音] 加载 Whisper base 模型...')
    model = WhisperModel(MODEL_PATH, device='cpu', compute_type='int8')

    # 4. 初始化 TTS
    tts = TTSSpeaker()
    if tts.engine is None:
        print('[语音] 警告: 无可用 TTS 引擎, 仅文字输出')

    # 5. 启动语音助手
    assistant = VoiceAssistant(state, tts, use_vad=use_vad)

    if use_vad:
        assistant.run_vad(model)
    else:
        assistant.run_push_to_talk(model)

    print('[语音] 助手已退出')


if __name__ == '__main__':
    main()
