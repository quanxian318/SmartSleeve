#!/usr/bin/env python3
"""
语音交互 v7 — 智能训练助手：暂停/恢复、实时警告、定期播报、AI对话

新增功能 vs v6:
  - 训练暂停/恢复控制
  - 后台 TrainingMonitor：每5s检查告警、进度、分步提示
  - TTS 优先级队列 + 打断：电极脱落可中断当前播报
  - DeepSeek API：自然对话 + 训练总结（网络不可用时静默降级）
  - 分步提示：每步告诉用户当前可说的指令

用法:
  python3 voice_demo_v7.py                  # 按键模式
  python3 voice_demo_v7.py --vad            # VAD模式
  python3 voice_demo_v7.py --daemon         # 后台守护模式
  python3 voice_demo_v7.py --no-llm         # 禁用 AI 对话
"""

import subprocess, os, sys, time, wave, struct, json, math, queue, signal
import threading, argparse, random
from collections import deque
import numpy as np
from faster_whisper import WhisperModel

# LLM 客户端
try:
    from voice_llm import DeepSeekClient
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False
    print("[语音] voice_llm.py 未找到, AI对话功能不可用")

# ============================== 配置 ==============================

def _find_whisper_model():
    """优先使用 tiny 模型 (ARM64上比 base 快 3x, 指令识别精度够用)"""
    import glob
    # 检查已下载的 tiny 模型
    tiny_base = '/opt/whisper-models/models--Systran--faster-whisper-tiny/snapshots'
    if os.path.isdir(tiny_base):
        dirs = sorted(glob.glob(os.path.join(tiny_base, '*')))
        for d in dirs:
            if os.path.isfile(os.path.join(d, 'model.bin')):
                return d
    # 回退到 base
    base = '/opt/whisper-models/models--Systran--faster-whisper-base/snapshots'
    if os.path.isdir(base):
        dirs = sorted(glob.glob(os.path.join(base, '*')))
        for d in dirs:
            if os.path.isfile(os.path.join(d, 'model.bin')):
                return d
    # 最后: 让 faster-whisper 自动下载 tiny
    return 'tiny'

MODEL_PATH = _find_whisper_model()
MIC_DEV    = 'plughw:0,0'     # Jieli 无线麦
RATE       = 16000  # 必须与 Whisper 模型输入一致, preprocess_audio 不做重采样
RECORD_SEC = 4  # 最大录音时长
VAD_ENERGY_THRESHOLD = 300
VAD_SILENCE_SEC = 0.4  # 静音判定阈值（秒），降低尾延迟
VAD_CHUNK_SEC = 0.5  # USB麦克风最小稳定录音粒度, 0.2秒会触发pcm_read中断

# 告警冷却时间 (秒)
ALERT_COOLDOWNS = {
    'electrode_dropout': 20,
    'electrode_impedance': 45,
    'compensation_severe': 25,
    'compensation_moderate': 45,
    'compensation_mild': 60,
    'form_wrong': 30,
    'heart_rate_high': 30,
    'heart_rate_low': 30,
}

# 进度播报间隔
PROGRESS_INTERVAL_SEC = 120     # 每2分钟
REP_MILESTONES = [10, 20, 30, 50, 100]

# 无人声超时提醒 (秒, 播报文字) — 逐级递增
INACTIVITY_LEVELS = [
    (60,  '没有听到您说话。如果需要帮助，可以说"帮助"查看可用指令。'),
    (180, '已经有一段时间没有听到您说话了，我还在监听中。需要帮助随时叫我。'),
    (600, '系统仍在运行，等待您的语音指令。说"帮助"了解我能做什么。'),
]

# 音量控制 (ALSA card 2: SPKL/SPKR, 范围 0-191)
VOLUME_CARD = 2
VOLUME_MAX_RAW = 191
VOLUME_STEP_PCT = 10  # 每次调 10%
VOLUME_STEP_RAW = round(VOLUME_MAX_RAW * VOLUME_STEP_PCT / 100)  # 19


# ============================== PriorityTTSSpeaker ==============================

class PriorityTTSSpeaker:
    """优先级 TTS：0=普通 1=提示 2=警告 3=严重。高优先可打断低优先。
    内置 WAV 缓存，相同文本只合成一次。"""

    # 启动时预缓存的常用语 (文本 → 缓存key)
    PRECACHE_TEXTS = [
        # 基础问候
        '你好，我是智能健康袖套语音助手。请说"帮助"查看可用指令。',
        '再见，祝您康复顺利！',
        '不客气，这是我应该做的。祝您康复顺利！',
        '抱歉，没有听清，请再说一次。',
        '未检测到语音信号，请检查麦克风是否连接。',
        '录音失败，请检查麦克风是否正常连接。',
        # 帮助
        '当前训练模式为肘关节屈伸。使用流程：首先说"开始校准"，然后"开始训练"。训练中可"暂停训练"或"继续训练"，完成后"结束训练"。数据查询：检查状态、当前角度、肌肉状态。分析报告：训练报告、AI报告、诊断。',
        # 开始训练 - 全部分支
        '训练已经在进行中。你可以说"暂停训练"休息，或"结束训练"停止。',
        '训练当前已暂停。说"继续训练"恢复即可。',
        '校准正在进行中，请先完成校准再开始训练。',
        '请先进行校准再开始训练。说"开始校准"，跟随语音提示完成校准流程。',
        '请先进行语音校准再开始训练。说"开始校准"，跟随提示完全放松再弯曲至九十度。',
        '校准数据读取失败，请重新"开始校准"。',
        '肘关节屈伸训练已开始。请从手臂伸直开始，缓慢弯曲至最大角度，再缓慢伸直回原位。系统会自动记录每次完整屈伸。说"暂停训练"休息，"结束训练"停止。',
        # 暂停
        '当前没有进行中的训练。说"开始训练"启动。',
        '训练已经暂停。说"继续训练"恢复。',
        '训练已暂停。休息好了说"继续训练"恢复。',
        '暂停失败，请重试。',
        # 继续
        '训练正在运行中。',
        '训练已恢复。继续加油！',
        '恢复失败，请重试。',
        # 校准
        '开始校准模式。请将手臂放在90度位置，保持稳定3秒。',
        '校准完成！现在可以开始训练了。',
        '校准失败，请重新说"开始校准"。',
        '请先自然垂下手臂，完全放松，不要用力。保持放松5秒。',
        '很好。现在请缓慢弯曲手臂到90度位置，保持这个姿势5秒。',
        '校准数据已保存。',
        # 诊断
        '诊断功能需要AI支持，请先配置DeepSeek API密钥。',
        # 提示（TrainingMonitor用）
        '请伸直手臂。',
        '请弯曲手臂。',
        '保持稳定。',
        '做得很好！',
        '注意不要耸肩。',
        # 无人声超时提醒
        '没有听到您说话。如果需要帮助，可以说"帮助"查看可用指令。',
        '已经有一段时间没有听到您说话了，我还在监听中。需要帮助随时叫我。',
        '系统仍在运行，等待您的语音指令。说"帮助"了解我能做什么。',
        # 音量控制
        '音量已调大，当前约百分之',
        '音量已调小，当前约百分之',
        '音量已调至最大。',
        '音量已调至最小。',
        '当前音量约百分之',
        '音量已设为百分之',
        # 心率预警
        '警告：心率过低！当前心率低于40次每分钟，请注意身体状况。',
        '警告：心率过高！当前心率超过180次每分钟，请立即停止运动并休息。',
    ]

    def __init__(self, cooldown_sec=0.5):
        import shutil, hashlib
        self._has_edge = shutil.which('edge-tts') is not None
        self._has_espeak = shutil.which('espeak-ng') is not None
        self._has_ffmpeg = shutil.which('ffmpeg') is not None
        self.engine = 'edge' if self._has_edge else ('espeak' if self._has_espeak else None)
        self._hashlib = hashlib

        self._cache_dir = '/tmp/tts_cache'
        os.makedirs(self._cache_dir, exist_ok=True)

        self._queue = queue.PriorityQueue()
        self._seq = 0           # 同优先级 FIFO
        self._player = None     # 当前 aplay Popen 句柄
        self._stopping = threading.Event()
        self._running = True
        self._speaking = False  # TTS 正在播放中（VAD 用）
        self._last_tts_end = 0  # TTS 结束时间戳
        self.cooldown_sec = cooldown_sec  # TTS 播完后冷却时间, 防止音箱回声被麦克风录入
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def prewarm_cache(self):
        """后台预生成常用语 WAV 缓存"""
        def _warm():
            for text in self.PRECACHE_TEXTS:
                try:
                    self._get_or_generate_wav(text)
                except Exception:
                    pass
        t = threading.Thread(target=_warm, daemon=True)
        t.start()

    @property
    def is_speaking(self):
        return self._speaking

    @property
    def tts_cooldown_active(self):
        """TTS 刚结束后冷却期内不监听，避免回音"""
        return (time.time() - self._last_tts_end) < self.cooldown_sec

    def speak(self, text, priority=0):
        """入队播放。priority: 0=普通 1=提示 2=警告 3=严重"""
        if not text:
            return
        self._seq += 1
        # PriorityQueue 是最小堆，用负优先级确保高优先先出
        self._queue.put((-priority, self._seq, text))

    def speak_now(self, text, priority=3):
        """立即播放：终止当前 aplay，清空队列，直接播放"""
        if not text:
            return
        self._interrupt_current()
        # 清空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._seq += 1
        self._speak_one(text, priority)

    def stop(self):
        """停止播放并退出工作线程"""
        self._running = False
        self._interrupt_current()
        # 放入哨兵唤醒工作线程
        self._queue.put((0, 0, None))

    # ---- 内部 ----

    def _interrupt_current(self):
        """安全终止当前 aplay 进程"""
        self._stopping.set()
        if self._player is not None:
            try:
                if self._player.poll() is None:
                    self._player.terminate()
                    self._player.wait(timeout=2)
            except Exception:
                try:
                    self._player.kill()
                except Exception:
                    pass
            self._player = None

    def _worker(self):
        """后台消费队列"""
        while self._running:
            try:
                neg_priority, seq, text = self._queue.get(timeout=1)
                if text is None:  # 哨兵
                    break
                priority = -neg_priority
                # 检查是否需要打断当前播放
                self._stopping.clear()
                self._speak_one(text, priority)
            except queue.Empty:
                continue

    def _speak_one(self, text, priority=0):
        """播放单条文本，直到完成或被 _stopping 事件打断"""
        if not text:
            return
        self._speaking = True
        print(f'  🔊 [{self._priority_name(priority)}] {text}')
        if self.engine == 'edge' and self._has_edge:
            self._speak_edge(text)
        elif self._has_espeak:
            self._speak_espeak(text)
        self._speaking = False
        self._last_tts_end = time.time()

    def _speak_edge(self, text):
        try:
            # 检查缓存
            cached = self._get_cached_wav(text)
            if cached:
                if self._stopping.is_set():
                    return
                self._player = subprocess.Popen(
                    ['aplay', '-q', cached],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self._player.wait(timeout=30)
                self._player = None
                return

            # 缓存未命中，合成并缓存
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                mp3_path = f.name
            wav_path = mp3_path.replace('.mp3', '.wav')

            subprocess.run([
                'edge-tts', '--text', text,
                '--voice', 'zh-CN-XiaoxiaoNeural',
                '--rate', '+0%', '--volume', '+0%',
                '--write-media', mp3_path
            ], capture_output=True, timeout=10)  # 微软TTS通常在2-4s内返回

            if self._stopping.is_set():
                self._cleanup_files(mp3_path, wav_path)
                return

            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                subprocess.run(['ffmpeg', '-i', mp3_path, '-y', wav_path],
                               capture_output=True, timeout=10)

                if self._stopping.is_set():
                    self._cleanup_files(mp3_path, wav_path)
                    return

                if os.path.exists(wav_path):
                    # 存入缓存
                    self._save_to_cache(text, wav_path)
                    self._player = subprocess.Popen(
                        ['aplay', '-q', wav_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    self._player.wait(timeout=30)
                    self._player = None

            self._cleanup_files(mp3_path, wav_path)

        except Exception as e:
            print(f'  [TTS edge] 失败: {e}')
            if self._has_espeak:
                self._speak_espeak(text)

    def _speak_espeak(self, text):
        try:
            self._player = subprocess.Popen(
                ['espeak-ng', '-v', 'cmn', '-s', '160', '-p', '50', '--', text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._player.wait(timeout=15)
            self._player = None
        except Exception as e:
            print(f'  [TTS espeak] 失败: {e}')

    def _cache_key(self, text):
        """文本 → MD5 缓存键"""
        return self._hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_cached_wav(self, text):
        """检查文本是否有缓存 WAV，有则返回路径，无则返回 None"""
        key = self._cache_key(text)
        wav_path = os.path.join(self._cache_dir, key + '.wav')
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
            return wav_path
        return None

    def _save_to_cache(self, text, src_wav):
        """将合成好的 WAV 复制到缓存"""
        try:
            key = self._cache_key(text)
            dst = os.path.join(self._cache_dir, key + '.wav')
            import shutil as _shutil
            _shutil.copy2(src_wav, dst)
        except Exception:
            pass

    def _get_or_generate_wav(self, text):
        """获取或生成缓存的 WAV。供 prewarm 使用。"""
        cached = self._get_cached_wav(text)
        if cached:
            return cached
        # 同步合成 (仅在 prewarm 阶段使用)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            mp3_path = f.name
        wav_path = mp3_path.replace('.mp3', '.wav')
        try:
            subprocess.run([
                'edge-tts', '--text', text,
                '--voice', 'zh-CN-XiaoxiaoNeural',
                '--rate', '+0%', '--volume', '+0%',
                '--write-media', mp3_path
            ], capture_output=True, timeout=15)
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                subprocess.run(['ffmpeg', '-i', mp3_path, '-y', wav_path],
                               capture_output=True, timeout=10)
                if os.path.exists(wav_path):
                    self._save_to_cache(text, wav_path)
                    return wav_path
        except Exception:
            pass
        finally:
            self._cleanup_files(mp3_path, wav_path)
        return None

    def _cleanup_files(self, *paths):
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass

    @staticmethod
    def _priority_name(priority):
        names = {0: '普通', 1: '提示', 2: '⚠警告', 3: '🚨严重'}
        return names.get(priority, '?')


# ============================== SystemState (增强) ==============================

class SystemState:
    """线程安全系统状态：角度/EMG/验证/告警 + 训练状态机 + 校准"""

    def __init__(self):
        self._lock = threading.Lock()

        # 传感器
        self.left_angle = None
        self.right_angle = None
        self.left_valid = False
        self.right_valid = False
        self.angle_timestamp = 0

        # EMG
        self.emg_biceps = None
        self.emg_triceps = None
        self.emg_timestamp = 0

        # 交叉验证 + 告警
        self.validation_report = None
        self.validation_timestamp = 0
        self.latest_alerts = []
        self.alerts_timestamp = 0

        # 心率
        self.bpm = None
        self.bpm_timestamp = 0

        # 训练
        self.is_training = False
        self.training_paused = False
        self.training_start_time = 0
        self.training_rep_count = 0
        self.training_angles = []
        self.training_emg_b = []
        self.training_emg_t = []
        self._rep_state = 'waiting'
        self._rep_peak_angle = 0
        self._rep_min_angle = 180
        self._rep_debounce = 0      # 连续帧确认计数器(防抖动)

        # 暂停计时
        self.pause_start_time = 0
        self.total_paused_seconds = 0.0
        self.pause_count = 0

        # 训练阶段 (for step prompts)
        self.training_phase = 'idle'   # idle | active | paused | completed
        self.last_rep_announced = 0

        # 告警频率控制
        self._alert_spoken_times = {}
        self._announced_alert_ids = set()

        # 训练历史 (LLM 总结用, 最多10条)
        self.training_history = []

        # 异常计数
        self.compensation_alerts = 0
        self.electrode_alerts = 0
        self.form_alerts = 0

        # 校准
        self.calib_phase = 'idle'
        self.calib_rest_b = None
        self.calib_rest_t = None
        self.calib_v90_b = None
        self.calib_v90_t = None
        self.calib_phase_start = 0
        self.calib_buffer = []  # 校准窗口采样缓冲 [(b,t), ...]

    # ---- 传感器更新 ----
    def update_angle(self, side, angle, valid):
        with self._lock:
            if side == 'left':
                self.left_angle = angle
                self.left_valid = valid
            else:
                self.right_angle = angle
                self.right_valid = valid
            self.angle_timestamp = time.time()

            if self.is_training and not self.training_paused and valid and angle is not None:
                self.training_angles.append((time.time(), angle))
                self._count_rep(angle)

    def update_emg(self, biceps, triceps):
        with self._lock:
            self.emg_biceps = biceps
            self.emg_triceps = triceps
            self.emg_timestamp = time.time()

            if self.is_training and not self.training_paused:
                self.training_emg_b.append(biceps)
                self.training_emg_t.append(triceps)

            # 校准时收集样本
            if self.calib_phase in ('relax', 'bend'):
                if biceps is not None and triceps is not None:
                    self.calib_buffer.append((biceps, triceps))
                    # 只保留最近 2 秒 (~20 samples at 10Hz)
                    if len(self.calib_buffer) > 20:
                        self.calib_buffer.pop(0)

    def update_validation(self, report):
        with self._lock:
            self.validation_report = report
            self.validation_timestamp = time.time()

    def update_alerts(self, alerts):
        with self._lock:
            self.latest_alerts = alerts
            self.alerts_timestamp = time.time()

    def update_bpm(self, bpm):
        with self._lock:
            self.bpm = bpm
            self.bpm_timestamp = time.time()

    # ---- 训练控制 ----
    def start_training(self):
        with self._lock:
            self.is_training = True
            self.training_paused = False
            self.training_start_time = time.time()
            self.training_rep_count = 0
            self.training_angles = []
            self.training_emg_b = []
            self.training_emg_t = []
            self._rep_state = 'waiting'
            self._rep_peak_angle = 0
            self._rep_min_angle = 180
            self._rep_debounce = 0
            self.total_paused_seconds = 0.0
            self.pause_count = 0
            self.training_phase = 'active'
            self.last_rep_announced = 0
            self.compensation_alerts = 0
            self.electrode_alerts = 0
            self.form_alerts = 0

    def pause_training(self):
        with self._lock:
            if not self.is_training or self.training_paused:
                return False
            self.training_paused = True
            self.pause_start_time = time.time()
            self.pause_count += 1
            self.training_phase = 'paused'
            return True

    def resume_training(self):
        with self._lock:
            if not self.is_training or not self.training_paused:
                return False
            self.training_paused = False
            self.total_paused_seconds += time.time() - self.pause_start_time
            self.training_phase = 'active'
            return True

    def stop_training(self):
        with self._lock:
            # 先记录暂停状态, 再复位 (避免读取已复位的 training_paused)
            was_paused = self.training_paused
            pause_start = self.pause_start_time

            self.is_training = False
            self.training_paused = False
            self.training_phase = 'completed'

            now = time.time()
            total_dur = now - self.training_start_time if self.training_start_time else 0
            # 如果还在暂停中，加上最后一段暂停时长
            extra_pause = 0
            if was_paused and pause_start > 0:
                extra_pause = now - pause_start
            eff_dur = max(0, total_dur - self.total_paused_seconds - extra_pause)

            angles = [a for _, a in self.training_angles] if self.training_angles else [0]
            emg_b = self.training_emg_b if self.training_emg_b else [0]
            emg_t = self.training_emg_t if self.training_emg_t else [0]

            report = {
                'total_duration_sec': round(total_dur, 1),
                'effective_duration_sec': round(eff_dur, 1),
                'rep_count': self.training_rep_count,
                'target_reps': 0,
                'pause_count': self.pause_count,
                'max_angle': round(max(angles), 1),
                'min_angle': round(min(angles), 1),
                'avg_angle': round(sum(angles) / len(angles), 1),
                'avg_emg_biceps': round(sum(emg_b) / len(emg_b), 1),
                'avg_emg_triceps': round(sum(emg_t) / len(emg_t), 1),
                'compensation_count': self.compensation_alerts,
                'electrode_issue_count': self.electrode_alerts,
                'form_issue_count': self.form_alerts,
                'angle_samples': len(angles),
            }
            # 保存到历史
            self.training_history.append(report)
            if len(self.training_history) > 10:
                self.training_history = self.training_history[-10:]

            return report

    # ---- 训练计次 (带防抖) ----
    DEBOUNCE_FRAMES = 3  # 连续3帧确认才触发状态迁移

    def _count_rep(self, angle):
        if self.training_paused:
            return

        if self._rep_state == 'waiting':
            if angle > 60:
                self._rep_debounce += 1
                if self._rep_debounce >= self.DEBOUNCE_FRAMES:
                    self._rep_state = 'bending'
                    self._rep_peak_angle = angle
                    self._rep_debounce = 0
            else:
                self._rep_debounce = 0

        elif self._rep_state == 'bending':
            if angle > self._rep_peak_angle:
                self._rep_peak_angle = angle
                self._rep_debounce = 0  # 仍在弯曲, 重置下降计数
            elif angle < self._rep_peak_angle - 15:
                self._rep_debounce += 1
                if self._rep_debounce >= self.DEBOUNCE_FRAMES:
                    self._rep_state = 'extending'
                    self._rep_min_angle = angle
                    self._rep_debounce = 0
            else:
                self._rep_debounce = 0

        elif self._rep_state == 'extending':
            if angle < self._rep_min_angle:
                self._rep_min_angle = angle
                self._rep_debounce = 0  # 仍在伸展, 重置上升计数
            elif angle > self._rep_min_angle + 10:
                self._rep_debounce += 1
                if self._rep_debounce >= self.DEBOUNCE_FRAMES:
                    self.training_rep_count += 1
                    self._rep_state = 'bending'
                    self._rep_peak_angle = angle
                    self._rep_debounce = 0
            else:
                self._rep_debounce = 0

    # ---- 告警频率控制 ----
    def should_speak_alert(self, alert_type):
        """检查告警类型是否已过冷却时间"""
        with self._lock:
            last = self._alert_spoken_times.get(alert_type, 0)
            cooldown = ALERT_COOLDOWNS.get(alert_type, 60)
            return (time.time() - last) >= cooldown

    def record_alert_spoken(self, alert_type):
        """记录告警已播报"""
        with self._lock:
            self._alert_spoken_times[alert_type] = time.time()

    def has_alert_been_announced(self, alert_id):
        with self._lock:
            return alert_id in self._announced_alert_ids

    def mark_alert_announced(self, alert_id):
        with self._lock:
            self._announced_alert_ids.add(alert_id)
            # 防止集合无限增长
            if len(self._announced_alert_ids) > 100:
                self._announced_alert_ids.clear()

    def increment_alert_counter(self, alert_type):
        with self._lock:
            if alert_type == 'compensation':
                self.compensation_alerts += 1
            elif alert_type == 'electrode':
                self.electrode_alerts += 1
            elif alert_type == 'form':
                self.form_alerts += 1

    # ---- 读取 ----
    def get_angle(self):
        with self._lock:
            return self.left_angle, self.left_valid

    def get_emg(self):
        with self._lock:
            age = time.time() - self.emg_timestamp if self.emg_timestamp else 999
            return self.emg_biceps, self.emg_triceps, age

    def get_alerts(self):
        with self._lock:
            age = time.time() - self.alerts_timestamp if self.alerts_timestamp else 999
            return list(self.latest_alerts), age

    def get_bpm(self):
        with self._lock:
            age = time.time() - self.bpm_timestamp if self.bpm_timestamp else 999
            return self.bpm, age

    def get_validation(self):
        with self._lock:
            age = time.time() - self.validation_timestamp if self.validation_timestamp else 999
            return self.validation_report, age

    def get_training_live(self):
        with self._lock:
            if not self.is_training:
                return None
            return {
                'duration': round(time.time() - self.training_start_time, 1),
                'effective_duration': round(
                    time.time() - self.training_start_time - self.total_paused_seconds
                    - (time.time() - self.pause_start_time if self.training_paused else 0), 1
                ),
                'reps': self.training_rep_count,
                'angle': self.left_angle,
                'paused': self.training_paused,
                'phase': self.training_phase,
            }

    def get_last_completed_training(self):
        with self._lock:
            return self.training_history[-1] if self.training_history else None

    # ---- 校准 ----
    def start_calibration(self):
        with self._lock:
            self.calib_phase = 'relax'
            self.calib_phase_start = time.time()
            self.calib_rest_b = None
            self.calib_rest_t = None
            self.calib_v90_b = None
            self.calib_v90_t = None
            self.calib_buffer = []

    def finish_calibration(self):
        """校准完成, 安全复位 phase (线程安全)"""
        with self._lock:
            self.calib_phase = 'idle'

    # 有效 EMG 范围: 排除噪声(<5µV)和饱和(>10000µV)
    EMG_VALID_MIN = 5.0
    EMG_VALID_MAX = 10000.0

    def _calib_window_mean(self):
        """取校准缓冲区最近样本的平均值 (过滤异常尖峰)"""
        if not self.calib_buffer:
            return None, None
        # 取最近 1.5 秒 (后 15 个样本), 排除离群值(±3σ)
        recent = self.calib_buffer[-15:] if len(self.calib_buffer) >= 15 else self.calib_buffer
        b_vals = [v[0] for v in recent if self.EMG_VALID_MIN < v[0] < self.EMG_VALID_MAX]
        t_vals = [v[1] for v in recent if self.EMG_VALID_MIN < v[1] < self.EMG_VALID_MAX]
        if not b_vals or not t_vals:
            return None, None
        b_arr = np.array(b_vals)
        t_arr = np.array(t_vals)
        # 排除 ±3σ 离群值
        b_mean, b_std = float(b_arr.mean()), float(b_arr.std())
        t_mean, t_std = float(t_arr.mean()), float(t_arr.std())
        b_filt = b_arr[abs(b_arr - b_mean) < 3 * b_std] if b_std > 0 else b_arr
        t_filt = t_arr[abs(t_arr - t_mean) < 3 * t_std] if t_std > 0 else t_arr
        return (float(b_filt.mean()) if len(b_filt) > 0 else float(b_arr.mean()),
                float(t_filt.mean()) if len(t_filt) > 0 else float(t_arr.mean()))

    def calib_tick(self):
        with self._lock:
            phase = self.calib_phase
            elapsed = time.time() - self.calib_phase_start

        if phase == 'idle':
            return 'idle', None, False

        elif phase == 'relax':
            if elapsed > 3.0:
                with self._lock:
                    b_avg, t_avg = self._calib_window_mean()
                    if b_avg is not None and self.EMG_VALID_MIN < b_avg < self.EMG_VALID_MAX:
                        self.calib_rest_b = b_avg
                        self.calib_rest_t = t_avg
                    elif self.emg_biceps is not None:
                        self.calib_rest_b = self.emg_biceps
                        self.calib_rest_t = self.emg_triceps or 0
                    self.calib_phase = 'bend'
                    self.calib_phase_start = time.time()
                return 'relax', None, False
            return 'relax', '请完全放松手臂，自然下垂，保持静止。', False

        elif phase == 'bend':
            if elapsed > 5.0:
                with self._lock:
                    angle = self.left_angle
                    b_avg, t_avg = self._calib_window_mean()
                    if angle is not None and angle > 45:
                        if b_avg is not None and self.EMG_VALID_MIN < b_avg < self.EMG_VALID_MAX:
                            self.calib_v90_b = b_avg
                            self.calib_v90_t = t_avg
                        elif self.emg_biceps is not None:
                            self.calib_v90_b = self.emg_biceps
                            self.calib_v90_t = self.emg_triceps
                    self.calib_phase = 'done'
                    self.calib_phase_start = time.time()
                return 'bend', None, False
            return 'bend', '请缓慢弯曲手肘至九十度，并保持住。', False

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


# ============================== ROS2 订阅节点 ==============================

class VoiceStateNode:
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
        self._node.create_subscription(String, '/raw_emg', self._on_raw_emg, 10)
        self._node.get_logger().info('语音状态监听已就绪 (v7)')

    @property
    def node(self):
        return self._node

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

    def _on_raw_emg(self, msg):
        """从 UDP EMG 接收器获取 BPM 心率数据"""
        try:
            data = json.loads(msg.data)
            bpm = data.get('bpm', 0)
            if bpm and bpm > 0:
                self._state.update_bpm(int(bpm))
        except Exception:
            pass


def ros2_spin_thread(state: SystemState):
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

def safe_arecord(wav_path, duration, device=MIC_DEV):
    """带超时保护的录音，防止 USB 设备卡死"""
    try:
        subprocess.run(
            ['timeout', str(int(duration) + 5), 'arecord',
             '-q', '-D', device, '-d', str(int(duration)),
             '-f', 'S16_LE', '-r', str(RATE), '-c', '1', wav_path],
            timeout=int(duration) + 10,
            stderr=subprocess.DEVNULL)  # 静默 EINTR 等无害错误
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def record_wav(wav_path, duration=RECORD_SEC):
    return safe_arecord(wav_path, duration, MIC_DEV)

def preprocess_audio(wav_path):
    with wave.open(wav_path, 'rb') as w:
        params = w.getparams()
        frames = w.readframes(params.nframes)
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    peak_raw = float(abs(data).max())
    data_clean = data - data.mean()
    zcr = float(np.sum(np.abs(np.diff(np.sign(data_clean))) > 0) / len(data_clean))
    data_clean = data_clean / (abs(data_clean).max() + 1e-8)
    out_path = wav_path.replace('.wav', '_norm.wav')
    data_out = (data_clean * 32767).astype(np.int16)
    with wave.open(out_path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(data_out.tobytes())
    return out_path, peak_raw, zcr

def record_vad(wav_path, max_duration=RECORD_SEC, timeout=30):
    CHUNK = int(RATE * VAD_CHUNK_SEC)
    started = False
    silent_chunks = 0
    silence_needed = int(VAD_SILENCE_SEC / VAD_CHUNK_SEC)
    all_frames = []
    total_chunks = 0
    max_chunks = int(max_duration / VAD_CHUNK_SEC)
    timeout_chunks = int(timeout / VAD_CHUNK_SEC)

    safe_arecord('/dev/null', 1, MIC_DEV)

    consec_failures = 0  # 连续失败计数器, 防止麦克风故障时死循环
    while total_chunks < timeout_chunks:
        chunk_file = f'/tmp/vad_chunk_{total_chunks}.wav'
        ok = safe_arecord(chunk_file, VAD_CHUNK_SEC, MIC_DEV)  # 录 VAD_CHUNK_SEC 秒, 不是硬编码1秒
        if not ok:
            consec_failures += 1
            total_chunks += 1  # 失败也递增, 避免死循环
            if consec_failures >= 5:
                print(' ⚠️ 麦克风连续录音失败, 跳过本轮监听')
                return False, 0, 0
            time.sleep(0.5)
            continue
        consec_failures = 0  # 成功则重置

        if not os.path.exists(chunk_file):
            total_chunks += 1
            break
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

    combined = np.concatenate(all_frames)
    peak_raw = float(abs(combined).max())
    combined_clean = combined - combined.mean()
    zcr = float(np.sum(np.abs(np.diff(np.sign(combined_clean))) > 0) / len(combined_clean))

    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(combined.astype(np.int16).tobytes())

    return True, peak_raw, zcr


# ============================== TrainingMonitor ==============================

class TrainingMonitor:
    """后台线程: 每5s检查告警/训练进度/分步提示"""

    def __init__(self, state: SystemState, tts: PriorityTTSSpeaker):
        self.state = state
        self.tts = tts
        self.running = True
        self._last_progress_announce = 0.0
        self._last_step_prompt = 0.0
        self._step_prompts_spoken = set()   # 已说过的分步提示
        self._alert_seq = 0

    def run(self):
        while self.running:
            try:
                self._check_alerts()
                self._check_heart_rate()
                self._check_training_progress()
                self._check_step_prompts()
            except Exception as e:
                print(f'  [Monitor] 异常: {e}')
            time.sleep(5)

    def reset_step_prompts(self):
        """训练重新开始时清除已播提示, 确保下一轮训练的 idle/completed 提示能再次触发"""
        self._step_prompts_spoken.clear()
        self._last_step_prompt = 0.0

    def stop(self):
        self.running = False

    # ---- 告警检查 ----
    def _check_alerts(self):
        alerts, age = self.state.get_alerts()
        if not alerts or age > 30:
            return

        for alert in alerts:
            self._handle_alert(alert)

    # ---- 心率预警 ----
    HR_LOW_THRESHOLD = 40
    HR_HIGH_THRESHOLD = 180

    def _check_heart_rate(self):
        """检查心率是否异常：<40 过低，>180 过高"""
        bpm, age = self.state.get_bpm()
        if bpm is None or age > 15:  # 数据超过15秒则跳过
            return

        if bpm < self.HR_LOW_THRESHOLD:
            sub = 'heart_rate_low'
            if self.state.should_speak_alert(sub):
                msg = '警告：心率过低！当前心率低于40次每分钟，请注意身体状况。'
                self.state.record_alert_spoken(sub)
                self.tts.speak_now(msg)  # 生命攸关，立即打断播报
        elif bpm > self.HR_HIGH_THRESHOLD:
            sub = 'heart_rate_high'
            if self.state.should_speak_alert(sub):
                msg = '警告：心率过高！当前心率超过180次每分钟，请立即停止运动并休息。'
                self.state.record_alert_spoken(sub)
                self.tts.speak_now(msg)  # 生命攸关，立即打断播报

    def _handle_alert(self, alert):
        """分类处理告警，带冷却"""
        alert_type = alert.get('type', '')
        alert_id = f"{alert_type}_{alert.get('level', '')}_{self._alert_seq}"
        self._alert_seq += 1

        if self.state.has_alert_been_announced(alert_id):
            return

        if alert_type == 'compensation':
            level = alert.get('level', 'moderate')
            sub = f'compensation_{level}'
            if level == 'severe':
                msg = '严重警告：检测到严重肌肉代偿。请注意用二头肌发力，避免耸肩或借助其他肌群。'
                pri = 2
            elif level == 'moderate':
                msg = '提醒：检测到中度肌肉代偿。请专注于二头肌发力，保持动作标准。'
                pri = 2
            else:
                msg = '提示：检测到轻度肌肉代偿，请保持正确的发力模式。'
                pri = 1

        elif alert_type == 'electrode':
            if alert.get('dropout_b') or alert.get('dropout_t'):
                sub = 'electrode_dropout'
                msg = '警告：电极信号脱落！请检查二头肌和三头肌电极是否贴合皮肤。'
                pri = 3
            else:
                sub = 'electrode_impedance'
                msg = '提示：电极接触不良，请调整电极位置确保贴合。'
                pri = 2

        elif alert_type == 'form':
            sub = 'form_wrong'
            msg = '注意：动作不标准。请缓慢匀速屈伸手肘，保持稳定。'
            pri = 2
        else:
            return

        # 冷却检查
        if not self.state.should_speak_alert(sub):
            return

        # 播报
        self.state.record_alert_spoken(sub)
        self.state.mark_alert_announced(alert_id)
        self.state.increment_alert_counter(alert_type)

        if pri >= 3:
            self.tts.speak_now(msg)
        else:
            self.tts.speak(msg, priority=pri)

    # ---- 训练进度播报 ----
    def _check_training_progress(self):
        live = self.state.get_training_live()
        if not live or live.get('paused'):
            return

        effective = live.get('effective_duration', 0)
        reps = live.get('reps', 0)

        # 每 PROGRESS_INTERVAL_SEC 播报一次进度
        if effective - self._last_progress_announce >= PROGRESS_INTERVAL_SEC:
            mins = int(effective // 60)
            angle = live.get('angle', 0) or 0
            msg = (f'训练已进行{mins}分钟，完成{reps}次屈伸。'
                   f'当前角度{angle:.0f}度。继续加油！')
            self.tts.speak(msg, priority=0)
            self._last_progress_announce = effective

        # 次数里程碑
        for mile in REP_MILESTONES:
            with self.state._lock:
                announced = self.state.last_rep_announced
            if reps >= mile and announced < mile:
                msg = f'已完成{mile}次，做得很好！'
                self.tts.speak(msg, priority=1)
                with self.state._lock:
                    self.state.last_rep_announced = mile
                break

    # ---- 分步提示 ----
    def _check_step_prompts(self):
        """根据系统状态提示可用的语音指令"""
        now = time.time()
        if now - self._last_step_prompt < 60:
            return  # 最少60s间隔

        phase = self.state.training_phase

        if phase == 'idle':
            if 'idle' not in self._step_prompts_spoken:
                self.tts.speak(
                    '语音助手就绪。训练前请先说"开始校准"，再"开始训练"。'
                    '当前训练模式为肘关节屈伸。说"帮助"查看所有指令。',
                    priority=1
                )
                self._step_prompts_spoken.add('idle')
                self._last_step_prompt = now

        elif phase == 'paused':
            # 暂停超过30s才提示
            live = self.state.get_training_live()
            if live:
                dur = live.get('duration', 0)
                eff = live.get('effective_duration', 0)
                paused_for = dur - eff
                if paused_for > 30:
                    msg = '训练仍在暂停中。休息好了说"继续训练"恢复，或"结束训练"查看报告。'
                    self.tts.speak(msg, priority=1)
                    self._last_step_prompt = now

        elif phase == 'completed':
            if 'completed' not in self._step_prompts_spoken:
                msg = ('肘关节屈伸训练已完成。你可以说"训练报告"查看数据，'
                       '说"A I报告"获取康复分析，或"开始训练"进行下一轮。')
                self.tts.speak(msg, priority=0)
                self._step_prompts_spoken.add('completed')
                self._last_step_prompt = now


# ============================== VoiceAssistant ==============================

class VoiceAssistant:
    """语音助手 v7: 关键词优先 → LLM 回退 → 基础回显"""

    def __init__(self, state, tts, use_vad=False, enable_llm=True):
        self.state = state
        self.tts = tts
        self.use_vad = use_vad
        self.running = True
        self.monitor = None  # 由 run_xxx 设置

        # 校准
        self._calib_active = False
        self._calib_last_prompt = None

        # 无人声超时提醒
        self._last_interaction_time = time.time()
        self._inactivity_level = 0  # 已触发的最高提醒级别 (0=未触发)

        # LLM
        self.llm_client = None
        self._llm_unavailable_spoken = False
        if enable_llm and _HAS_LLM:
            self.llm_client = DeepSeekClient()
            if self.llm_client.is_available():
                print(f'  [LLM] DeepSeek API 已配置 (模型: {self.llm_client.model})')
            else:
                print('  [LLM] 未设置 DEEPSEEK_API_KEY, AI对话功能不可用')
                self.llm_client = None

    # ---- 意图匹配 (关键词优先) ----
    def match_command(self, text):
        text = text.strip().lower().replace(' ', '')
        patterns = [
            (['再见', '退出', '拜拜'], 'goodbye'),
            (['谢谢', '多谢', '感谢'], 'thanks'),
            (['帮助', '你能做什么', '有什么功能', '指令', '命令'], 'help'),
            (['结束训练', '停止训练', '停止记录', '结束记录'], 'stop_training'),
            (['暂停训练', '暂停', '休息一下', '先停一下'], 'pause_training'),
            (['继续训练', '恢复训练', '继续', '取消暂停'], 'resume_training'),
            (['开始训练', '启动训练', '开始记录', '启动记录', '训练开始'], 'start_training'),
            (['训练报告', '训练统计', '训练结果', '训练总结'], 'training_report'),
            (['AI报告', '智能报告', '生成报告', '康复报告', 'AI诊断', '人工智能报告'], 'ai_report'),
            (['开始校准', '校准', '重新校准', '自动校准'], 'start_calib'),
            (['肌肉状态', '肌肉数据', '肌电数据', '肌电'], 'muscle_status'),
            (['当前角度', '角度多少', '手臂角度', '关节角度', '现在多少度'], 'current_angle'),
            (['诊断', '健康检查', '系统检测', '自检'], 'diagnosis'),
            (['检查状态', '系统状态', '运行状态', '状态'], 'status'),
            (['你好', '嗨', '哈喽', 'hello', 'hi'], 'greeting'),
            (['最大音量', '音量最大', '声音最大'], 'volume_max'),
            (['最小音量', '音量最小', '声音最小'], 'volume_min'),
            (['大声一点', '音量加大', '增大音量', '声音大一点', '调大音量', '音量调大'], 'volume_up'),
            (['小声一点', '音量减小', '减小音量', '声音小一点', '调小音量', '音量调小', '降低音量'], 'volume_down'),
        ]

        for keywords, cmd_id in patterns:
            for kw in keywords:
                if kw in text:
                    return cmd_id
        return None

    # ---- 统一入口: 关键词 → LLM → 回退 ----
    def process_utterance(self, text):
        """处理用户语音: 先关键词匹配, 失败则走 LLM, 再失败回显"""
        # Phase 0: 音量调到具体数值 (如"音量调到50"、"音量百分之80")
        vol_pct = self._extract_volume_number(text)
        if vol_pct is not None:
            print(f'  >>> 执行: volume_set {vol_pct}%')
            response = self._cmd_volume_set(vol_pct)
            if response:
                self._reset_inactivity()
            return response

        # Phase 1: 关键词匹配
        cmd_id = self.match_command(text)
        if cmd_id:
            print(f'  >>> 执行: {cmd_id}')
            response = self.handle(cmd_id)
            if response:
                self._reset_inactivity()
            return response

        # Phase 2: LLM 自然对话
        if self.llm_client and self.llm_client.is_available():
            print(f'  [LLM] 发送到 DeepSeek...')
            reply = self.llm_client.natural_chat(text)
            if reply:
                print(f'  [LLM] 回复: {reply[:60]}...')
                self._reset_inactivity()
                return reply

        # Phase 3: 回退
        if len(text) > 3:
            self._reset_inactivity()
            if not self._llm_unavailable_spoken:
                self._llm_unavailable_spoken = True
                return f'你说的是：{text}。AI对话功能暂未配置，请说"帮助"查看可用指令。'
            return f'你说的是：{text}。请说"帮助"查看可用指令。'
        return None

    def _extract_volume_number(self, text):
        """从文本提取音量百分比, 如'音量调到50'→50, '音量百分之80'→80。无匹配返回None"""
        import re
        # "音量调到百分之50" / "音量调到50" / "音量调至百分之50" / "音量百分之50"
        m = re.search(r'音量.*?(?:百分之)?(\d{1,3})', text)
        if m:
            pct = int(m.group(1))
            if 0 <= pct <= 100:
                return pct
        return None

    def _reset_inactivity(self):
        """用户有交互，重置无人声超时计时器"""
        self._last_interaction_time = time.time()
        self._inactivity_level = 0

    def _check_inactivity(self):
        """检查无人声超时，逐级播报提醒"""
        elapsed = time.time() - self._last_interaction_time
        for i, (threshold, text) in enumerate(INACTIVITY_LEVELS):
            if elapsed >= threshold and self._inactivity_level <= i:
                self._inactivity_level = i + 1  # 标记已触发，避免重复播报
                print(f'  ⏰ 无人声 {elapsed:.0f}s → 播报提醒 (Lv{i+1})')
                self.tts.speak(text, priority=0)
                break

    # ---- 命令处理 ----
    def handle(self, cmd_id):
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
        elif cmd_id == 'pause_training':
            return self._cmd_pause_training()
        elif cmd_id == 'resume_training':
            return self._cmd_resume_training()
        elif cmd_id == 'training_report':
            return self._cmd_training_report()
        elif cmd_id == 'ai_report':
            return self._cmd_ai_report()
        elif cmd_id == 'start_calib':
            return self._cmd_start_calib()
        elif cmd_id == 'help':
            return self._cmd_help()
        elif cmd_id == 'goodbye':
            self.running = False
            if self.llm_client:
                self.llm_client.reset_history()
            return '再见，祝您康复顺利！'
        elif cmd_id == 'thanks':
            return '不客气，这是我应该做的。祝您康复顺利！'
        elif cmd_id == 'volume_up':
            return self._cmd_volume_up()
        elif cmd_id == 'volume_down':
            return self._cmd_volume_down()
        elif cmd_id == 'volume_max':
            return self._cmd_volume_max()
        elif cmd_id == 'volume_min':
            return self._cmd_volume_min()
        return None

    # ---- 音量控制辅助 ----
    def _get_volume_pct(self):
        """读取 SPKL 当前音量, 返回百分比 0-100"""
        try:
            out = subprocess.check_output(
                ['amixer', '-c', str(VOLUME_CARD), 'sget', 'SPKL'],
                stderr=subprocess.DEVNULL).decode()
            m = __import__('re').search(r'Mono: Playback (\d+)', out)
            if m:
                raw = int(m.group(1))
                return round(raw / VOLUME_MAX_RAW * 100)
        except Exception:
            pass
        return 70  # 默认值

    def _set_volume_raw(self, raw_value):
        """设置 SPKL/SPKR 原始音量 (0-191)"""
        v = max(0, min(VOLUME_MAX_RAW, int(raw_value)))
        subprocess.run(['amixer', '-c', str(VOLUME_CARD), 'sset', 'SPKL', str(v)],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(['amixer', '-c', str(VOLUME_CARD), 'sset', 'SPKR', str(v)],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def _set_volume_pct(self, pct):
        """设置音量为指定百分比 (0-100)"""
        v = max(0, min(100, int(pct)))
        raw = round(v / 100 * VOLUME_MAX_RAW)
        self._set_volume_raw(raw)
        return v

    def _cmd_volume_up(self):
        cur = self._get_volume_pct()
        new_pct = min(100, cur + VOLUME_STEP_PCT)
        self._set_volume_pct(new_pct)
        return f'音量已调大，当前约百分之{new_pct}。'

    def _cmd_volume_down(self):
        cur = self._get_volume_pct()
        new_pct = max(0, cur - VOLUME_STEP_PCT)
        self._set_volume_pct(new_pct)
        return f'音量已调小，当前约百分之{new_pct}。'

    def _cmd_volume_max(self):
        self._set_volume_pct(100)
        return '音量已调至最大。'

    def _cmd_volume_min(self):
        self._set_volume_pct(5)  # 最小但非静音
        return '音量已调至最小。'

    def _cmd_volume_set(self, pct):
        v = self._set_volume_pct(pct)
        return f'音量已设为百分之{v}。'

    # ---- 具体命令 ----
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
            dur = live.get('effective_duration', live.get('duration', 0))
            parts.append(f'训练已进行{dur:.0f}秒，完成{live["reps"]}次。')
            if live.get('paused'):
                parts.append('当前已暂停。')
        parts.append('请说"帮助"查看可用指令。')
        return ' '.join(parts)

    def _cmd_status(self):
        angle, valid = self.state.get_angle()
        emg_b, emg_t, emg_age = self.state.get_emg()
        val_rpt, val_age = self.state.get_validation()
        parts = []

        if valid and angle is not None:
            parts.append(f'肘关节角度{angle:.0f}度。')
        else:
            parts.append('角度传感器暂未连接。')

        if emg_b is not None and emg_age < 5:
            parts.append(f'二头肌{emg_b:.0f}微伏，三头肌{emg_t:.0f}微伏。')
        else:
            parts.append('肌电数据暂未更新。')

        if val_rpt and val_age < 30:
            overall = val_rpt.get('overall', {})
            b = overall.get('biceps', {}) or {}
            t = overall.get('triceps', {}) or {}
            if b.get('r2') is not None:
                parts.append(f'模型准确度：二头肌R方{b["r2"]:.2f}。')

        live = self.state.get_training_live()
        if live:
            dur = live.get('effective_duration', live.get('duration', 0))
            parts.append(f'训练进行中，已有效训练{dur:.0f}秒，{live["reps"]}次。')
            if live.get('paused'):
                parts.append('当前已暂停。')

        alerts, alert_age = self.state.get_alerts()
        if alerts and alert_age < 30:
            for a in alerts[:2]:
                if a['type'] == 'compensation':
                    parts.append(f'检测到{a.get("level","")}肌肉代偿。')
                elif a['type'] == 'electrode':
                    parts.append('电极信号异常，请检查。')

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
        return '角度传感器暂未连接，请确认摄像头或骨架追踪已启动。'

    def _cmd_muscle(self):
        emg_b, emg_t, age = self.state.get_emg()
        if emg_b is None or age > 5:
            return '肌电数据暂未更新。'
        parts = [f'二头肌{emg_b:.0f}微伏，三头肌{emg_t:.0f}微伏。']
        if emg_b < 100:
            parts.append('二头肌处于放松状态。')
        elif emg_b < 400:
            parts.append('二头肌轻度激活。')
        elif emg_b < 800:
            parts.append('二头肌中度收缩。')
        else:
            parts.append('二头肌强力收缩。')
        return ' '.join(parts)

    def _cmd_diagnosis(self):
        val_rpt, val_age = self.state.get_validation()
        if not val_rpt or val_age > 60:
            return '暂无交叉验证报告，请至少采集十组样本。'
        diag = val_rpt.get('diagnosis', [])
        if not diag:
            return '交叉验证数据不足。'
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
        comp = val_rpt.get('compensation', {})
        if comp:
            cs = comp.get('compensation_score', 0)
            cl = comp.get('compensation_level', 'unknown')
            if cl in ('severe', 'moderate'):
                parts.append(f'肌肉代偿评分{cs:.0f}分，{cl}，请注意发力模式。')
        elec = val_rpt.get('electrode_health', {})
        if elec:
            qb = elec.get('biceps_quality', 100)
            qt = elec.get('triceps_quality', 100)
            if qb < 50 or qt < 50:
                parts.append(f'电极接触质量偏低，请检查贴合。')
        return ' '.join(parts)

    def _cmd_start_training(self):
        if self.state.is_training and not self.state.training_paused:
            return '训练已经在进行中。你可以说"暂停训练"休息，或"结束训练"停止。'
        if self.state.is_training and self.state.training_paused:
            return '训练当前已暂停。说"继续训练"恢复即可。'
        if self._calib_active:
            return '校准正在进行中，请先完成校准再开始训练。'

        # 检查是否已完成校准
        calib_path = os.path.expanduser('~/calibration_save.json')
        if not os.path.exists(calib_path):
            return '请先进行校准再开始训练。说"开始校准"，跟随语音提示完成校准流程。'
        try:
            with open(calib_path) as f:
                saved = json.load(f)
            source = saved.get('source', '')
            if 'voice_calibration' not in source and 'auto_calibrate' not in source:
                return '请先进行语音校准再开始训练。说"开始校准"，跟随提示完全放松再弯曲至九十度。'
        except Exception:
            return '校准数据读取失败，请重新"开始校准"。'

        self.state.start_training()
        # 重置分步提示标记，确保新训练轮次的 idle/completed 提示能再次触发
        if self.monitor:
            self.monitor.reset_step_prompts()
        return ('肘关节屈伸训练已开始。'
                '请从手臂伸直开始，缓慢弯曲至最大角度，再缓慢伸直回原位。'
                '系统会自动记录每次完整屈伸。说"暂停训练"休息，"结束训练"停止。')

    def _cmd_pause_training(self):
        if not self.state.is_training:
            return '当前没有进行中的训练。说"开始训练"启动。'
        if self.state.training_paused:
            return '训练已经暂停。说"继续训练"恢复。'
        ok = self.state.pause_training()
        if ok:
            return '训练已暂停。休息好了说"继续训练"恢复。'
        return '暂停失败，请重试。'

    def _cmd_resume_training(self):
        if not self.state.is_training:
            return '当前没有进行中的训练。说"开始训练"启动。'
        if not self.state.training_paused:
            return '训练正在运行中。'
        ok = self.state.resume_training()
        if ok:
            return f'训练已恢复。继续加油！'
        return '恢复失败，请重试。'

    def _cmd_stop_training(self):
        if not self.state.is_training:
            return '当前没有进行中的训练。说"开始训练"启动。'

        report = self.state.stop_training()
        eff_sec = int(report['effective_duration_sec'])
        mins = eff_sec // 60
        secs = eff_sec % 60

        parts = [
            '训练已停止。',
            f'有效训练时长{mins}分{secs}秒，',
            f'完成{report["rep_count"]}次屈伸，',
            f'最大角度{report["max_angle"]:.0f}度，',
            f'平均角度{report["avg_angle"]:.0f}度，',
            f'平均二头肌肌电{report["avg_emg_biceps"]:.0f}微伏。',
        ]
        if report.get('pause_count', 0) > 0:
            parts.append(f'共暂停{report["pause_count"]}次。')

        parts.append('你可以说"A I报告"获取康复评估，或"开始训练"进行下一轮。')
        return ' '.join(parts)

    def _cmd_training_report(self):
        # 实时报告：训练进行中时输出当前进度，不停止训练
        live = self.state.get_training_live()
        if live:
            eff = live.get('effective_duration', live.get('duration', 0))
            mins = int(eff // 60)
            secs = int(eff % 60)
            angle = live.get('angle', 0) or 0
            emg_b, emg_t, emg_age = self.state.get_emg()
            parts = [
                '当前训练进度：',
                f'已有效训练{mins}分{secs}秒，',
                f'完成{live["reps"]}次屈伸，',
                f'当前角度{angle:.0f}度。',
            ]
            if live.get('paused'):
                parts.append('训练已暂停。')
            if emg_b is not None and emg_age < 5:
                parts.append(f'当前二头肌{emg_b:.0f}微伏，三头肌{emg_t:.0f}微伏。')
            parts.append('说"结束训练"停止并查看完整报告。')
            return ' '.join(parts)

        report = self.state.get_last_completed_training()
        if not report:
            return '没有训练数据。请先说"开始训练"启动训练。'

        eff_sec = int(report['effective_duration_sec'])
        mins = eff_sec // 60
        secs = eff_sec % 60
        parts = [
            f'训练报告：有效训练时长{mins}分{secs}秒，',
            f'完成{report["rep_count"]}次屈伸，',
            f'角度范围{report["min_angle"]:.0f}至{report["max_angle"]:.0f}度，',
            f'平均{report["avg_angle"]:.0f}度，',
            f'二头肌平均{report["avg_emg_biceps"]:.0f}微伏，',
            f'三头肌平均{report["avg_emg_triceps"]:.0f}微伏。',
        ]
        return ' '.join(parts)

    def _cmd_ai_report(self):
        """LLM 训练总结，训练进行中返回实时进度，失败则回退到基础报告"""
        # 训练进行中 → 返回实时进度, 不停止训练 (与 _cmd_training_report 行为一致)
        live = self.state.get_training_live()
        if live:
            eff = live.get('effective_duration', live.get('duration', 0))
            mins = int(eff // 60)
            secs = int(eff % 60)
            angle = live.get('angle', 0) or 0
            return (f'训练进行中：已有效训练{mins}分{secs}秒，'
                    f'完成{live["reps"]}次屈伸，当前角度{angle:.0f}度。'
                    f'请先"结束训练"再查看AI报告。')

        report = self.state.get_last_completed_training()
        if not report:
            return '没有训练记录。请先完成一次训练。'

        val_rpt, val_age = self.state.get_validation()

        if self.llm_client and self.llm_client.is_available():
            print('  [LLM] 生成训练总结...')
            summary = self.llm_client.generate_training_summary(report, val_rpt)
            if summary:
                # 前置基础数据 + LLM 评估
                eff_sec = int(report['effective_duration_sec'])
                base = (f'训练总结：有效时长{eff_sec // 60}分{eff_sec % 60}秒，'
                        f'完成{report["rep_count"]}次。')
                return base + summary
            else:
                return 'AI评估暂时不可用，以下为基础数据：' + self._cmd_training_report()

        return 'AI报告功能未配置。你可以说"训练报告"查看基础数据。'

    def _cmd_start_calib(self):
        if self._calib_active:
            return '校准已在执行中，请跟随语音提示完成动作。'
        if self.state.is_training:
            return '训练正在进行中，请先"结束训练"再进行校准。'
        self.state.start_calibration()
        self._calib_active = True
        self._calib_last_prompt = None
        return None  # 由 _calib_loop 驱动

    def _calib_loop(self):
        if not self._calib_active:
            return
        phase, prompt, is_done = self.state.calib_tick()

        if is_done:
            result = self.state.get_calib_result()
            self._calib_active = False
            self.state.finish_calibration()  # 线程安全复位

            # 保存到 calibration_save.json，桥接语音校准和TCN推理管线
            calib_path = os.path.expanduser('~/calibration_save.json')
            try:
                # 尝试加载已有校准（保留非EMG字段）
                calib_vec = None
                if os.path.exists(calib_path):
                    try:
                        with open(calib_path) as f:
                            saved = json.load(f)
                        calib_vec = saved.get('calib_vec')
                    except (json.JSONDecodeError, Exception):
                        print('  [校准] 校准文件损坏, 使用默认值重建')
                if not calib_vec or len(calib_vec) != 16:
                    calib_vec = [200, 500, 500, 5000, 10, 0.1,
                                 80, 120, 120, 2000, 5, 0.1,
                                 170, 70, 22.0, 0]

                # 更新语音校准采集的4个值
                if result['rest_b'] is not None:
                    calib_vec[0] = round(float(result['rest_b']), 1)   # b_rest
                if result['v90_b'] is not None:
                    calib_vec[1] = round(float(result['v90_b']), 1)     # b_90
                if result['rest_t'] is not None:
                    calib_vec[6] = round(float(result['rest_t']), 1)   # t_rest
                if result['v90_t'] is not None:
                    calib_vec[7] = round(float(result['v90_t']), 1)     # t_90

                data = {
                    'calib_vec': [float(v) for v in calib_vec],
                    'b_rest': float(calib_vec[0]),
                    'b_90': float(calib_vec[1]),
                    't_rest': float(calib_vec[6]),
                    't_90': float(calib_vec[7]),
                    'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'voice_calibration',
                }
                with open(calib_path, 'w') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f'  [校准] 已保存到 {calib_path}')
            except Exception as e:
                print(f'  [校准] 保存失败: {e}')

            # 播报结果
            r = result
            parts = ['校准完成。']
            if r['rest_b'] is not None and r['v90_b'] is not None:
                parts.append(f'二头肌：放松{r["rest_b"]:.0f}微伏，九十度{r["v90_b"]:.0f}微伏。')
            if r['rest_t'] is not None and r['v90_t'] is not None:
                parts.append(f'三头肌：放松{r["rest_t"]:.0f}微伏，九十度{r["v90_t"]:.0f}微伏。')
            if any(v is None for v in [r['rest_b'], r['v90_b'], r['rest_t'], r['v90_t']]):
                parts.append('部分校准值缺失，请确认EMG电极已贴合。')
            parts.append('校准值约30秒后自动生效，无需重启。你可以说"开始训练"启动康复训练。')
            self.tts.speak(' '.join(parts), priority=1)
            return

        if prompt and prompt != self._calib_last_prompt:
            self._calib_last_prompt = prompt
            self.tts.speak(prompt, priority=1)

    def _cmd_help(self):
        return (
            '当前训练模式为肘关节屈伸。'
            '使用流程：首先说"开始校准"，然后"开始训练"。'
            '训练中可"暂停训练"或"继续训练"，完成后"结束训练"。'
            '数据查询：检查状态、当前角度、肌肉状态。'
            '分析报告：训练报告、AI报告、诊断。'
            '音量控制：大声一点、小声一点、最大音量、最小音量、音量调到百分之五十。'
        )

    # ---- 主循环 ----
    def run_push_to_talk(self, model):
        print('=' * 55)
        print('  智能健康袖套 — 语音助手 v7 [按键模式]')
        print(f'  麦克风: {MIC_DEV}')
        print('  指令: 开始/暂停/继续/结束训练 | 状态 | 角度 | AI报告 | 帮助')
        print('  按 Enter 开始说话，Ctrl+C 退出')
        print('=' * 55)

        # 启动后台监控
        monitor = TrainingMonitor(self.state, self.tts)
        self.monitor = monitor
        monitor_thread = threading.Thread(target=monitor.run, daemon=True)
        monitor_thread.start()

        # 初始提示
        self.tts.speak(
            '语音助手已启动。你可以说"开始训练"启动康复训练，'
            '或"开始校准"进行系统校准。说"帮助"查看所有指令。',
            priority=0
        )

        while self.running:
            try:
                self._calib_loop()

                try:
                    import select
                    if select.select([sys.stdin], [], [], 0.5)[0]:
                        sys.stdin.readline()
                    else:
                        continue
                except Exception:
                    try:
                        input('\n[按 Enter 开始录音]')
                    except EOFError:
                        time.sleep(0.5)
                        continue

                for i in [3, 2, 1]:
                    print(f'{i}...', end='', flush=True)
                    time.sleep(0.7)
                print('说！')

                wav = '/tmp/voice_cmd.wav'
                if not record_wav(wav, RECORD_SEC):
                    print('⚠️ 录音失败, 请检查麦克风连接')
                    self.tts.speak('录音失败，请检查麦克风是否正常连接。', priority=1)
                    continue
                wav_norm, peak, zcr = preprocess_audio(wav)
                print(f'  峰值={peak:.0f}  过零率={zcr:.3f}  ', end='')

                if zcr < 0.02:
                    print('⚠️ 信号异常')
                    self.tts.speak('未检测到语音信号，请检查麦克风是否连接。', priority=1)
                    self._cleanup_wavs(wav, wav_norm)
                    continue

                print('识别中...')
                t0 = time.time()
                segments, info = model.transcribe(wav_norm, language='zh',
                                                  beam_size=1, vad_filter=True)
                text = ''.join(seg.text for seg in segments)
                elapsed = time.time() - t0

                if text.strip():
                    text = text.strip()
                    print(f'  [{elapsed:.1f}s] 识别: "{text}"')
                    response = self.process_utterance(text)
                    if response:
                        self.tts.speak(response)
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

        monitor.stop()
        if self.llm_client:
            self.llm_client.reset_history()

    def run_vad(self, model):
        print('=' * 55)
        print('  智能健康袖套 — 语音助手 v7 [VAD模式]')
        print(f'  麦克风: {MIC_DEV}')
        print('  语音活动检测中... 直接说话即可')
        print('  Ctrl+C 退出')
        print('=' * 55)

        # 启动后台监控
        monitor = TrainingMonitor(self.state, self.tts)
        self.monitor = monitor
        monitor_thread = threading.Thread(target=monitor.run, daemon=True)
        monitor_thread.start()

        # 初始化无人声计时器
        self._last_interaction_time = time.time()
        self._inactivity_level = 0
        _was_tts_busy = False

        while self.running:
            try:
                self._calib_loop()

                # TTS 播放中或刚结束 → 跳过，避免音箱→麦克风回授
                tts_busy = self.tts.is_speaking or self.tts.tts_cooldown_active
                if tts_busy:
                    _was_tts_busy = True
                    time.sleep(0.3)
                    continue

                # TTS 刚结束 → 重置无人声计时器 (系统播报也算"交互")
                if _was_tts_busy:
                    _was_tts_busy = False
                    self._last_interaction_time = time.time()
                    self._inactivity_level = 0

                print('  🎧 监听中...', end='', flush=True)
                wav = '/tmp/voice_cmd_vad.wav'
                detected, peak, zcr = record_vad(wav, timeout=12)  # 最多录12s, 防止噪音环境无限录

                # TTS 在录音期间启动了 → 丢弃
                if self.tts.is_speaking:
                    if detected:
                        print(' [TTS打断, 丢弃]')
                    self._cleanup_wavs(wav)
                    time.sleep(0.5)
                    continue

                if not detected:
                    print(' (静默)')
                    self._check_inactivity()
                    time.sleep(0.3)  # 避免空闲时 CPU 空转
                    continue

                wav_norm, peak2, zcr2 = preprocess_audio(wav)
                if zcr2 < 0.02:
                    print(f'  ⚠️ 信号异常 ZCR={zcr2:.3f}')
                    self._cleanup_wavs(wav, wav_norm)
                    continue

                print(f'  识别中...', end='', flush=True)
                t0 = time.time()
                segments, info = model.transcribe(wav_norm, language='zh',
                                                  beam_size=1, vad_filter=True)
                text = ''.join(seg.text for seg in segments)
                elapsed = time.time() - t0

                if text.strip():
                    text = text.strip()
                    print(f' [{elapsed:.1f}s] "{text}"')
                    response = self.process_utterance(text)
                    if response:
                        self.tts.speak(response)
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

        monitor.stop()
        if self.llm_client:
            self.llm_client.reset_history()

    def run_text_mode(self):
        """降级模式: 文字输入交互 (Whisper 不可用时)"""
        print('=' * 55)
        print('  智能健康袖套 — 语音助手 [文字模式]')
        print('  语音识别不可用, 输入文字指令')
        print('  输入 help 查看指令, quit 退出')
        print('=' * 55)

        monitor = TrainingMonitor(self.state, self.tts)
        self.monitor = monitor
        monitor_thread = threading.Thread(target=monitor.run, daemon=True)
        monitor_thread.start()

        self.tts.speak('语音助手已启动，当前为文字输入模式。', priority=0)

        print('  📝 输入指令: ', end='', flush=True)
        while self.running:
            try:
                self._calib_loop()
                try:
                    text = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print('\n退出')
                    break

                if not text:
                    continue

                if text.lower() in ('quit', 'exit', 'q', '退出', '再见'):
                    self.handle('goodbye')
                    break

                print(f'  >>> "{text}"')
                response = self.process_utterance(text, tts=self.tts)
                if response:
                    print(f'  🔊 {response}')
                    self.tts.speak(response)
                print('  📝 输入指令: ', end='', flush=True)

            except KeyboardInterrupt:
                print('\n退出')
                break
            except Exception as e:
                print(f'  错误: {e}')
                import traceback
                traceback.print_exc()

        monitor.stop()
        if self.llm_client:
            self.llm_client.reset_history()

    def _cleanup_wavs(self, *paths):
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ============================== 主入口 ==============================

def main():
    parser = argparse.ArgumentParser(description='智能健康袖套语音助手 v7')
    parser.add_argument('--vad', action='store_true', help='VAD自动触发模式')
    parser.add_argument('--daemon', action='store_true', help='后台守护模式 (等同于 --vad)')
    parser.add_argument('--no-llm', action='store_true', help='禁用 AI 对话功能')
    args = parser.parse_args()

    use_vad = args.vad or args.daemon
    enable_llm = not args.no_llm

    # SIGTERM 处理 (systemd 友好)
    shutdown_flag = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: shutdown_flag.set())
    signal.signal(signal.SIGINT, lambda signum, frame: shutdown_flag.set())

    def check_shutdown():
        """后台线程: 检测到 SIGTERM/SIGINT 时退出"""
        shutdown_flag.wait()
        print('\n[语音] 收到退出信号, 正在关闭...')
        os._exit(0)

    threading.Thread(target=check_shutdown, daemon=True).start()

    # 1. 初始化共享状态
    state = SystemState()

    # 2. ROS2 后台线程
    print('[语音] 启动 ROS2 监听...')
    ros2_thread = threading.Thread(target=ros2_spin_thread, args=(state,), daemon=True)
    ros2_thread.start()
    time.sleep(1.5)

    # 3. Whisper
    print('[语音] 加载 Whisper base 模型...')
    model = None
    try:
        model = WhisperModel(MODEL_PATH, device='cpu', compute_type='int8',
                             download_root='/opt/whisper-models')
        print(f'[语音] Whisper 模型就绪 (路径: {MODEL_PATH})')
    except Exception as e:
        print(f'[语音] 警告: Whisper 加载失败: {e}')
        if use_vad:
            print('[语音] VAD 模式需要 Whisper, 降级为按键模式')
            use_vad = False
        print('[语音] 语音识别不可用, 将使用文字输入模式')

    # 4. TTS
    tts = PriorityTTSSpeaker(cooldown_sec=1.5)  # TTS后冷却1.5s, 防止音箱回声触发VAD
    if tts.engine is None:
        print('[语音] 警告: 无可用 TTS 引擎, 仅文字输出')
    elif tts.engine == 'edge':
        print('[语音] 预缓存常用语音...')
        tts.prewarm_cache()

    # 5. 语音助手
    assistant = VoiceAssistant(state, tts, use_vad=use_vad, enable_llm=enable_llm)

    if model is not None:
        if use_vad:
            assistant.run_vad(model)
        else:
            assistant.run_push_to_talk(model)
    else:
        # 降级: 纯文字交互模式 (无语音识别)
        assistant.run_text_mode()

    tts.stop()
    print('[语音] 助手已退出')


if __name__ == '__main__':
    main()
