#!/usr/bin/env python3
"""
screen_server.py — RDK X5 触屏服务：患者端 Web UI

端口: 8081
协议: HTTP + MJPEG 摄像头流 + WebSocket 实时数据 + DeepSeek AI 报告
"""

import asyncio
import json
import logging
import os
import threading
import time
import sys

try:
    from aiohttp import web
except ImportError:
    print("请先安装 aiohttp: pip3 install aiohttp")
    sys.exit(1)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[WARN] 未找到 ROS2, 将使用模拟数据模式")

# DeepSeek API (可选, 用于 AI 报告)
try:
    import aiohttp as _aiohttp
    DEEPSEEK_KEY = None
    for key_path in [os.path.expanduser('~/.deepseek_key'), '/root/.deepseek_key']:
        if os.path.exists(key_path):
            with open(key_path) as f:
                DEEPSEEK_KEY = f.read().strip()
            break
    if not DEEPSEEK_KEY:
        DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
    HAS_DEEPSEEK = bool(DEEPSEEK_KEY)
except Exception:
    HAS_DEEPSEEK = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('screen_server')


# ============================== 共享状态 ==============================

class SharedState:
    """ROS2 → WebSocket 桥接, 线程安全"""

    def __init__(self):
        self._lock = threading.Lock()

        # 角度 (右臂为主)
        self.elbow_angle = 90.0       # 肘关节角度
        self.upper_angle = 0.0        # 大臂角度
        self.valid = False

        # 预测 EMG (TCN)
        self.pred_biceps_uv = 0.0
        self.pred_triceps_uv = 0.0

        # 真实 EMG (ESP32 BLE/UDP)
        self.real_biceps_uv = 0.0
        self.real_triceps_uv = 0.0

        # 心率
        self.heart_rate = 0

        # 交叉验证
        self.r2_biceps = 0.0
        self.r2_triceps = 0.0
        self.compensation_score = 0.0
        self.compensation_level = ""
        self.biceps_deficiency = False
        self.triceps_overactivation = False
        self.cocontraction_index = 0.0

        # 电极健康
        self.biceps_quality = 100.0
        self.triceps_quality = 100.0
        self.dropout_b = False
        self.dropout_t = False

        # 告警
        self.quality_alerts = []

        # 关键点
        self.keypoints = []
        self.camera_width = 640
        self.camera_height = 480

        # 摄像头帧
        self._camera_frame = None
        self._camera_frame_time = 0.0

        # AI 报告
        self._report = ""
        self._report_time = 0.0

        self.msg_count = 0
        self.last_update = time.time()

    # ---- 更新方法 ----

    def update_angle(self, data: dict):
        with self._lock:
            self.valid = data.get('right_valid', False)

            # 只在有效时更新角度,避免追踪丢失时归零
            if self.valid:
                ea = data.get('right_elbow_angle')
                ua = data.get('right_upper_angle')
                if ea is not None:
                    self.elbow_angle = float(ea) if ea is not None else self.elbow_angle
                if ua is not None:
                    self.upper_angle = float(ua) if ua is not None else self.upper_angle

            # 关键点
            points = data.get('points', None)
            if points and isinstance(points, list) and len(points) >= 17:
                kps = []
                for p in points[:17]:
                    if p is None:
                        kps.append(None)
                    elif isinstance(p, dict):
                        kps.append({'x': float(p.get('x', 0)), 'y': float(p.get('y', 0)), 'c': float(p.get('c', 0))})
                    elif isinstance(p, (list, tuple)) and len(p) >= 3:
                        kps.append({'x': float(p[0]), 'y': float(p[1]), 'c': float(p[2])})
                    else:
                        kps.append(None)
                self.keypoints = kps

            self.msg_count += 1
            self.last_update = time.time()

    def update_emg(self, data: dict):
        """预测 EMG (TCN)"""
        with self._lock:
            right = data.get('right', {}) if isinstance(data, dict) else {}
            if not right:
                # 兼容旧格式
                self.pred_biceps_uv = data.get('right_biceps_uv') or data.get('left_biceps_uv') or 0.0
                self.pred_triceps_uv = data.get('right_triceps_uv') or data.get('left_triceps_uv') or 0.0
            else:
                self.pred_biceps_uv = right.get('biceps_uv') or 0.0
                self.pred_triceps_uv = right.get('triceps_uv') or 0.0

    def update_raw_emg(self, data: dict):
        """真实 EMG (ESP32 原始ADC → μV 换算 + EMA平滑) + 心率"""
        with self._lock:
            BICEPS_SCALE = 0.25
            TRICEPS_SCALE = 0.06
            ALPHA = 0.15  # 平滑系数, 越小越稳 (0.15 = 15%新值 + 85%旧值)
            raw_b = float(data.get('biceps_uv', 0) or 0)
            raw_t = float(data.get('triceps_uv', 0) or 0)
            new_b = round(raw_b * BICEPS_SCALE, 1)
            new_t = round(raw_t * TRICEPS_SCALE, 1)
            # EMA: self.real_xxx_uv 上次值, new_b 新原始值
            self.real_biceps_uv = round(ALPHA * new_b + (1 - ALPHA) * self.real_biceps_uv, 1)
            self.real_triceps_uv = round(ALPHA * new_t + (1 - ALPHA) * self.real_triceps_uv, 1)
            bpm = data.get('bpm', 0)
            if bpm and int(bpm) > 0:
                self.heart_rate = int(bpm)

    def update_validation(self, data: dict):
        with self._lock:
            self.r2_biceps = data.get('r2_biceps', self.r2_biceps)
            self.r2_triceps = data.get('r2_triceps', self.r2_triceps)
            self.compensation_score = data.get('compensation_score', self.compensation_score)
            self.compensation_level = data.get('compensation_level', self.compensation_level)
            self.biceps_deficiency = data.get('biceps_deficiency', self.biceps_deficiency)
            self.triceps_overactivation = data.get('triceps_overactivation', self.triceps_overactivation)
            self.cocontraction_index = data.get('cocontraction_index', self.cocontraction_index)
            self.biceps_quality = data.get('biceps_quality', self.biceps_quality)
            self.triceps_quality = data.get('triceps_quality', self.triceps_quality)
            self.dropout_b = data.get('dropout_b', self.dropout_b)
            self.dropout_t = data.get('dropout_t', self.dropout_t)

    def update_alerts(self, data: list):
        with self._lock:
            self.quality_alerts = data if isinstance(data, list) else []

    def update_camera_frame(self, jpeg_bytes: bytes, width: int = 640, height: int = 480):
        with self._lock:
            self._camera_frame = jpeg_bytes
            self._camera_frame_time = time.time()
            if width: self.camera_width = width
            if height: self.camera_height = height

    def get_frame(self):
        with self._lock:
            return self._camera_frame, self._camera_frame_time

    def set_report(self, text: str):
        with self._lock:
            self._report = text
            self._report_time = time.time()

    @staticmethod
    def _safe_round(val, ndigits=1):
        if val is None: return 0.0
        try: return round(float(val), ndigits)
        except (TypeError, ValueError): return 0.0

    def to_dict(self) -> dict:
        with self._lock:
            r = self._safe_round
            return {
                'elbow_angle': r(self.elbow_angle),
                'upper_angle': r(self.upper_angle),
                'valid': bool(self.valid),
                # 预测 EMG
                'pred_biceps_uv': r(self.pred_biceps_uv),
                'pred_triceps_uv': r(self.pred_triceps_uv),
                # 真实 EMG
                'real_biceps_uv': r(self.real_biceps_uv),
                'real_triceps_uv': r(self.real_triceps_uv),
                # 心率
                'heart_rate': self.heart_rate,
                # 交叉验证
                'r2_biceps': r(self.r2_biceps, 4),
                'r2_triceps': r(self.r2_triceps, 4),
                'compensation_score': r(self.compensation_score),
                'compensation_level': str(self.compensation_level or ''),
                'biceps_deficiency': bool(self.biceps_deficiency),
                'triceps_overactivation': bool(self.triceps_overactivation),
                'cocontraction_index': r(self.cocontraction_index),
                'biceps_quality': r(self.biceps_quality),
                'triceps_quality': r(self.triceps_quality),
                'dropout_b': bool(self.dropout_b),
                'dropout_t': bool(self.dropout_t),
                'quality_alerts': list(self.quality_alerts) if self.quality_alerts else [],
                'keypoints': list(self.keypoints) if self.keypoints else [],
                'camera_width': self.camera_width or 640,
                'camera_height': self.camera_height or 480,
                'msg_count': self.msg_count,
                'timestamp': time.time(),
            }


state = SharedState()


# ============================== ROS2 订阅线程 ==============================

def ros2_thread():
    if not HAS_ROS2:
        logger.warning('ROS2 不可用, 跳过订阅')
        return

    # 开机时 ROS2 可能未完全就绪, 重试初始化
    for attempt in range(5):
        try:
            rclpy.init(args=['--ros-args', '--log-level', 'warn'])
            break
        except Exception as e:
            if attempt < 4:
                logger.warning(f'ROS2 初始化失败 (尝试 {attempt+1}/5), 3秒后重试...')
                time.sleep(3)
            else:
                logger.error(f'ROS2 初始化失败, 放弃: {e}')
                return

    node = Node('screen_bridge')

    def _parse_json(msg, updater):
        try: updater(json.loads(msg.data))
        except Exception: pass

    node.create_subscription(String, '/body_arm_angles',
        lambda m: _parse_json(m, state.update_angle), 10)
    node.create_subscription(String, '/virtual_emg',
        lambda m: _parse_json(m, state.update_emg), 10)
    node.create_subscription(String, '/raw_emg',
        lambda m: _parse_json(m, state.update_raw_emg), 10)
    node.create_subscription(String, '/emg_validation',
        lambda m: _parse_json(m, state.update_validation), 10)
    node.create_subscription(String, '/emg_alerts',
        lambda m: _parse_json(m, state.update_alerts), 10)

    try:
        from sensor_msgs.msg import CompressedImage
        def on_img(msg):
            try: state.update_camera_frame(msg.data)
            except Exception: pass
        node.create_subscription(CompressedImage, '/image_jpeg', on_img, 10)
        logger.info('已订阅 /image_jpeg (摄像头)')
    except Exception:
        logger.info('未找到 /image_jpeg 话题')

    logger.info('已订阅 ROS2: angle/emg/raw_emg/validation/alerts')

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


# ============================== HTTP 路由 ==============================

async def camera_stream(request):
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    await response.prepare(request)
    logger.info('MJPEG 客户端已连接')
    try:
        while True:
            frame_bytes, _ = state.get_frame()
            if frame_bytes:
                try:
                    await response.write(
                        b'--frame\r\nContent-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                        + frame_bytes + b'\r\n'
                    )
                except ConnectionResetError: break
            await asyncio.sleep(0.05)
    except (ConnectionResetError, asyncio.CancelledError): pass
    logger.info('MJPEG 客户端已断开')
    return response


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info('WS 客户端已连接')
    last_msg_count = 0
    try:
        while True:
            data = state.to_dict()
            if data['msg_count'] == last_msg_count and last_msg_count > 0:
                await asyncio.sleep(0.05)
                continue
            last_msg_count = data['msg_count']
            try: await ws.send_json(data)
            except ConnectionResetError: break
            await asyncio.sleep(0.05)
    except (ConnectionResetError, asyncio.CancelledError): pass
    logger.info('WS 客户端已断开')
    return ws


async def calibrate_save(request):
    """保存触屏校准结果到 ~/calibration_save.json"""
    try:
        data = await request.json()
        rest_b = float(data.get('rest_b', 200))
        rest_t = float(data.get('rest_t', 80))
        v90_b = float(data.get('v90_b', 500))
        v90_t = float(data.get('v90_t', 120))
    except Exception:
        return web.json_response({'error': 'invalid JSON'}, status=400)

    calib_path = os.path.expanduser('~/calibration_save.json')
    try:
        saved = {}
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                saved = json.load(f)
        calib_vec = saved.get('calib_vec', [200, 500, 500, 5000, 10, 0.1,
                                             200, 500, 500, 5000, 10, 0.1,
                                             170, 70, 22, 0])
        if len(calib_vec) != 16:
            calib_vec = [200, 500, 500, 5000, 10, 0.1, 200, 500, 500, 5000, 10, 0.1, 170, 70, 22, 0]
        calib_vec[0] = rest_b   # b_rest
        calib_vec[1] = v90_b    # b_90
        calib_vec[6] = rest_t   # t_rest
        calib_vec[7] = v90_t    # t_90
        saved['calib_vec'] = [float(v) for v in calib_vec]
        saved['b_rest'] = rest_b
        saved['b_90'] = v90_b
        saved['t_rest'] = rest_t
        saved['t_90'] = v90_t
        saved['source'] = 'screen_calibration'
        saved['timestamp'] = time.time()
        with open(calib_path, 'w') as f:
            json.dump(saved, f, ensure_ascii=False)
        logger.info(f'校准已保存: b_rest={rest_b} b_90={v90_b} t_rest={rest_t} t_90={v90_t}')
        return web.json_response({'status': 'ok'})
    except Exception as e:
        logger.error(f'校准保存失败: {e}')
        return web.json_response({'error': str(e)}, status=500)


async def simulate_update(request):
    """模拟数据注入 — 直接更新共享状态"""
    try:
        data = await request.json()
        with state._lock:
            for key, val in data.items():
                if hasattr(state, key):
                    setattr(state, key, val)
            state.msg_count += 1
            state.last_update = time.time()
        return web.json_response({'status': 'ok'})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=400)


async def health_check(request):
    data = state.to_dict()
    return web.json_response({
        'status': 'ok',
        'msg_count': data['msg_count'],
        'has_camera': state.get_frame()[0] is not None,
        'last_update': data['timestamp'],
    })


async def ai_report(request):
    """AI 报告生成 (DeepSeek API)"""
    if not HAS_DEEPSEEK:
        return web.json_response({'error': 'DeepSeek API 未配置, 请设置 ~/.deepseek_key'}, status=400)

    data = state.to_dict()
    prompt = f"""你是一位康复训练专家。请根据以下患者的实时训练数据，生成一份简洁的康复训练报告（200字以内）：

- 肘关节角度: {data['elbow_angle']}°
- 大臂角度: {data['upper_angle']}°
- 预测肌电(TCN): 二头肌 {data['pred_biceps_uv']}μV, 三头肌 {data['pred_triceps_uv']}μV
- 真实肌电(ESP32): 二头肌 {data['real_biceps_uv']}μV, 三头肌 {data['real_triceps_uv']}μV
- 心率: {data['heart_rate']} BPM
- 代偿评分: {data['compensation_score']} ({data['compensation_level']})
- 电极质量: 二头肌 {data['biceps_quality']}%, 三头肌 {data['triceps_quality']}%
- R² (预测vs真实): 二头肌 {data['r2_biceps']}, 三头肌 {data['r2_triceps']}

请给出：
1. 当前训练状态评价
2. 肌电数据解读（预测vs真实差异分析）
3. 改进建议

直接输出报告，不要有markdown标题。"""

    try:
        timeout = _aiohttp.ClientTimeout(total=30)
        async with _aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                'https://api.deepseek.com/chat/completions',
                headers={
                    'Authorization': f'Bearer {DEEPSEEK_KEY}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': '你是康复训练专家助手。'},
                        {'role': 'user', 'content': prompt},
                    ],
                    'max_tokens': 500,
                    'temperature': 0.7,
                },
            ) as resp:
                result = await resp.json()
                report = result['choices'][0]['message']['content']
                state.set_report(report)
                return web.json_response({'report': report, 'time': time.time()})
    except Exception as e:
        logger.error(f'AI 报告生成失败: {e}')
        return web.json_response({'error': f'生成失败: {str(e)}'}, status=500)


# ============================== 主入口 ==============================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RDK X5 触屏服务')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--static-dir', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screen_ui'))
    args = parser.parse_args()

    static_dir = args.static_dir
    if not os.path.isdir(static_dir):
        logger.warning(f'静态目录不存在: {static_dir}')
        static_dir = None

    threading.Thread(target=ros2_thread, daemon=True, name='ros2-sub').start()
    logger.info('ROS2 订阅线程已启动')

    app = web.Application()
    app.router.add_get('/api/status', health_check)
    app.router.add_post('/api/calibrate/save', calibrate_save)
    app.router.add_post('/api/simulate', simulate_update)
    app.router.add_get('/api/report', ai_report)
    app.router.add_get('/camera/stream', camera_stream)
    app.router.add_get('/ws', ws_handler)

    if static_dir:
        app.router.add_static('/static/', os.path.join(static_dir, 'static'), show_index=False)
        async def index(request):
            return web.FileResponse(os.path.join(static_dir, 'index.html'))
        app.router.add_get('/', index)
        logger.info(f'前端目录: {static_dir}')

    logger.info(f'触屏服务启动: http://0.0.0.0:{args.port}')
    web.run_app(app, host='0.0.0.0', port=args.port, print=lambda *a: None)


if __name__ == '__main__':
    main()
