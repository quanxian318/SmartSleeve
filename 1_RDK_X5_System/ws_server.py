#!/usr/bin/env python3
"""
ws_server.py — RDK WebSocket 服务: 实时推送角度+EMG给小程序

端口: 8765
协议: JSON over WebSocket, 20Hz

数据格式 (完全兼容 dataManager.js):
{
  "left_elbow_angle": 120.0,
  "right_elbow_angle": 115.0,
  "left_upper_angle": 0.0,
  "right_upper_angle": 0.0,
  "left_valid": true,
  "right_valid": true,
  "points": null,

  // 扩展: TCN 预测肌电 (μV)
  "left_biceps_uv": 450.0,
  "left_triceps_uv": 100.0,
  "right_biceps_uv": 430.0,
  "right_triceps_uv": 95.0,

  "timestamp": 1234567890.123
}
"""

import asyncio
import json
import logging
import threading
import time
import sys

import websockets
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

logger = logging.getLogger('ws_server')


# ============================== 共享状态 ==============================

class SharedState:
    """ROS2 → WebSocket 桥接, 线程安全"""

    def __init__(self):
        self._lock = threading.Lock()
        self._angle = {}
        self._emg = {}
        self._validation = {}
        self._alerts = []
        self.msg_count = 0

    def update_angle(self, data):
        with self._lock:
            self._angle = data

    def update_emg(self, data):
        with self._lock:
            self._emg = data

    def update_validation(self, data):
        with self._lock:
            self._validation = data

    def update_alerts(self, data):
        with self._lock:
            self._alerts = data.get('alerts', [])

    def build_message(self):
        """合并骨架+EMG+验证+告警, 生成推送消息"""
        with self._lock:
            angle = dict(self._angle)
            emg = dict(self._emg)
            validation = dict(self._validation)
            alerts = list(self._alerts)

        left = emg.get('left', {})
        right = emg.get('right', {})

        # 提取代偿信息
        comp = validation.get('compensation', {})
        elec = validation.get('electrode_health', {})

        msg = {
            # ── 骨架 (dataManager.js 必需字段) ──
            'left_elbow_angle':  float(angle.get('left_elbow_angle', 0)),
            'right_elbow_angle': float(angle.get('right_elbow_angle', 0)),
            'left_upper_angle':  float(angle.get('left_upper_arm_angle', 0)),
            'right_upper_angle': float(angle.get('right_upper_arm_angle', 0)),
            'left_valid':  bool(angle.get('left_valid', False)),
            'right_valid': bool(angle.get('right_valid', False)),
            'points':      angle.get('points', None),

            # ── TCN 预测肌电 (μV) ──
            'left_biceps_uv':   float(left.get('biceps_uv', 0)),
            'left_triceps_uv':  float(left.get('triceps_uv', 0)),
            'right_biceps_uv':  float(right.get('biceps_uv', 0)),
            'right_triceps_uv': float(right.get('triceps_uv', 0)),

            # ── 肌肉代偿 ──
            'compensation_score': comp.get('compensation_score', 0),
            'compensation_level': comp.get('compensation_level', 'unknown'),
            'biceps_deficiency':  comp.get('biceps_deficiency', 0),
            'triceps_overactivation': comp.get('triceps_overactivation', 0),
            'cocontraction_index': comp.get('cocontraction_index', 0),

            # ── 电极健康 ──
            'biceps_quality':   elec.get('biceps_quality', 100),
            'triceps_quality':  elec.get('triceps_quality', 100),
            'dropout_b':        elec.get('dropout_b', False),
            'dropout_t':        elec.get('dropout_t', False),
            'impedance_b':      elec.get('impedance_issue_b', False),
            'impedance_t':      elec.get('impedance_issue_t', False),

            # ── 交叉验证摘要 ──
            'r2_biceps':  (validation.get('overall', {}).get('biceps', {}) or {}).get('r2'),
            'r2_triceps': (validation.get('overall', {}).get('triceps', {}) or {}).get('r2'),

            # ── 活跃告警 ──
            'quality_alerts': alerts,

            'timestamp': time.time(),
        }
        self.msg_count += 1
        return msg


state = SharedState()


# ============================== ROS2 订阅线程 ==============================

def ros2_thread():
    """后台线程: 订阅 /body_arm_angles + /virtual_emg"""
    rclpy.init(args=['--ros-args', '--log-level', 'warn'])

    node = Node('ws_bridge')

    def on_angle(msg):
        try:
            state.update_angle(json.loads(msg.data))
        except Exception:
            pass

    def on_emg(msg):
        try:
            state.update_emg(json.loads(msg.data))
        except Exception:
            pass

    def on_validation(msg):
        try:
            state.update_validation(json.loads(msg.data))
        except Exception:
            pass

    def on_alerts(msg):
        try:
            state.update_alerts(json.loads(msg.data))
        except Exception:
            pass

    node.create_subscription(String, '/body_arm_angles', on_angle, 10)
    node.create_subscription(String, '/virtual_emg', on_emg, 10)
    node.create_subscription(String, '/emg_validation', on_validation, 10)
    node.create_subscription(String, '/emg_alerts', on_alerts, 10)

    logger.info('已订阅 /body_arm_angles + /virtual_emg + /emg_validation + /emg_alerts')

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


# ============================== WebSocket 服务 ==============================

CONNECTED = set()

async def ws_handler(websocket):
    """每客户端一个协程"""
    CONNECTED.add(websocket)
    peer = f'{websocket.remote_address[0]}:{websocket.remote_address[1]}'
    logger.info(f'客户端连接: {peer}  (共 {len(CONNECTED)} 个)')

    try:
        # 等待 ROS2 数据就绪 (最多 5 秒)
        for _ in range(50):
            if state.msg_count > 0:
                break
            await asyncio.sleep(0.1)

        # 主循环: 20Hz 推送
        while True:
            msg = state.build_message()
            await websocket.send(json.dumps(msg, ensure_ascii=False))
            await asyncio.sleep(0.05)

    except websockets.exceptions.ConnectionClosed:
        logger.info(f'客户端断开: {peer}')
    except Exception as e:
        logger.error(f'错误 [{peer}]: {e}')
    finally:
        CONNECTED.discard(websocket)


async def ws_main():
    logger.info('WebSocket 服务启动: ws://0.0.0.0:8765')
    async with websockets.serve(
        ws_handler, '0.0.0.0', 8765,
        ping_interval=30, ping_timeout=10,
        max_size=2 ** 20
    ):
        await asyncio.Future()  # 永远运行


# ============================== 主入口 ==============================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[WS] %(asctime)s %(message)s',
        datefmt='%H:%M:%S'
    )

    logger.info('启动中...')

    # 1. ROS2 后台线程
    ros2_t = threading.Thread(target=ros2_thread, name='ros2-ws', daemon=True)
    ros2_t.start()
    time.sleep(1.5)

    # 2. WebSocket (asyncio 主循环)
    try:
        asyncio.run(ws_main())
    except KeyboardInterrupt:
        logger.info('服务已停止')


if __name__ == '__main__':
    main()
