#!/usr/bin/env python3
import math
import time
import json
import rclpy
from rclpy.node import Node

from ai_msgs.msg import PerceptionTargets
from std_msgs.msg import String


# COCO人体关键点编号
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def calc_joint_angle(a, b, c):
    """
    计算三点夹角，角点在 b。
    例如：肩-肘-腕，计算的是肘关节角度。
    a,b,c 格式: (x, y)
    """
    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]
    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    dot = ba_x * bc_x + ba_y * bc_y
    norm_ba = math.sqrt(ba_x * ba_x + ba_y * ba_y)
    norm_bc = math.sqrt(bc_x * bc_x + bc_y * bc_y)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cos_value = dot / (norm_ba * norm_bc)
    cos_value = max(-1.0, min(1.0, cos_value))

    angle = math.degrees(math.acos(cos_value))
    return angle


def calc_direction_angle(a, b):
    """
    计算从 a 点指向 b 点的方向角。
    图像坐标系中 y 轴向下，这里用 -dy 转成数学坐标习惯。
    返回范围大致为 [-180, 180]
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    angle = math.degrees(math.atan2(-dy, dx))
    return angle


def ema_filter(current, previous, alpha=0.6):
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


def smooth_point(curr_pt, prev_pt, alpha=0.6):
    if curr_pt is None:
        return prev_pt
    if prev_pt is None:
        return curr_pt

    x = ema_filter(curr_pt[0], prev_pt[0], alpha)
    y = ema_filter(curr_pt[1], prev_pt[1], alpha)
    return (x, y)


def smooth_angle(curr_angle, prev_angle, alpha=0.5):
    if curr_angle is None:
        return prev_angle
    if prev_angle is None:
        return curr_angle
    return ema_filter(curr_angle, prev_angle, alpha)


def suppress_jump(curr_angle, prev_angle, max_jump):
    """
    普通角度跳变抑制，适用于肘夹角
    """
    if curr_angle is None:
        return prev_angle
    if prev_angle is None:
        return curr_angle

    if abs(curr_angle - prev_angle) > max_jump:
        return prev_angle
    return curr_angle


def angle_diff_deg(a, b):
    """
    计算两个方向角之间的最短差值，范围 [-180, 180]
    """
    diff = a - b
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def suppress_direction_jump(curr_angle, prev_angle, max_jump):
    """
    方向角跳变抑制，处理 -180~180 环绕问题
    """
    if curr_angle is None:
        return prev_angle
    if prev_angle is None:
        return curr_angle

    diff = angle_diff_deg(curr_angle, prev_angle)
    if abs(diff) > max_jump:
        return prev_angle
    return curr_angle


class BodyAngleNode(Node):
    def __init__(self):
        super().__init__('body_angle_node')

        self.sub = self.create_subscription(
            PerceptionTargets,
            '/hobot_mono2d_body_detection',
            self.listener_callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            '/body_arm_angles',
            10
        )

        # 平滑历史缓存：按 track_id 保存
        self.history = {}

        # 平滑参数
        self.point_alpha = 0.6
        self.angle_alpha = 0.5

        # 短时丢帧补偿
        self.hold_timeout = 0.3

        # 发布频率控制
        self.publish_interval = 0.1
        self.last_publish_time = 0.0

        # 跳变抑制阈值
        self.max_joint_jump = 35.0
        self.max_dir_jump = 45.0

        # 历史清理计时 (防止 track_id 丢失导致的无限增长)
        self._last_cleanup = 0.0

        self.get_logger().info('body_angle_node started.')
        self.get_logger().info('Subscribing: /hobot_mono2d_body_detection')
        self.get_logger().info('Publishing : /body_arm_angles (std_msgs/String JSON)')

    def init_track_history(self, track_id):
        if track_id not in self.history:
            self.history[track_id] = {
                'left_shoulder': None,
                'left_elbow': None,
                'left_wrist': None,
                'right_shoulder': None,
                'right_elbow': None,
                'right_wrist': None,

                'left_elbow_angle': None,
                'left_upper_arm_angle': None,
                'left_forearm_angle': None,

                'right_elbow_angle': None,
                'right_upper_arm_angle': None,
                'right_forearm_angle': None,

                'left_result': None,
                'right_result': None,

                'left_last_valid_time': 0.0,
                'right_last_valid_time': 0.0,
            }

    def round_or_none(self, value, ndigits=2):
        if value is None:
            return None
        return round(float(value), ndigits)

    def publish_angle_json(self, track_id, left_valid, right_valid, left_result, right_result, timestamp,
                           points=None, confs=None):
        msg_dict = {
            'track_id': int(track_id),
            'timestamp': float(timestamp),
            'left_valid': bool(left_valid),
            'right_valid': bool(right_valid),

            'left_elbow_angle': self.round_or_none(left_result[0]) if left_result else None,
            'left_upper_angle': self.round_or_none(left_result[1]) if left_result else None,
            'left_forearm_angle': self.round_or_none(left_result[2]) if left_result else None,

            'right_elbow_angle': self.round_or_none(right_result[0]) if right_result else None,
            'right_upper_angle': self.round_or_none(right_result[1]) if right_result else None,
            'right_forearm_angle': self.round_or_none(right_result[2]) if right_result else None,
        }

        # 添加 17 个 COCO 关键点 (像素坐标 + 置信度)
        if points is not None and confs is not None and len(points) >= 17:
            kps = []
            for i in range(17):
                if i < len(points) and i < len(confs) and confs[i] >= 0.1:
                    kps.append({'x': round(points[i].x, 1), 'y': round(points[i].y, 1), 'c': round(float(confs[i]), 3)})
                else:
                    kps.append(None)
            msg_dict['points'] = kps

        ros_msg = String()
        ros_msg.data = json.dumps(msg_dict, ensure_ascii=False)

        self.get_logger().info(
            f"L角={msg_dict.get('left_elbow_angle')} "
            f"R角={msg_dict.get('right_elbow_angle')} "
            f"LV={msg_dict.get('left_valid')} "
            f"RV={msg_dict.get('right_valid')}"
        )

        self.pub.publish(ros_msg)

    def listener_callback(self, msg):
        now = time.time()

        # 每60秒清理一次超时的历史记录 (>10秒无更新的track)
        if now - self._last_cleanup > 60.0:
            stale_ids = [
                tid for tid, h in self.history.items()
                if now - max(h['left_last_valid_time'], h['right_last_valid_time']) > 10.0
            ]
            for tid in stale_ids:
                del self.history[tid]
            self._last_cleanup = now

        # 只限制发布，不限制计算
        allow_publish = (now - self.last_publish_time) >= self.publish_interval

        for target in msg.targets:
            if target.type != 'person':
                continue

            track_id = target.track_id
            self.init_track_history(track_id)
            hist = self.history[track_id]

            body_kps = None
            for pts in target.points:
                if pts.type == 'body_kps':
                    body_kps = pts
                    break

            if body_kps is None:
                continue

            points = body_kps.point
            confs = body_kps.confidence

            if len(points) < 17:
                self.get_logger().warn(f'关键点数量不足，当前数量: {len(points)}')
                continue

            conf_th = 0.2

            def valid(idx):
                if idx >= len(points):
                    return False
                if idx >= len(confs):
                    return False
                return confs[idx] >= conf_th

            def get_xy(idx):
                return (points[idx].x, points[idx].y)

            # =========================
            # 左臂
            # =========================
            left_result = None
            left_valid = False

            if valid(LEFT_SHOULDER) and valid(LEFT_ELBOW) and valid(LEFT_WRIST):
                curr_left_shoulder = get_xy(LEFT_SHOULDER)
                curr_left_elbow = get_xy(LEFT_ELBOW)
                curr_left_wrist = get_xy(LEFT_WRIST)

                left_shoulder = smooth_point(
                    curr_left_shoulder, hist['left_shoulder'], self.point_alpha
                )
                left_elbow = smooth_point(
                    curr_left_elbow, hist['left_elbow'], self.point_alpha
                )
                left_wrist = smooth_point(
                    curr_left_wrist, hist['left_wrist'], self.point_alpha
                )

                left_elbow_angle = calc_joint_angle(left_shoulder, left_elbow, left_wrist)
                left_upper_arm_angle = calc_direction_angle(left_shoulder, left_elbow)
                left_forearm_angle = calc_direction_angle(left_elbow, left_wrist)

                left_elbow_angle = suppress_jump(
                    left_elbow_angle,
                    hist['left_elbow_angle'],
                    self.max_joint_jump
                )
                left_upper_arm_angle = suppress_direction_jump(
                    left_upper_arm_angle,
                    hist['left_upper_arm_angle'],
                    self.max_dir_jump
                )
                left_forearm_angle = suppress_direction_jump(
                    left_forearm_angle,
                    hist['left_forearm_angle'],
                    self.max_dir_jump
                )

                left_elbow_angle = smooth_angle(
                    left_elbow_angle, hist['left_elbow_angle'], self.angle_alpha
                )
                left_upper_arm_angle = smooth_angle(
                    left_upper_arm_angle, hist['left_upper_arm_angle'], self.angle_alpha
                )
                left_forearm_angle = smooth_angle(
                    left_forearm_angle, hist['left_forearm_angle'], self.angle_alpha
                )

                hist['left_shoulder'] = left_shoulder
                hist['left_elbow'] = left_elbow
                hist['left_wrist'] = left_wrist

                hist['left_elbow_angle'] = left_elbow_angle
                hist['left_upper_arm_angle'] = left_upper_arm_angle
                hist['left_forearm_angle'] = left_forearm_angle

                left_result = (
                    left_elbow_angle,
                    left_upper_arm_angle,
                    left_forearm_angle
                )
                hist['left_result'] = left_result
                hist['left_last_valid_time'] = now
                left_valid = True

            if left_result is None:
                if (now - hist['left_last_valid_time'] <= self.hold_timeout and
                        hist['left_result'] is not None):
                    left_result = hist['left_result']
                    left_valid = True

            # =========================
            # 右臂
            # =========================
            right_result = None
            right_valid = False

            if valid(RIGHT_SHOULDER) and valid(RIGHT_ELBOW) and valid(RIGHT_WRIST):
                curr_right_shoulder = get_xy(RIGHT_SHOULDER)
                curr_right_elbow = get_xy(RIGHT_ELBOW)
                curr_right_wrist = get_xy(RIGHT_WRIST)

                right_shoulder = smooth_point(
                    curr_right_shoulder, hist['right_shoulder'], self.point_alpha
                )
                right_elbow = smooth_point(
                    curr_right_elbow, hist['right_elbow'], self.point_alpha
                )
                right_wrist = smooth_point(
                    curr_right_wrist, hist['right_wrist'], self.point_alpha
                )

                right_elbow_angle = calc_joint_angle(right_shoulder, right_elbow, right_wrist)
                right_upper_arm_angle = calc_direction_angle(right_shoulder, right_elbow)
                right_forearm_angle = calc_direction_angle(right_elbow, right_wrist)

                right_elbow_angle = suppress_jump(
                    right_elbow_angle,
                    hist['right_elbow_angle'],
                    self.max_joint_jump
                )
                right_upper_arm_angle = suppress_direction_jump(
                    right_upper_arm_angle,
                    hist['right_upper_arm_angle'],
                    self.max_dir_jump
                )
                right_forearm_angle = suppress_direction_jump(
                    right_forearm_angle,
                    hist['right_forearm_angle'],
                    self.max_dir_jump
                )

                right_elbow_angle = smooth_angle(
                    right_elbow_angle, hist['right_elbow_angle'], self.angle_alpha
                )
                right_upper_arm_angle = smooth_angle(
                    right_upper_arm_angle, hist['right_upper_arm_angle'], self.angle_alpha
                )
                right_forearm_angle = smooth_angle(
                    right_forearm_angle, hist['right_forearm_angle'], self.angle_alpha
                )

                hist['right_shoulder'] = right_shoulder
                hist['right_elbow'] = right_elbow
                hist['right_wrist'] = right_wrist

                hist['right_elbow_angle'] = right_elbow_angle
                hist['right_upper_arm_angle'] = right_upper_arm_angle
                hist['right_forearm_angle'] = right_forearm_angle

                right_result = (
                    right_elbow_angle,
                    right_upper_arm_angle,
                    right_forearm_angle
                )
                hist['right_result'] = right_result
                hist['right_last_valid_time'] = now
                right_valid = True

            if right_result is None:
                if (now - hist['right_last_valid_time'] <= self.hold_timeout and
                        hist['right_result'] is not None):
                    right_result = hist['right_result']
                    right_valid = True

            # =========================
            # 正式 ROS2 发布
            # =========================
            if allow_publish:
                self.publish_angle_json(
                    track_id=track_id,
                    left_valid=left_valid,
                    right_valid=right_valid,
                    left_result=left_result,
                    right_result=right_result,
                    timestamp=now,
                    points=points,
                    confs=confs
                )
                self.last_publish_time = now
                break


def main(args=None):
    rclpy.init(args=args)
    node = BodyAngleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
