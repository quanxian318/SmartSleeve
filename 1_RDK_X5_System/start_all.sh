#!/bin/bash
# ============================================================
#  智能健康袖套 — 一键启动脚本
#  启动: ./start_all.sh
#  停止: Ctrl+C (自动清理所有子进程)
# ============================================================

set -e
SCRIPT_DIR=$(dirname $(readlink -f "$0"))
LOG_DIR=$HOME/logs/emg_system
mkdir -p $LOG_DIR

# 禁用 Python 输出缓冲, 确保日志实时写入
export PYTHONUNBUFFERED=1

# 加载 DeepSeek API Key (语音助手 AI 对话)
if [ -f "$HOME/.deepseek_key" ]; then
    export DEEPSEEK_API_KEY=$(cat "$HOME/.deepseek_key")
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
PASS="${GREEN}[OK]${NC}"; FAIL="${RED}[FAIL]${NC}"; WARN="${YELLOW}[WARN]${NC}"

PID_LIST=""
cleanup() {
    echo -e "\n${YELLOW}正在停止所有节点...${NC}"
    for pid in $PID_LIST; do
        kill $pid 2>/dev/null && echo "  已停止 PID=$pid"
    done
    echo -e "${GREEN}所有节点已停止${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ==================== 1. 环境检查 ====================
echo -e "${CYAN}━━━ 1. 环境检查 ━━━${NC}"

# ROS2
if source /opt/ros/humble/setup.bash 2>/dev/null; then
    echo -e "  $PASS ROS2 Humble"
else
    echo -e "  $FAIL ROS2 未安装"
    exit 1
fi

# TogetheROS (地平线 RDK AI SDK)
TROS_OK=false
if source /opt/tros/humble/setup.bash 2>/dev/null; then
    echo -e "  $PASS TogetheROS (ai_msgs)"
    TROS_OK=true
else
    echo -e "  $WARN TogetheROS 未安装 (骨架追踪不可用)"
fi

# 摄像头
if [ -e /dev/video0 ]; then
    echo -e "  $PASS 摄像头 /dev/video0"
else
    echo -e "  $WARN 摄像头未检测到 (骨架追踪不可用)"
fi

# 音箱
SPK_CARD=$(grep -i duplex /proc/asound/cards | head -1 | awk '{print $1}')
if [ -n "$SPK_CARD" ]; then
    echo -e "  $PASS 音箱 card$SPK_CARD (ES8326)"
else
    echo -e "  $WARN ES8326 未检测到"
fi

# 麦克风
MIC_CARD=$(grep -iv 'SDYH\|duplex\|es8326\|simple-card' /proc/asound/cards | grep -i 'USB\|Device\|Jieli' | head -1 | awk '{print $1}')
if [ -n "$MIC_CARD" ]; then
    echo -e "  $PASS 麦克风 card$MIC_CARD"
else
    echo -e "  $WARN USB麦克风未检测到"
fi

# TCN 模型
MODEL_DIR=$HOME
MODEL_FILES=("anchorcalib_tcn_bpu_v2.bin" "motion_scaler_63subj.pkl" "calib_scaler_63subj.pkl" "calibration_config_63subj.json")
ALL_OK=true
for f in "${MODEL_FILES[@]}"; do
    if [ -f "$MODEL_DIR/$f" ]; then
        echo -e "  $PASS $f"
    else
        echo -e "  $FAIL $f 缺失"
        ALL_OK=false
    fi
done
$ALL_OK || exit 1

# DeepSeek API Key (语音AI功能)
if [ -f $HOME/.deepseek_key ]; then
    export DEEPSEEK_API_KEY=$(cat $HOME/.deepseek_key)
    echo -e "  $PASS DeepSeek API Key 已加载"
elif [ -n "$DEEPSEEK_API_KEY" ]; then
    echo -e "  $PASS DeepSeek API Key (环境变量)"
else
    echo -e "  $WARN DeepSeek API Key 未配置 (语音AI功能不可用)"
    echo -e "  ${CYAN}  创建 ~/.deepseek_key 文件并写入你的API Key${NC}"
fi

# 用户体征 (默认值，可通过参数覆盖)
BMI=${BMI:-22.0}
HEIGHT=${HEIGHT:-170.0}
WEIGHT=${WEIGHT:-70.0}
GENDER=${GENDER:-0}
B_REST=${B_REST:-200}
B_90=${B_90:-500}
T_REST=${T_REST:-80}
T_90=${T_90:-120}

echo -e "  BMI=$BMI  身高=${HEIGHT}cm  体重=${WEIGHT}kg"
echo -e "  校准: 二头rest=${B_REST} v90=${B_90}  三头rest=${T_REST} v90=${T_90}"

# ==================== 2. 启动节点 ====================
echo ""
echo -e "${CYAN}━━━ 2. 启动节点 ━━━${NC}"

# 构建共用校准参数
CALIB_PARAMS="--b_rest $B_REST --b_90 $B_90 --t_rest $T_REST --t_90 $T_90 --bmi $BMI --height $HEIGHT --weight $WEIGHT --gender $GENDER"

# ---- 节点1: 骨架追踪 (摄像头 + AI检测 + 角度计算) ----
# 自动检测摄像头类型: USB vs MIPI
BODY_ANGLE_PID=""
BODY_DET_PID=""
CAM_TYPE="none"
if [ -e /dev/video0 ] && $TROS_OK && [ -f $HOME/body_angle_node.py ]; then
    # 检测是USB摄像头还是MIPI（通过sysfs）
    CAM_TYPE="mipi"
    if readlink /sys/class/video4linux/video0 2>/dev/null | grep -qi "usb"; then
        CAM_TYPE="usb"
    fi
    echo -e "  [1/5] 骨架追踪 (摄像头: ${CAM_TYPE})..."

    if [ "$CAM_TYPE" = "usb" ]; then
        # === USB 摄像头路径 ===
        # 1a: USB摄像头节点 (→ /image rgb8)
        echo -e "       启动 USB 摄像头..."
        ros2 launch hobot_usb_cam hobot_usb_cam.launch.py \
            usb_video_device:=/dev/video0 \
            usb_image_width:=640 usb_image_height:=480 \
            usb_pixel_format:=yuyv2rgb usb_framerate:=30 \
            &> $LOG_DIR/usb_cam.log &
        USB_CAM_PID=$!
        PID_LIST="$PID_LIST $USB_CAM_PID"
        echo -e "       $PASS hobot_usb_cam (PID=$USB_CAM_PID)"
        sleep 3

        # 1b: Codec 编码 (rgb8 → jpeg)
        echo -e "       启动 codec 编码 (rgb8→jpeg)..."
        ros2 launch hobot_codec hobot_codec_encode.launch.py \
            codec_in_mode:=ros codec_in_format:=rgb8 \
            codec_out_mode:=ros codec_out_format:=jpeg \
            codec_sub_topic:=/image codec_pub_topic:=/image_jpeg \
            codec_jpg_quality:=80.0 log_level:=error \
            &> $LOG_DIR/codec_encode.log &
        CODEC_ENC_PID=$!
        PID_LIST="$PID_LIST $CODEC_ENC_PID"
        echo -e "       $PASS codec_encode (PID=$CODEC_ENC_PID)"
        sleep 2

        # 1c: Codec 解码 (jpeg → 共享内存 nv12)
        echo -e "       启动 codec 解码 (jpeg→共享内存)..."
        ros2 launch hobot_codec hobot_codec_decode.launch.py \
            codec_in_mode:=ros codec_in_format:=jpeg \
            codec_out_mode:=shared_mem codec_out_format:=nv12 \
            codec_sub_topic:=/image_jpeg codec_pub_topic:=/hbmem_img \
            log_level:=warn \
            &> $LOG_DIR/codec_decode.log &
        CODEC_DEC_PID=$!
        PID_LIST="$PID_LIST $CODEC_DEC_PID"
        echo -e "       $PASS codec_decode (PID=$CODEC_DEC_PID)"
        sleep 2

        # 1d: AI 人体检测 (共享内存 → /hobot_mono2d_body_detection)
        echo -e "       启动 AI 人体检测..."
        ros2 launch mono2d_body_detection mono2d_body_detection_without_camera.launch.py \
            &> $LOG_DIR/body_detection.log &
        BODY_DET_PID=$!
        PID_LIST="$PID_LIST $BODY_DET_PID"
        echo -e "       $PASS mono2d_body_detection (PID=$BODY_DET_PID)"
        sleep 3
    else
        # === MIPI 摄像头路径 ===
        # 1a: MIPI摄像头 (→ /hbmem_img)
        echo -e "       启动 MIPI 摄像头..."
        ros2 launch mipi_cam mipi_cam.launch.py \
            &> $LOG_DIR/mipi_cam.log &
        MIPI_CAM_PID=$!
        PID_LIST="$PID_LIST $MIPI_CAM_PID"
        echo -e "       $PASS mipi_cam (PID=$MIPI_CAM_PID)"
        sleep 3

        # 1b: AI 人体检测 (共享内存 → /hobot_mono2d_body_detection)
        echo -e "       启动 AI 人体检测..."
        ros2 launch mono2d_body_detection mono2d_body_detection.launch.py \
            &> $LOG_DIR/body_detection.log &
        BODY_DET_PID=$!
        PID_LIST="$PID_LIST $BODY_DET_PID"
        echo -e "       $PASS mono2d_body_detection (PID=$BODY_DET_PID)"
        sleep 3
    fi

    # 1e: 骨架角度计算 (关键点 → 关节角度 → /body_arm_angles)
    echo -e "       启动角度计算 (body_angle_node)..."
    export PYTHONPATH="/opt/tros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
    python3 $HOME/body_angle_node.py &> $LOG_DIR/body_angle.log &
    BODY_ANGLE_PID=$!
    PID_LIST="$PID_LIST $BODY_ANGLE_PID"
    echo -e "       $PASS body_angle_node (PID=$BODY_ANGLE_PID)"
    sleep 2
else
    echo -e "  [1/5] 骨架追踪 ${YELLOW}跳过${NC} (无摄像头/无TROS)"
fi

# ---- 节点2: EMG 桥接 (TCN ONNX推理) ----
echo -e "  [2/5] EMG桥接 (AnchorCalib-TCN)..."
python3 $HOME/ros2_emg_bridge_v2.py \
    --ml_predict \
    --tcn_bin $HOME/anchorcalib_tcn_bpu_v2.bin \
    --motion_scaler $HOME/motion_scaler_63subj.pkl \
    --calib_scaler $HOME/calib_scaler_63subj.pkl \
    --calib_config $HOME/calibration_config_63subj.json \
    --load_calib \
    $CALIB_PARAMS \
    &> $LOG_DIR/bridge.log &
PID_LIST="$PID_LIST $!"
BRIDGE_PID=$!
echo -e "    $PASS ros2_emg_bridge_v2 (PID=$!)"
sleep 1

# ---- 节点2b: UDP EMG接收器 (ESP32 → ROS2 /raw_emg, 含BPM心率) ----
echo -e "  [2b] UDP EMG接收器 (端口8766, 含BPM)..."
python3 $HOME/udp_emg_receiver.py \
    --ros2 \
    --port 8766 \
    --rate 20 \
    &> $LOG_DIR/udp_emg.log &
PID_LIST="$PID_LIST $!"
UDP_EMG_PID=$!
echo -e "    $PASS udp_emg_receiver (PID=$!, 端口8766 → /raw_emg)"
sleep 1

# ---- 节点3: 交叉验证 (TCN预测 vs 真实EMG + 质量监测) ----
echo -e "  [3/5] 交叉验证 (TCN + 代偿/电极检测)..."
python3 $HOME/emg_cross_validation_v2.py \
    --simulate_real \
    --auto_calibrate \
    --load_calib \
    --emit_interval 10 \
    --tcn_bin $HOME/anchorcalib_tcn_bpu_v2.bin \
    --motion_scaler $HOME/motion_scaler_63subj.pkl \
    --calib_scaler $HOME/calib_scaler_63subj.pkl \
    --calib_config $HOME/calibration_config_63subj.json \
    $CALIB_PARAMS \
    &> $LOG_DIR/crossval.log &
PID_LIST="$PID_LIST $!"
CROSSVAL_PID=$!
echo -e "    $PASS emg_cross_validation_v2 (PID=$!)"
sleep 1

# ---- 节点4: 语音助手 (可选, 需 --with-voice) ----
VOICE_PID=""
if [ "${WITH_VOICE:-0}" = "1" ]; then
    VOICE_SCRIPT=""
    # v7 优先 (暂停/恢复、实时告警、AI对话)
    if [ -f $HOME/voice_demo_v7.py ]; then
        VOICE_SCRIPT="$HOME/voice_demo_v7.py"
    elif [ -f $HOME/voice_demo_v6.py ]; then
        VOICE_SCRIPT="$HOME/voice_demo_v6.py"
    elif [ -f $HOME/voice_demo.py ]; then
        VOICE_SCRIPT="$HOME/voice_demo.py"
    fi

    if [ -n "$VOICE_SCRIPT" ]; then
        echo -e "  [4/5] 语音助手 (VAD守护模式)..."
        if [ "$VOICE_SCRIPT" = "$HOME/voice_demo_v7.py" ]; then
            # v7: 支持暂停/恢复、实时告警、AI对话
            AI_FLAG=""
            [ -z "$DEEPSEEK_API_KEY" ] && AI_FLAG="--no-llm"
            python3 $VOICE_SCRIPT --daemon $AI_FLAG &> $LOG_DIR/voice.log &
            VOICE_PID=$!
            PID_LIST="$PID_LIST $VOICE_PID"
            echo -e "    $PASS 语音助手 v7 (PID=$VOICE_PID)"
            if [ -n "$DEEPSEEK_API_KEY" ]; then
                echo -e "    ${CYAN}  AI对话已启用 | 暂停/恢复 | 实时告警${NC}"
            else
                echo -e "    ${CYAN}  基础指令 | 暂停/恢复 | 实时告警${NC}"
            fi
        elif [ "$VOICE_SCRIPT" = "$HOME/voice_demo_v6.py" ]; then
            python3 $VOICE_SCRIPT --daemon &> $LOG_DIR/voice.log &
            VOICE_PID=$!
            PID_LIST="$PID_LIST $VOICE_PID"
            echo -e "    $PASS 语音助手 v6 (PID=$VOICE_PID, VAD自动触发)"
            echo -e "    ${CYAN}  直接说话即可控制, 无需按键${NC}"
        else
            echo -e "    ${YELLOW}v5 需要交互模式, 请在新终端手动运行:${NC}"
            echo -e "    ${CYAN}  python3 $VOICE_SCRIPT${NC}"
        fi
    else
        echo -e "  [4/5] 语音助手 ${YELLOW}跳过${NC} (脚本未找到)"
    fi
else
    echo -e "  [4/5] 语音助手 ${YELLOW}跳过${NC} (使用 --with-voice 开启)"
fi

# ---- 节点5: WebSocket 服务 (小程序连接) ----
echo -e "  [5/5] WebSocket 服务..."
if [ -f $HOME/ws_server.py ]; then
    python3 $HOME/ws_server.py &> $LOG_DIR/ws_server.log &
    WS_PID=$!
    PID_LIST="$PID_LIST $WS_PID"
    echo -e "    $PASS ws_server (PID=$WS_PID, ws://0.0.0.0:8765)"
    echo -e "    ${CYAN}  小程序可通过 ws://IP:8765 连接${NC}"
else
    echo -e "  [5/5] WebSocket ${YELLOW}跳过${NC} (ws_server.py 未找到)"
    WS_PID=""
fi

# ==================== 3. 运行监控 ====================
echo ""
echo -e "${CYAN}━━━ 3. 运行中 ━━━${NC}"
echo -e "  ${GREEN}全系统已启动!${NC}"
echo -e "  日志目录: $LOG_DIR"
echo -e "  按 ${YELLOW}Ctrl+C${NC} 停止所有节点"
echo ""
echo -e "${CYAN}━━━ ROS2 话题 ━━━${NC}"

# 等待话题出现
sleep 3
TIMEOUT=15
while [ $TIMEOUT -gt 0 ]; do
    TOPICS=$(ros2 topic list 2>/dev/null)
    if echo "$TOPICS" | grep -q "virtual_emg"; then
        break
    fi
    sleep 1
    TIMEOUT=$((TIMEOUT - 1))
done

echo ""
ros2 topic list 2>/dev/null | head -20 | while read t; do
    echo "    $t"
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  系统运行中 | 日志: tail -f $LOG_DIR/*.log${NC}"
echo -e "${GREEN}  Ctrl+C 停止${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# 主循环：监控子进程，崩溃自动重启
while true; do
    # 检查 bridge
    if ! kill -0 $BRIDGE_PID 2>/dev/null; then
        echo -e "  ${RED}[$(date +%H:%M:%S)] Bridge 崩溃, 重启中...${NC}"
        python3 $HOME/ros2_emg_bridge_v2.py --ml_predict --load_calib \
            --tcn_bin $HOME/anchorcalib_tcn_bpu_v2.bin \
            --motion_scaler $HOME/motion_scaler_63subj.pkl \
            --calib_scaler $HOME/calib_scaler_63subj.pkl \
            --calib_config $HOME/calibration_config_63subj.json \
            $CALIB_PARAMS \
            &>> $LOG_DIR/bridge.log &
        BRIDGE_PID=$!
        PID_LIST="$PID_LIST $BRIDGE_PID"
        echo -e "  ${GREEN}  Bridge 已重启 PID=$BRIDGE_PID${NC}"
    fi

    # 检查 udp_emg_receiver (ESP32 UDP → /raw_emg)
    if [ -n "$UDP_EMG_PID" ] && ! kill -0 $UDP_EMG_PID 2>/dev/null; then
        echo -e "  ${RED}[$(date +%H:%M:%S)] UDP_EMG 崩溃, 重启中...${NC}"
        python3 $HOME/udp_emg_receiver.py --ros2 --port 8766 --rate 20 \
            &>> $LOG_DIR/udp_emg.log &
        UDP_EMG_PID=$!
        PID_LIST="$PID_LIST $UDP_EMG_PID"
        echo -e "  ${GREEN}  UDP_EMG 已重启 PID=$UDP_EMG_PID${NC}"
    fi

    # 检查 crossval
    if ! kill -0 $CROSSVAL_PID 2>/dev/null; then
        echo -e "  ${RED}[$(date +%H:%M:%S)] CrossVal 崩溃, 重启中...${NC}"
        python3 $HOME/emg_cross_validation_v2.py --simulate_real --auto_calibrate --load_calib \
            --emit_interval 10 \
            --tcn_bin $HOME/anchorcalib_tcn_bpu_v2.bin \
            --motion_scaler $HOME/motion_scaler_63subj.pkl \
            --calib_scaler $HOME/calib_scaler_63subj.pkl \
            --calib_config $HOME/calibration_config_63subj.json \
            $CALIB_PARAMS \
            &>> $LOG_DIR/crossval.log &
        CROSSVAL_PID=$!
        PID_LIST="$PID_LIST $CROSSVAL_PID"
        echo -e "  ${GREEN}  CrossVal 已重启 PID=$CROSSVAL_PID${NC}"
    fi

    # 检查 voice (v6 daemon 模式)
    if [ -n "$VOICE_PID" ] && ! kill -0 $VOICE_PID 2>/dev/null; then
        echo -e "  ${RED}[$(date +%H:%M:%S)] Voice 崩溃, 重启中...${NC}"
        # v7 崩溃重启
        VOICE_SCRIPT_NAME="voice_demo_v7.py"
        [ ! -f "$HOME/$VOICE_SCRIPT_NAME" ] && VOICE_SCRIPT_NAME="voice_demo_v6.py"
        AI_FLAG=""
        [ -z "$DEEPSEEK_API_KEY" ] && AI_FLAG="--no-llm"
        python3 $HOME/$VOICE_SCRIPT_NAME --daemon $AI_FLAG &>> $LOG_DIR/voice.log &
        VOICE_PID=$!
        PID_LIST="$PID_LIST $VOICE_PID"
        echo -e "  ${GREEN}  Voice 已重启 PID=$VOICE_PID${NC}"
    fi

    # 检查 WebSocket
    if [ -n "$WS_PID" ] && ! kill -0 $WS_PID 2>/dev/null; then
        echo -e "  ${RED}[$(date +%H:%M:%S)] WebSocket 崩溃, 重启中...${NC}"
        python3 $HOME/ws_server.py &>> $LOG_DIR/ws_server.log &
        WS_PID=$!
        PID_LIST="$PID_LIST $WS_PID"
        echo -e "  ${GREEN}  WebSocket 已重启 PID=$WS_PID${NC}"
    fi

    # 检查 body_angle_node
    if [ -n "$BODY_ANGLE_PID" ] && ! kill -0 $BODY_ANGLE_PID 2>/dev/null; then
        echo -e "  ${RED}[$(date +%H:%M:%S)] BodyAngle 崩溃, 重启中...${NC}"
        export PYTHONPATH="/opt/tros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
        python3 $HOME/body_angle_node.py &>> $LOG_DIR/body_angle.log &
        BODY_ANGLE_PID=$!
        PID_LIST="$PID_LIST $BODY_ANGLE_PID"
        echo -e "  ${GREEN}  BodyAngle 已重启 PID=$BODY_ANGLE_PID${NC}"
    fi

    sleep 5
done
