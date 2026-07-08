# 智能健康袖套 Smart Health Sleeve

> 语言 / Language：**简体中文** ｜ [English (README.md)](README.md)

基于 **RDK X5 + ESP32** 的可穿戴肌电康复训练系统。银织物干电极采集表面肌电（sEMG）信号，`TCN + MLP` 深度学习模型预测肌肉激活比例，地平线 **BPU** 加速推理（27 ms → 1.98 ms），微信小程序以 **3D 数字孪生** 实时可视化肌肉状态，并配套**离线中文语音助手**引导康复训练。

---

## 目录

- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [模型信息](#模型信息)
- [语音交互子系统](#语音交互子系统)
- [快速开始](#快速开始)
- [硬件引脚](#硬件引脚)
- [环境要求](#环境要求)
- [版本信息](#版本信息)
- [许可证](#许可证)

---

## 系统架构

```
┌─ 硬件层 ──────────────────────────────────────────────────────┐
│  5V 电源 → 自研扩展板 → ESP32 + 银织物干电极 → 肌电采集 → BLE 广播  │
│  RDK X5 + USB 摄像头 → 人体骨骼跟踪 → BPU 推理                     │
│  MX1508 电机驱动 ×8 → 1030 空心杯振动马达 → 姿势纠偏触觉反馈        │
└──────────────────────────────────────────────────────────────┘

┌─ 通信层 ──────────────────────────────────────────────────────┐
│  ESP32   ──BLE────────────→ 微信小程序                          │
│  RDK X5  ──WiFi WebSocket──→ 微信小程序（ws://<ip>:8765）        │
└──────────────────────────────────────────────────────────────┘

┌─ 应用层 ──────────────────────────────────────────────────────┐
│  微信小程序：3D 数字孪生 + RandomForest 轻量推理 + 训练管理        │
│  RDK X5   ：AnchorCalibTCN BPU 推理（27 ms → 1.98 ms）+ 语音助手   │
└──────────────────────────────────────────────────────────────┘
```

**双模型交叉验证**：小程序端跑轻量 RandomForest 做实时 3D 渲染；RDK X5 端跑更大的 TCN 模型并配合摄像头骨架追踪作为 Ground Truth，交叉校验「预测 EMG」与「BLE 实测 EMG」，实现精度修正。

---

## 项目结构

```
SmartSleeve/
├── README.md                          # 英文说明（默认）
├── README_cn.md                       # 本文件（中文说明）
│
├── 1_RDK_X5_System/                   # RDK X5 板端部署 + 语音助手
│   ├── start_all.sh                   # 一键启动脚本（骨架/EMG/验证/语音/WebSocket）
│   ├── ros2_emg_bridge.py             # EMG 推理桥接（ROS2 节点）
│   ├── ros2_emg_bridge_v2.py          # EMG 桥接 v2（ML 预测 + 校准加载）
│   ├── tcn_bpu_predictor.py           # BPU 推理封装（替代 onnxruntime）
│   ├── body_angle_node.py             # 人体骨骼角度检测（ROS2）
│   ├── emg_cross_validation_v2.py     # 交叉验证 + 肌肉代偿/电极质量检测
│   ├── udp_emg_receiver.py            # UDP EMG/心率接收器
│   ├── screen_server.py               # HTTP/WebSocket 数据服务
│   ├── ws_server.py                   # ROS2 → WebSocket 桥接
│   ├── emg_deploy.py                  # 独立推理脚本
│   ├── voice_demo_v7.py               # 语音助手 v7（VAD/按键 + 训练控制 + LLM）
│   ├── voice_demo_v6.py               # 语音助手 v6（历史版本）
│   ├── voice_agent_tts.py             # TTS 语音合成模块
│   ├── scripts/emg-system.service     # systemd 开机自启服务
│   ├── motion_scaler_63subj.pkl       # 运动特征 StandardScaler
│   ├── calib_scaler_63subj.pkl        # 校准向量 StandardScaler
│   ├── calibration_config_63subj.json # 校准配置
│   └── anchorcalib_tcn_bpu_v2.bin     # BPU 编译模型（bayes-e）
│
├── 2_EMG_Model/                       # 肌电预测模型
│   ├── training/                      # 训练脚本 + 权重
│   │   ├── model.py                   # 原始 PyTorch 模型（Conv1d）
│   │   ├── model_bpu.py               # BPU 原生模型（Conv2d）
│   │   ├── train.py                   # LOSO 交叉验证训练
│   │   ├── losses.py                  # 组合损失函数
│   │   ├── cache_data.py              # 数据缓存
│   │   ├── export_bpu_v2.py           # ONNX opset=11 导出
│   │   ├── export_bpu.py              # 原始 BPU ONNX 导出
│   │   ├── migrate_weights.py         # Conv1d → Conv2d 权重迁移
│   │   ├── adapt_zenodo.py            # Zenodo 数据集适配
│   │   ├── adapt_lucchetti.py         # Lucchetti 数据集适配
│   │   ├── anchorcalib_tcn.pt         # 10 人模型权重
│   │   ├── anchorcalib_tcn_63subj.pt  # 63 人模型权重
│   │   └── anchorcalib_tcn_bpu_v2.pt  # BPU 迁移后权重
│   ├── deployed/                      # 部署用模型
│   │   ├── anchorcalib_tcn_bpu_v2.onnx  # BPU-ready ONNX（8.6 MB，opset=11）
│   │   └── anchorcalib_tcn_bpu_v2.bin   # BPU 编译模型（2.5 MB）
│   └── report.html                    # BPU 部署精度/性能报告
│
├── 3_Quantization/                    # 量化分析工具
│   ├── quant_analysis.py              # INT8 量化精度分析
│   ├── operator_level_quant.py        # 底层算子级量化
│   ├── onnx_node_analysis.py          # ONNX 算子级分析
│   ├── compare_models.py              # 三模型对比
│   └── gen_figures.py                 # 图表生成
│
├── 4_ESP32_Firmware/                  # ESP32 固件
│   ├── PulseSensorAmped_Arduino_1dot2.ino  # 主程序（BLE + PWM + ADC）
│   ├── BLE_Manager.ino                # 蓝牙心率 + 肌电服务
│   ├── WiFi_Manager.ino               # WiFi 管理（预留）
│   ├── Interrupt.ino                  # 定时中断（脉搏检测 ISR）
│   └── ABX00083-datasheet.pdf         # Arduino Nano ESP32 数据手册
│
├── 5_Mini_Program/                    # 微信小程序
│   ├── app.js / app.json / app.wxss   # 入口 + 全局配置
│   ├── project.config.json            # 项目配置
│   ├── utils/
│   │   ├── dataManager.js             # 全局传感器数据单例（WebSocket + BLE）
│   │   ├── rfInference.js             # RandomForest JS 推理引擎
│   │   ├── roleManager.js             # 医生/患者角色管理
│   │   └── scorer.js                  # 训练评分
│   ├── pages/
│   │   ├── index/                     # 首页（蓝牙连接 + 实时监控）
│   │   ├── digitalTwin/               # 3D 数字孪生
│   │   ├── history/                   # 个人中心 / 健康档案
│   │   ├── roleSelect/                # 角色选择
│   │   ├── taskDetail/                # 训练任务详情
│   │   ├── taskPublish/               # 发布任务（医生）
│   │   ├── patientDetail/             # 患者详情（医生）
│   │   └── actionRecord/              # 标准动作录制（医生）
│   └── cloudfunctions/                # 云函数
│       ├── setUserRole/               # 用户角色
│       ├── joinDoctor/                # 患者邀请码绑定
│       ├── joinDoctorByQR/            # 患者扫码绑定
│       ├── manageTask/                # 任务 CRUD
│       ├── fetchMyTasks/              # 获取任务
│       ├── actionTemplates/           # 动作模板
│       ├── getDoctorPatients/         # 医生患者列表
│       └── getAiReport/               # AI 康复报告
│
├── 6_PCB/                             # 硬件设计
│   └── Altium_SmartSleeve1.zip        # 扩展板 PCB（Altium）
│
└── 7_3D_Models/                       # 3D 手臂模型
    ├── arm_model.glb                  # 完整模型（2.8 MB）
    ├── arm_model_clean.glb            # 清理版（2.7 MB）
    ├── arm_model_r108.glb             # Three.js r108 兼容版（1.5 MB）
    ├── arm_model_compressed.glb       # 压缩版（0.6 MB）
    ├── arm_skinned.glb                # 骨骼蒙皮版（4.3 MB）
    ├── GLTFLoader_raw.js              # GLTF 加载器源码
    └── add_armature.py                # Blender 骨架导出脚本
```

---

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | AnchorCalibTCN（TCN + MLP Fusion） |
| 参数量 | 2,193,730 |
| 输入 | `[1, 26, 1, 64]`（motion 10ch + calib 16ch） |
| 输出 | `[biceps_ratio, triceps_ratio]`（二头肌 / 三头肌激活比例） |
| ONNX opset | 11 |
| BPU 架构 | bayes-e（RDK X5） |

### 推理性能

| 平台 | 延迟 | FPS | 模型大小 |
|------|------|-----|----------|
| RDK X5 CPU（onnxruntime） | 27.2 ms | 37 | 8.6 MB |
| RDK X5 BPU | **1.98 ms** | **499** | 2.5 MB |

### INT8 量化精度

| 指标 | 值 |
|------|-----|
| 余弦相似度 | 0.9935 |
| R² | 0.9298 |
| Pearson r | 0.9648 |
| MAE | 0.2105 |
| RMSE | 0.3356 |

---

## 语音交互子系统

RDK X5 视觉端内置一套**本地化中文语音助手**（`1_RDK_X5_System/voice_demo_v7.py`），围绕肘关节屈伸康复训练构建「听 → 理解 → 说」闭环，全程可离线运行。

```
杰理无线麦克风 → faster-whisper（离线中文 ASR）→ 意图三级瀑布 → PriorityTTS 语音播报
      ↑                                              ↓
   16 kHz 采集                        关键词匹配 → DeepSeek LLM → 原文回显
                                              ↓
   ROS2 订阅 /body_arm_angles、/virtual_emg、/emg_validation、/emg_alerts、/raw_emg
```

| 环节 | 实现 |
|------|------|
| 语音识别 | faster-whisper（优先 tiny，回退 base，int8 量化），支持**按键触发**与 **VAD 自动/守护**两种模式 |
| 意图理解 | 三级瀑布：关键词匹配 → DeepSeek（`deepseek-chat`）多轮对话 → 原文回显；网络不可用时静默降级 |
| 语音合成 | `PriorityTTSSpeaker`：在线 edge-tts（`zh-CN-XiaoxiaoNeural`）+ 离线 espeak-ng 回退，WAV 缓存/预热、四级优先级队列，高优先级告警可打断当前播报 |
| 状态感知 | `SystemState` 线程安全汇聚角度/EMG/验证/告警/心率，含训练状态机（开始/暂停/恢复/结束、带防抖自动计次）与语音校准流程 |
| 后台巡检 | `TrainingMonitor` 每 5 s 检查：电极脱落/肌肉代偿/动作告警、心率高低预警、定期进度播报、分步提示、无人声分级提醒 |

**支持的语音指令**：开始训练 / 暂停训练 / 继续训练 / 结束训练 / 训练报告 / AI 报告 / 诊断 / 检查状态 / 当前角度 / 肌肉状态 / 开始校准 / 音量调节（大声一点、小声一点、音量调到百分之五十）/ 帮助。

启动方式：

```bash
python3 voice_demo_v7.py --daemon    # 守护模式（VAD 自动触发，推荐）
python3 voice_demo_v7.py --vad       # 语音活动检测模式
python3 voice_demo_v7.py             # 按键模式（调试用）
python3 voice_demo_v7.py --no-llm    # 仅关键词，禁用 AI 对话
```

DeepSeek API Key 配置（可选，用于 AI 对话与康复总结）：

```bash
# 获取 Key: https://platform.deepseek.com/api_keys
echo 'sk-your-api-key' > ~/.deepseek_key
```

---

## 快速开始

### 1. RDK X5 板端部署

```bash
scp 1_RDK_X5_System/* root@<rdk-ip>:/root/
cp /root/scripts/emg-system.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable emg-system
/root/start_all.sh          # 一键启动骨架追踪 / EMG 桥接 / 交叉验证 / 语音 / WebSocket
```

启动后验证：

```bash
ros2 topic list                 # 查看话题
ss -lntp | grep 8765            # 验证 WebSocket 端口
```

### 2. 模型训练（GPU 服务器）

```bash
cd 2_EMG_Model/training
python train.py --data_dir ../data/ --output_dir ./results/
```

### 3. BPU 编译（需地平线 J5 OpenExplorer 工具链）

```bash
hb_mapper makertbin --model-type onnx --fast-perf \
  --model anchorcalib_tcn_bpu_v2.onnx \
  --march bayes-e -i merged_input 1x26x1x64
```

### 4. ESP32 固件烧录

1. Arduino IDE 打开 `4_ESP32_Firmware/PulseSensorAmped_Arduino_1dot2.ino`
2. 开发板选择 `Arduino Nano ESP32 (ABX00083)`
3. 接线：脉搏传感器 → A0，EMG#1 → A0/A1（详见硬件引脚）
4. 烧录后串口监视器（115200）应看到启动日志

### 5. 微信小程序

用微信开发者工具打开 `5_Mini_Program/` 目录，**工具 → 构建 npm** 后编译运行。首次进入数字孪生页会自动从云存储下载 AI 模型（约 6 MB）并缓存。

---

## 硬件引脚

| 组件 | 连接 |
|------|------|
| 电源 | 5V DC 接入自研扩展板，板载升压/分压电路供电 |
| PPG 脉搏传感器 | GND / SIG=A0 / 5V（扩展板引出） |
| EMG 电极（银织物） | 3.5mm TRS → AD8221 + TL074 仪表放大电路（±9V 供电，调理至 0–3.3V） |
| MX1508 电机驱动 | GPIO 控制，8 通道（扩展板集成，1kHz PWM） |
| RDK X5 | USB 连接摄像头，串口通信 |

---

## 环境要求

| 组件 | 要求 |
|------|------|
| RDK X5 | Ubuntu 22.04 ARM64，hobot-dnn ≥ 3.0，ROS2 Humble |
| 训练服务器 | Python 3.10+，PyTorch 2.x，CUDA 12+ |
| BPU 编译 | Ubuntu 20.04，Python 3.8，地平线 J5 OE v1.1.77+ |
| ESP32 | Arduino Nano ESP32（ABX00083），PulseSensor Playground |
| 微信小程序 | 微信开发者工具，threejs-miniprogram v0.0.8（Three.js r108）、pako v2.1.0 |
| 语音助手 | alsa-utils、espeak-ng、ffmpeg、edge-tts、faster-whisper |

---

## 版本信息

- 模型：AnchorCalibTCN 63-subject
- BPU 工具链：hbdk 3.49.13 / hb_mapper 1.24.1
- RDK X5 Runtime：hobot-dnn 3.0.4 / hbrt 3.15.55.0
- ESP32：Arduino Nano ESP32

---

## 许可证

本项目仅限学术研究和教育用途。硬件设计文件遵循 **CC BY-NC-SA 4.0**。
