# 智能健康袖套 Smart Health Sleeve

> 语言 / Language：**简体中文** ｜ [English (README.md)](README.md)

基于 **RDK X5 + ESP32** 的可穿戴肌电康复训练系统。银织物干电极采集表面肌电（sEMG）信号，`TCN + MLP` 深度学习模型预测肌肉激活比例，地平线 **BPU** 加速推理（27 ms → 1.98 ms），微信小程序以 **3D 数字孪生** 实时可视化肌肉状态，并配套**离线中文语音助手**引导康复训练。

---

## 目录

- [系统架构](#系统架构)
- [双模型交叉验证](#双模型交叉验证)
- [模型信息](#模型信息)
- [目录与文件详解](#目录与文件详解)
  - [根目录](#根目录)
  - [1_RDK_X5_System — RDK X5 板端部署 + 语音助手](#1_rdk_x5_system--rdk-x5-板端部署--语音助手)
  - [2_EMG_Model — 肌电预测模型](#2_emg_model--肌电预测模型)
  - [3_Quantization — 量化分析工具](#3_quantization--量化分析工具)
  - [4_ESP32_Firmware — ESP32 固件](#4_esp32_firmware--esp32-固件)
  - [5_Mini_Program — 微信小程序](#5_mini_program--微信小程序)
  - [6_PCB — 硬件设计](#6_pcb--硬件设计)
  - [7_3D_Models — 3D 手臂模型](#7_3d_models--3d-手臂模型)
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

## 双模型交叉验证

小程序端跑轻量 RandomForest 做实时 3D 渲染；RDK X5 端跑更大的 TCN 模型并配合摄像头骨架追踪作为 Ground Truth，交叉校验「预测 EMG」与「BLE 实测 EMG」，实现精度修正。

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

## 目录与文件详解

> 下表覆盖仓库内**每一个文件**。标注「空」的文件为 0 字节占位，功能尚未提交。

### 根目录

| 文件 | 大小 | 说明 |
|------|------|------|
| `README.md` | 8.7 KB | 英文说明（默认首页） |
| `README_cn.md` | — | 本文件，中文详细说明 |
| `.gitignore` | 389 B | Git 忽略规则（模型缓存、临时文件等） |

### 1_RDK_X5_System — RDK X5 板端部署 + 语音助手

板端所有运行时脚本、ROS2 节点、推理封装、语音助手与模型资源。

| 文件 | 大小 | 说明 |
|------|------|------|
| `start_all.sh` | 16 KB | 一键启动脚本：拉起骨架追踪 / EMG 桥接 / 交叉验证 / 语音助手 / WebSocket 全部节点，退出时自动清理子进程 |
| `body_angle_node.py` | 15 KB | 人体骨骼角度检测 ROS2 节点，摄像头 → AI 人体检测 → 发布 `/body_arm_angles` |
| `ros2_emg_bridge.py` | 14 KB | EMG 推理桥接 ROS2 节点（初版），角度 → TCN → 发布 `/virtual_emg` |
| `ros2_emg_bridge_v2.py` | 15 KB | EMG 桥接 v2，支持 `--ml_predict` 与 `--load_calib` 校准加载 |
| `tcn_bpu_predictor.py` | 6 KB | BPU 推理封装，替代 onnxruntime，加载 `.bin` 模型做硬件加速推理 |
| `emg_cross_validation_v2.py` | 26 KB | 交叉验证：TCN 预测 vs 真实 EMG，附肌肉代偿 / 电极质量检测，发布 `/emg_validation` + `/emg_alerts` |
| `emg_deploy.py` | 4.5 KB | 独立推理脚本（脱离 ROS2 单跑，用于快速验证模型） |
| `udp_emg_receiver.py` | **空** | UDP EMG / 心率接收器（0 字节占位，待实现） |
| `screen_server.py` | 19 KB | 板载屏幕 HTTP/WebSocket 数据服务（本地可视化 UI） |
| `ws_server.py` | 7 KB | ROS2 → WebSocket 桥接，对外暴露 `ws://0.0.0.0:8765` 供小程序连接 |
| `voice_demo_v7.py` | 82 KB | **语音助手 v7**：VAD/按键双模式、Whisper 离线识别、关键词+LLM 意图、优先级 TTS、训练状态机、后台告警巡检（详见[语音交互子系统](#语音交互子系统)） |
| `voice_demo_v6.py` | 38 KB | 语音助手 v6（历史版本，保留参考） |
| `voice_agent_tts.py` | **空** | TTS 语音合成模块（0 字节占位，实际 TTS 逻辑已内联在 v7 中） |
| `scripts/emg-system.service` | 987 B | systemd 开机自启服务单元 |
| `motion_scaler_63subj.pkl` | 823 B | 运动特征 StandardScaler（63 人数据集拟合） |
| `calib_scaler_63subj.pkl` | 967 B | 校准向量 StandardScaler |
| `calibration_config_63subj.json` | 730 B | 校准配置（默认校准向量、通道定义等） |
| `anchorcalib_tcn_bpu_v2.bin` | 2.4 MB | BPU 编译后模型（bayes-e 架构，板端实际加载） |

### 2_EMG_Model — 肌电预测模型

深度学习模型的训练脚本、权重与部署产物。

**`training/` — 训练脚本 + 权重**

| 文件 | 大小 | 说明 |
|------|------|------|
| `model.py` | 11 KB | 原始 PyTorch 模型定义（Conv1d 版 TCN + MLP 融合） |
| `model_bpu.py` | 8.5 KB | BPU 原生模型（Conv2d 版，适配地平线算子约束） |
| `train.py` | 47 KB | LOSO（Leave-One-Subject-Out）交叉验证训练主脚本 |
| `losses.py` | 2.3 KB | 组合损失函数（MSE + 相关性 + 平滑项等） |
| `cache_data.py` | 535 B | 数据预处理与缓存 |
| `export_bpu.py` | 15 KB | 原始 BPU ONNX 导出 |
| `export_bpu_v2.py` | 6.4 KB | ONNX opset=11 导出（当前部署使用） |
| `migrate_weights.py` | 7.6 KB | Conv1d → Conv2d 权重迁移工具 |
| `adapt_zenodo.py` | 4.8 KB | Zenodo 公开数据集适配 |
| `adapt_lucchetti.py` | 5.7 KB | Lucchetti 数据集适配 |
| `anchorcalib_tcn.pt` | 8.4 MB | 10 人模型权重 |
| `anchorcalib_tcn_63subj.pt` | 8.5 MB | 63 人模型权重（主力） |
| `anchorcalib_tcn_bpu_v2.pt` | 8.4 MB | BPU 迁移后 Conv2d 权重 |
| `motion_scaler_63subj.pkl` | 823 B | 运动特征标准化（与板端一致） |
| `calib_scaler_63subj.pkl` | 967 B | 校准向量标准化（与板端一致） |

**`deployed/` — 部署产物**

| 文件 | 大小 | 说明 |
|------|------|------|
| `anchorcalib_tcn_bpu_v2.onnx` | 8.4 MB | BPU-ready ONNX（opset=11，可用 onnxruntime 直跑） |
| `anchorcalib_tcn_bpu_v2.bin` | 2.4 MB | BPU 编译模型（`hb_mapper` 产物） |

**其他**

| 文件 | 大小 | 说明 |
|------|------|------|
| `report.html` | 6.9 KB | BPU 部署精度 / 性能报告（含量化对比图表） |

### 3_Quantization — 量化分析工具

INT8 量化的精度评估与算子级分析。

| 文件 | 大小 | 说明 |
|------|------|------|
| `quant_analysis.py` | 13 KB | INT8 量化整体精度分析（余弦相似度 / R² / MAE 等） |
| `operator_level_quant.py` | 20 KB | 底层算子级量化实验 |
| `onnx_node_analysis.py` | 20 KB | ONNX 逐算子（node）级误差分析 |
| `compare_models.py` | 21 KB | FP32 / ONNX / BPU 三模型输出对比 |
| `gen_figures.py` | 37 KB | 论文/报告图表批量生成 |

### 4_ESP32_Firmware — ESP32 固件

Arduino Nano ESP32（ABX00083）固件，负责 BLE 广播、PWM 电机驱动与 ADC 采集。

| 文件 | 大小 | 说明 |
|------|------|------|
| `PulseSensorAmped_Arduino_1dot2.ino` | 6.8 KB | 主程序：脉搏 ADC 采集 + PWM 电机 + 主循环 |
| `BLE_Manager.ino` | 5.6 KB | BLE 心率服务 + 自定义肌电服务，NOTIFY 上报 |
| `WiFi_Manager.ino` | 2.6 KB | WiFi 管理（预留，当前主用 BLE） |
| `Interrupt.ino` | 5.4 KB | 定时器 ISR，脉搏波峰检测与 IBI 计算 |
| `ABX00083-datasheet.pdf` | 3.3 MB | Arduino Nano ESP32 官方数据手册 |

### 5_Mini_Program — 微信小程序

原生微信小程序 + Three.js（r108），含患者端与医生端双角色。

**入口 / 全局配置**

| 文件 | 大小 | 说明 |
|------|------|------|
| `app.js` | 4.2 KB | 小程序入口，`globalData` 与云开发初始化 |
| `app.json` | 1.4 KB | 全局页面路由、tabBar、窗口配置 |
| `app.wxss` | 1.4 KB | 全局样式 |
| `project.config.json` | 988 B | 开发者工具项目配置 |
| `sitemap.json` | 191 B | 微信索引配置 |
| `package.json` | 238 B | npm 依赖声明（threejs-miniprogram、pako） |
| `package-lock.json` | 915 B | 依赖锁定 |

**`utils/` — 工具函数**

| 文件 | 大小 | 说明 |
|------|------|------|
| `dataManager.js` | 11 KB | 全局传感器数据单例：WebSocket 连接、共享状态、订阅者模式、`runInference()` 编排 |
| `rfInference.js` | 4.7 KB | RandomForest 二进制格式解析，`predict(features)` → `[biceps, triceps]` |
| `roleManager.js` | 4.3 KB | 医生 / 患者角色管理 |
| `scorer.js` | 7.9 KB | 训练动作评分算法 |

**`pages/` — 页面**（每个页面含 `.js / .json / .wxml / .wxss` 四件套）

| 页面 | 主脚本大小 | 说明 |
|------|------|------|
| `index/` | 40 KB | 首页：蓝牙连接 ESP32、实时心率/肌电、2D 手臂动画、AI 诊断、RDK WebSocket 初始化 |
| `digitalTwin/` | 35 KB | 3D 数字孪生页（详见下表） |
| `history/` | 13 KB | 个人中心 / 健康档案，云数据库读写历史训练记录 |
| `roleSelect/` | 1.3 KB | 首次进入选择医生 / 患者角色 |
| `taskDetail/` | 15 KB | 训练任务详情与执行 |
| `taskPublish/` | 5.6 KB | 医生端：发布训练任务 |
| `patientDetail/` | 3.5 KB | 医生端：查看单个患者详情 |
| `actionRecord/` | 6 KB | 医生端：录制标准动作模板 |

**`pages/digitalTwin/` — 3D 数字孪生细分**

| 文件 | 大小 | 说明 |
|------|------|------|
| `digitalTwin.js` | 35 KB | 页面逻辑：模型三级下载/缓存、渲染循环、ML 推理编排 |
| `gltfLoader.js` | 80 KB | GLTF/GLB 模型加载器（适配小程序环境） |
| `armModel.js` | 11 KB | 程序化 3D 手臂骨骼层级（shoulderPivot → 上臂 → elbowPivot → 前臂 → wristPivot） |
| `muscleMaterials.js` | 4 KB | EMG → 颜色映射（绿→黄→橙→红） |
| `orbitControls.js` | 5.7 KB | 触摸旋转/缩放控制 + `raycastMuscles()` 点击肌肉检测 |
| `threeAdapter.js` | 2 KB | WebGL renderer 工厂，封装 threejs-miniprogram |

**`components/navigation-bar/` — 自定义导航栏组件**

| 文件 | 大小 | 说明 |
|------|------|------|
| `navigation-bar.js / .json / .wxml / .wxss` | — | 自定义顶部导航栏组件 |

**`cloudfunctions/` — 云函数**（每个含 `config.json / index.js / package.json`）

| 云函数 | 说明 |
|--------|------|
| `setUserRole/` | 设置用户角色（医生 / 患者） |
| `joinDoctor/` | 患者通过邀请码绑定医生 |
| `joinDoctorByQR/` | 患者扫二维码绑定医生 |
| `manageTask/` | 训练任务增删改查（CRUD） |
| `fetchMyTasks/` | 拉取当前用户任务列表 |
| `actionTemplates/` | 标准动作模板读写 |
| `getDoctorPatients/` | 医生查询名下患者列表 |
| `getAiReport/` | 生成 AI 康复报告（调用大模型） |

**`images/` — tab 图标**

| 文件 | 说明 |
|------|------|
| `tab-monitor.png` / `tab-monitor-active.png` | 监控 tab 图标（常态 / 选中） |
| `tab-twin.png` / `tab-twin-active.png` | 数字孪生 tab 图标 |
| `tab-user.png` / `tab-user-active.png` | 个人中心 tab 图标 |

**`miniprogram_npm/` — 构建后的 npm 包**

| 文件 | 大小 | 说明 |
|------|------|------|
| `threejs-miniprogram/index.js` | 583 KB | Three.js r108 小程序版 |
| `pako/index.js` | 224 KB | gzip 解压库（解压 `model.bin.gz`） |
| `pako/index.js.map` | 265 KB | pako sourcemap |

### 6_PCB — 硬件设计

| 文件 | 大小 | 说明 |
|------|------|------|
| `Altium_SmartSleeve1.zip` | 1.1 MB | 自研扩展板 PCB 工程（Altium Designer，含原理图与 PCB 布局） |

### 7_3D_Models — 3D 手臂模型

Z-Anatomy 解剖模型的多档位版本 + Blender 脚本。

| 文件 | 大小 | 说明 |
|------|------|------|
| `arm_skinned.glb` | 4.3 MB | 骨骼蒙皮版（最完整） |
| `arm_model.glb` | 2.8 MB | 完整模型 |
| `arm_model_clean.glb` | 2.7 MB | 清理版（去除冗余节点） |
| `arm_model_r108.glb` | 1.5 MB | Three.js r108 兼容版 |
| `arm_model_compressed.glb` | 0.6 MB | 压缩版（小程序实际加载，兼顾体积） |
| `GLTFLoader_raw.js` | 80 KB | GLTF 加载器源码（参考） |
| `add_armature.py` | 6 KB | Blender 骨架绑定 / 导出脚本 |

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

> 注：AI 对话依赖 `voice_llm.py`（DeepSeek 客户端），该文件当前未随仓库提交；未提供时 v7 自动降级为「关键词 + 原文回显」模式，其余功能不受影响。

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
3. 接线：脉搏传感器 → A0，EMG 电极 → 仪表放大电路 → ADC
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
