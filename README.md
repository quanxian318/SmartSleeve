# 智能健康袖套 Smart Health Sleeve

基于 RDK X5 + ESP32 的可穿戴肌电康复训练系统。银织物干电极采集肌电信号，TCN+MLP 深度学习模型预测肌肉激活，BPU 加速推理，微信小程序 3D 数字孪生实时可视化。

---

## 项目结构

```
Smart_Sleeve/
├── README.md                          # 本文件
│
├── 1_RDK_X5_System/                   # RDK X5 板端部署
│   ├── start_all.sh                   # 一键启动脚本
│   ├── ros2_emg_bridge.py             # EMG 推理桥接 (ROS2 节点)
│   ├── tcn_bpu_predictor.py           # BPU 推理封装 (替代 onnxruntime)
│   ├── body_angle_node.py             # 人体骨骼角度检测 (ROS2)
│   ├── screen_server.py               # HTTP/WebSocket 数据服务
│   ├── emg_deploy.py                  # 独立推理脚本
│   ├── scripts/
│   │   └── emg-system.service         # systemd 自启动服务
│   ├── motion_scaler_63subj.pkl       # 运动特征 StandardScaler
│   ├── calib_scaler_63subj.pkl        # 校准向量 StandardScaler
│   ├── calibration_config_63subj.json # 校准配置
│   └── anchorcalib_tcn_bpu_v2.bin     # BPU 编译模型 (bayes-e)
│
├── 2_EMG_Model/                       # 肌电预测模型
│   ├── training/                      # 训练脚本 + 权重
│   │   ├── model.py                   # 原始 PyTorch 模型 (Conv1d)
│   │   ├── model_bpu.py               # BPU 原生模型 (Conv2d)
│   │   ├── train.py                   # LOSO 训练脚本
│   │   ├── losses.py                  # 组合损失函数
│   │   ├── cache_data.py              # 数据缓存
│   │   ├── export_bpu_v2.py           # ONNX opset=11 导出
│   │   ├── export_bpu.py              # 原始 BPU ONNX 导出
│   │   ├── migrate_weights.py         # Conv1d->Conv2d 权重迁移
│   │   ├── adapt_zenodo.py            # Zenodo 数据适配
│   │   ├── adapt_lucchetti.py         # Lucchetti 数据适配
│   │   ├── anchorcalib_tcn.pt         # 10人模型权重
│   │   ├── anchorcalib_tcn_63subj.pt  # 63人模型权重
│   │   ├── anchorcalib_tcn_bpu_v2.pt  # BPU迁移后权重
│   │   ├── motion_scaler_63subj.pkl   # 运动特征标准化
│   │   └── calib_scaler_63subj.pkl    # 校准向量标准化
│   ├── deployed/                      # 部署用模型文件
│   │   ├── anchorcalib_tcn_bpu_v2.onnx # BPU-ready ONNX (8.6 MB, opset=11)
│   │   └── anchorcalib_tcn_bpu_v2.bin  # BPU 编译模型 (2.5 MB)
│   └── report.html                    # BPU 部署精度/性能报告
│
├── 3_Quantization/                    # 量化分析工具
│   ├── quant_analysis.py              # INT8 量化精度分析
│   ├── quant_node_analysis.py         # 逐节点量化分析
│   ├── onnx_node_analysis.py          # ONNX 算子级分析
│   ├── operator_level_quant.py        # 底层算子量化
│   ├── compare_models.py              # 三模型对比
│   └── gen_figures.py                 # 图表生成
│
├── 4_ESP32_Firmware/                  # ESP32 固件
│   ├── PulseSensorAmped_Arduino_1dot2.ino  # 主程序
│   ├── BLE_Manager.ino                # 蓝牙管理
│   ├── WiFi_Manager.ino               # WiFi 管理
│   ├── Interrupt.ino                  # 定时中断
│   └── ABX00083-datasheet.pdf         # Arduino Nano ESP32 数据手册
│
├── 5_Mini_Program/                    # 微信小程序
│   ├── app.js / app.json / app.wxss   # 入口 + 全局配置
│   ├── project.config.json            # 项目配置
│   ├── utils/                         # 工具函数
│   │   ├── dataManager.js             # 全局传感器数据管理
│   │   ├── rfInference.js             # RandomForest 推理引擎
│   │   ├── roleManager.js             # 角色管理
│   │   └── scorer.js                  # 训练评分
│   ├── pages/                         # 页面
│   │   ├── index/                     # 首页 (蓝牙连接 + 监控)
│   │   ├── digitalTwin/               # 3D 数字孪生
│   │   ├── history/                   # 个人中心
│   │   ├── roleSelect/                # 角色选择
│   │   ├── taskDetail/                # 训练任务
│   │   ├── taskPublish/               # 发布任务 (医生)
│   │   ├── patientDetail/             # 患者详情 (医生)
│   │   └── actionRecord/              # 标准动作录制 (医生)
│   └── cloudfunctions/                # 云函数
│       ├── setUserRole/               # 用户角色
│       ├── joinDoctor/                # 患者邀请码绑定
│       ├── joinDoctorByQR/            # 患者扫码绑定
│       ├── manageTask/                # 任务 CRUD
│       ├── fetchMyTasks/              # 获取任务
│       ├── actionTemplates/           # 动作模板
│       ├── getDoctorPatients/         # 医生患者列表
│       └── getAiReport/               # AI康复报告
│

├── 7_3D_Models/                       # 3D 手臂模型
│   ├── arm_model.glb                  # 完整模型 (2.8 MB)
│   ├── arm_model_clean.glb            # 清理版 (2.7 MB)
│   ├── arm_model_r108.glb             # Three.js r108 兼容版 (1.5 MB)
│   ├── arm_model_compressed.glb       # 压缩版 (0.6 MB)
│   ├── arm_skinned.glb                # 骨骼蒙皮版 (4.3 MB)
│   ├── GLTFLoader_raw.js              # GLTF 加载器源码
│   └── add_armature.py                # Blender 骨架导出脚本
│
└── 6_PCB/                             # 硬件设计
    └── Altium_SmartSleeve1.zip        # 扩展板 PCB (Altium)
```

---

## 系统架构

硬件层: 5V电源 -> 自研扩展板 -> ESP32 + 银织物电极 -> 肌电采集 -> BLE 广播
        RDK X5 + USB摄像头 -> 骨骼跟踪 -> BPU 推理
        MX1508 电机驱动 x8 -> 振动触觉反馈

通信层: BLE (ESP32) -> 微信小程序
        WiFi WebSocket (RDK X5) -> 微信小程序

应用层: 微信小程序 (3D数字孪生 + RF模型 + 训练管理)
        RDK X5 (TCN+MLP BPU推理, 27ms->1.98ms)

---

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | AnchorCalibTCN (TCN + MLP Fusion) |
| 参数量 | 2,193,730 |
| 输入 | [1, 26, 1, 64] (motion 10ch + calib 16ch) |
| 输出 | [biceps_ratio, triceps_ratio] |
| ONNX opset | 11 |
| BPU 架构 | bayes-e (RDK X5) |

### 推理性能

| 平台 | 延迟 | FPS | 模型大小 |
|------|------|-----|----------|
| RDK X5 CPU (onnxruntime) | 27.2 ms | 37 | 8.6 MB |
| RDK X5 BPU | 1.98 ms | 499 | 2.5 MB |

### INT8 量化精度

| 指标 | 值 |
|------|-----|
| 余弦相似度 | 0.9935 |
| R-squared | 0.9298 |
| Pearson r | 0.9648 |
| MAE | 0.2105 |
| RMSE | 0.3356 |

---

## 快速开始

### RDK X5 板端部署

scp 1_RDK_X5_System/* root@(rdk-ip):/root/
cp /root/scripts/emg-system.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable emg-system
/root/start_all.sh

### 模型训练 (GPU 服务器)

cd 2_EMG_Model/training
python train.py --data_dir ../data/ --output_dir ./results/

### BPU 编译 (需要地平线 J5 OpenExplorer 工具链)

hb_mapper makertbin --model-type onnx --fast-perf --model anchorcalib_tcn_bpu_v2.onnx --march bayes-e -i merged_input 1x26x1x64

### 微信小程序

用微信开发者工具打开 5_Mini_Program/ 目录，构建 npm 后运行。

---

## 版本信息

- 模型: AnchorCalibTCN 63-subject
- BPU 工具链: hbdk 3.49.13 / hb_mapper 1.24.1
- RDK X5 Runtime: hobot-dnn 3.0.4 / hbrt 3.15.55.0
- ESP32: Arduino Nano ESP32

---

## 硬件引脚

| 组件 | 连接 |
|------|------|
| 电源 | 5V DC 接入自研扩展板，板载升压/分压电路供电 |
| PPG 脉搏传感器 | GND, SIG=A0, 5V (扩展板引出) |
| EMG 电极 (银织物) | 3.5mm TRS → AD8221+TL074 仪表放大电路 |
| MX1508 电机驱动 | GPIO 控制, 8 通道 (扩展板集成) |
| RDK X5 | USB 连接摄像头, 串口通信 |

---

## 环境要求

| 组件 | 要求 |
|------|------|
| RDK X5 | Ubuntu 22.04 ARM64, hobot-dnn >= 3.0, ROS2 Humble |
| 训练服务器 | Python 3.10+, PyTorch 2.x, CUDA 12+ |
| BPU 编译 | Ubuntu 20.04, Python 3.8, 地平线 J5 OE v1.1.77+ |
| ESP32 | Arduino Nano ESP32, PulseSensor Playground |
| 微信小程序 | 微信开发者工具, threejs-miniprogram v0.0.8 |

---

## 许可证

本项目仅限学术研究和教育用途。硬件设计文件遵循 CC BY-NC-SA 4.0。
