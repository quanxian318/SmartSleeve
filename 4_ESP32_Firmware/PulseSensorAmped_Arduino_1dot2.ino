/*
 * 智能健康袖套 - Arduino Nano ESP32 ABX00083
 * 脉搏传感器 -> A0 (PPG心率)
 * EMG传感器 -> A6 (肱二头肌), A7 (肱三头肌)
 * BLE心率/EMG服务 -> 微信小程序
 * WiFi UDP -> RDK X5 (EMG实时数据)
 */

// ===================== 配置 =====================
#define ENABLE_BLE      1     // 启用蓝牙BLE
#define ENABLE_WIFI     1     // 启用WiFi，向RDK X5发送EMG数据
#define DEVICE_NAME     "ECG-Monitor"

// WiFi 配置 (STA模式，连接到与RDK X5相同的路由器)
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASS       "YOUR_WIFI_PASSWORD"

// RDK X5 目标地址
#define RDK_X5_IP       "192.168.43.9"       // RDK X5的IP地址
#define RDK_X5_PORT     8766                 // UDP接收端口

// 测试模式: 1=发送固定常数(测试通信链路), 0=读取真实A6/A7 EMG
#define EMG_TEST_MODE   0

// ===================== 库引用 =====================
#if ENABLE_BLE
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#endif

#if ENABLE_WIFI
#include <WiFi.h>
#include <WiFiUdp.h>
#endif

// ===================== 传感器引脚 =====================
extern hw_timer_t * timer;        // 定义在 Interrupt.ino 中
int pulsePin = A0;                // 脉搏传感器接到A0
int emgPinA6 = A6;                // 肱二头肌 EMG (biceps)
int emgPinA7 = A7;                // 肱三头肌 EMG (triceps)
int blinkPin = 13;                // 板载LED
int fadePin = 5;                  // PWM呼吸灯
int fadeRate = 0;

volatile int BPM;
volatile int Signal;
volatile int EMG_A6_Signal;       // 肱二头肌原始信号
volatile int EMG_A7_Signal;       // 肱三头肌原始信号
volatile int EMG_A6_Smooth = 0;   // 平滑后的肱二头肌值
volatile int EMG_A7_Smooth = 0;   // 平滑后的肱三头肌值
volatile int IBI = 600;
volatile boolean Pulse = false;
volatile boolean QS = false;

// ===================== BLE变量 =====================
#if ENABLE_BLE
BLEServer* pServer = NULL;
BLECharacteristic* pHRMCharacteristic = NULL;
BLECharacteristic* pSignalCharacteristic = NULL;
BLECharacteristic* pIBICharacteristic = NULL;
BLECharacteristic* pEMG_BicepsCharacteristic = NULL;    // A6 肱二头肌
BLECharacteristic* pEMG_TricepsCharacteristic = NULL;   // A7 肱三头肌
bool deviceConnected = false;
bool oldDeviceConnected = false;

#define SERVICE_UUID_HEART_RATE    "0000180d-0000-1000-8000-00805f9b34fb"
#define CHARACTERISTIC_UUID_HRM   "00002a37-0000-1000-8000-00805f9b34fb"
#define CHARACTERISTIC_UUID_BSL   "00002a38-0000-1000-8000-00805f9b34fb"
#define SERVICE_UUID_CUSTOM       "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID_SIGNAL "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHARACTERISTIC_UUID_IBI   "beb5483e-36e1-4688-b7f5-ea07361b26a9"
#define CHARACTERISTIC_UUID_EMG_BICEPS   "beb5483e-36e1-4688-b7f5-ea07361b26aa"  // A6 肱二头肌
#define CHARACTERISTIC_UUID_EMG_TRICEPS  "beb5483e-36e1-4688-b7f5-ea07361b26ab"  // A7 肱三头肌
#endif

// ===================== WiFi变量 =====================
#if ENABLE_WIFI
WiFiUDP udpClient;
unsigned long lastWiFiCheck = 0;
bool wifiConnected = false;
#endif

// ===================== 初始化 =====================
void setup(){
  pinMode(blinkPin, OUTPUT);
  pinMode(fadePin, OUTPUT);
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Smart Sleeve Monitor Starting ===");

  // ADC配置: 12位高分辨率 + A6/A7低衰减(更灵敏)
  analogReadResolution(12);               // 12位 (0-4095), 比10位细腻4倍
  analogSetPinAttenuation(A6, ADC_0db);   // 0dB衰减, 量程~1.1V, 灵敏度最高
  analogSetPinAttenuation(A7, ADC_0db);
  // A0(脉搏)保持默认11dB, 量程~3.9V

  interruptSetup();

#if ENABLE_BLE
  initBLE();
#endif

#if ENABLE_WIFI
  initWiFi();
#endif

  Serial.println("Ready. Waiting for heartbeat...");
}

// ===================== 主循环 =====================
void loop(){
#if ENABLE_BLE
  // BLE连接状态变化
  if (deviceConnected && !oldDeviceConnected) {
    Serial.println("BLE: Connected!");
    oldDeviceConnected = deviceConnected;
  }
  if (!deviceConnected && oldDeviceConnected) {
    Serial.println("BLE: Disconnected");
    oldDeviceConnected = deviceConnected;
    delay(500);
    pServer->startAdvertising();
  }
#endif

  // 心跳检测：发送心率数据
  if (QS == true){
    fadeRate = 255;

#if ENABLE_BLE
    if (deviceConnected) {
      updateBLEHeartRate(BPM, Signal, IBI);
    }
#endif

    QS = false;
  }

  // 肌电数据：每20ms输出一次（50Hz）
  static unsigned long lastEMGTime = 0;
  if (millis() - lastEMGTime >= 20) {
    lastEMGTime = millis();

    int a0, a6, a7;

#if EMG_TEST_MODE
    // ============ 测试模式: 发送固定常数 ============
    // 方波交替变化，方便在RDK X5终端观察数据是否到达
    int wave = (millis() / 1000) % 2;               // 每1秒切换一次
    a0 = 512;                                        // 脉搏对照固定
    a6 = wave ? 700 : 300;                           // 肱二头肌方波 300<->700
    a7 = wave ? 200 : 600;                           // 肱三头肌反相方波 600<->200
    // 直接赋值（不做滤波）
    EMG_A6_Signal = a6;
    EMG_A7_Signal = a7;
    EMG_A6_Smooth = a6;
    EMG_A7_Smooth = a7;
#else
    // ============ 正常模式: 读取真实ADC (12位 + 过采样) ============
    timerAlarmDisable(timer);
    a0 = analogRead(pulsePin);        // 脉搏引脚 A0（对照）

    // 过采样: 每个EMG引脚读8次取平均，压制底噪
    int sum6 = 0, sum7 = 0;
    for (int i = 0; i < 8; i++) {
      sum6 += analogRead(emgPinA6);
      sum7 += analogRead(emgPinA7);
    }
    a6 = sum6 / 8;
    a7 = sum7 / 8;
    timerAlarmEnable(timer);

    // 指数移动平均滤波 (alpha=0.3)
    EMG_A6_Signal = a6;
    EMG_A7_Signal = a7;
    EMG_A6_Smooth = (EMG_A6_Signal * 3 + EMG_A6_Smooth * 7) / 10;
    EMG_A7_Smooth = (EMG_A7_Signal * 3 + EMG_A7_Smooth * 7) / 10;
#endif

    // 串口绘图器格式 (空格分隔)
    Serial.print(a0);
    Serial.print(" ");
    Serial.print(a6);
    Serial.print(" ");
    Serial.print(EMG_A6_Smooth);
    Serial.print(" ");
    Serial.print(a7);
    Serial.print(" ");
    Serial.println(EMG_A7_Smooth);

#if ENABLE_BLE
    if (deviceConnected) {
      updateBLEEMG(EMG_A6_Smooth, EMG_A7_Smooth);
    }
#endif

#if ENABLE_WIFI
    // 向 RDK X5 发送 EMG+心率数据 (UDP)
    sendEMGtoRDK(EMG_A6_Smooth, EMG_A7_Smooth, BPM);
#endif
  }

  ledFadeToBeat();

#if ENABLE_WIFI
  udpLoop();   // WiFi 状态检查
#endif

  delay(20);
}

void ledFadeToBeat(){
  fadeRate -= 15;
  fadeRate = constrain(fadeRate, 0, 255);
  analogWrite(fadePin, fadeRate);
}

void sendDataToProcessing(char symbol, int data){
  Serial.print(symbol);
  Serial.println(data);
}

