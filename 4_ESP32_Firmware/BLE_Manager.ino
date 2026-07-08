/*
 * BLE_Manager.ino - 蓝牙BLE心率服务管理
 * 标准心率服务(Heart Rate Service) + 自定义数据服务(双路EMG)
 * 肱二头肌 EMG (A6) + 肱三头肌 EMG (A7)
 */

#if ENABLE_BLE

// ===================== BLE连接回调 =====================
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("BLE: 客户端已连接");
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("BLE: 客户端已断开");
  }
};

// ===================== 初始化BLE =====================
void initBLE() {
  Serial.println("BLE: 正在初始化...");

  // 先检查蓝牙是否可用
  if (!BLEDevice::getInitialized()) {
    BLEDevice::init(DEVICE_NAME);
  }
  Serial.print("BLE: 设备名 = ");
  Serial.println(DEVICE_NAME);

  // 创建服务器
  pServer = BLEDevice::createServer();
  if (!pServer) {
    Serial.println("BLE: ERROR - 创建服务器失败!");
    return;
  }
  pServer->setCallbacks(new MyServerCallbacks());
  Serial.println("BLE: 服务器已创建");

  // --- 标准心率服务 (Heart Rate Service, 0x180D) ---
  BLEService* pHRService = pServer->createService(BLEUUID(SERVICE_UUID_HEART_RATE));
  if (!pHRService) {
    Serial.println("BLE: ERROR - 创建心率服务失败!");
    return;
  }

  // 心率测量值特征 (0x2A37) - 通知
  pHRMCharacteristic = pHRService->createCharacteristic(
    BLEUUID(CHARACTERISTIC_UUID_HRM),
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pHRMCharacteristic->addDescriptor(new BLE2902());
  Serial.println("BLE: 心率特征已创建 (0x2A37)");

  // 身体传感器位置 (0x2A38) - 读取
  BLECharacteristic* pBSLCharacteristic = pHRService->createCharacteristic(
    BLEUUID(CHARACTERISTIC_UUID_BSL),
    BLECharacteristic::PROPERTY_READ
  );
  uint8_t bodyLocation = 3; // 手指
  pBSLCharacteristic->setValue(&bodyLocation, 1);

  pHRService->start();
  Serial.println("BLE: 心率服务已启动");

  // --- 自定义数据服务 (原始信号 + IBI + 双路EMG) ---
  BLEService* pCustomService = pServer->createService(BLEUUID(SERVICE_UUID_CUSTOM));
  if (!pCustomService) {
    Serial.println("BLE: ERROR - 创建自定义服务失败!");
    return;
  }

  // 原始信号特征 - 通知
  pSignalCharacteristic = pCustomService->createCharacteristic(
    BLEUUID(CHARACTERISTIC_UUID_SIGNAL),
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pSignalCharacteristic->addDescriptor(new BLE2902());

  // IBI特征 - 通知
  pIBICharacteristic = pCustomService->createCharacteristic(
    BLEUUID(CHARACTERISTIC_UUID_IBI),
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pIBICharacteristic->addDescriptor(new BLE2902());

  // 肱二头肌 EMG (A6) 特征 - 通知
  pEMG_BicepsCharacteristic = pCustomService->createCharacteristic(
    BLEUUID(CHARACTERISTIC_UUID_EMG_BICEPS),
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pEMG_BicepsCharacteristic->addDescriptor(new BLE2902());
  Serial.println("BLE: 肱二头肌EMG特征已创建 (A6)");

  // 肱三头肌 EMG (A7) 特征 - 通知
  pEMG_TricepsCharacteristic = pCustomService->createCharacteristic(
    BLEUUID(CHARACTERISTIC_UUID_EMG_TRICEPS),
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pEMG_TricepsCharacteristic->addDescriptor(new BLE2902());
  Serial.println("BLE: 肱三头肌EMG特征已创建 (A7)");

  pCustomService->start();
  Serial.println("BLE: 自定义服务已启动 (双路EMG)");

  // --- 广播设置 ---
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(BLEUUID(SERVICE_UUID_HEART_RATE));
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinInterval(32);    // 32 * 0.625ms = 20ms
  pAdvertising->setMaxInterval(64);    // 64 * 0.625ms = 40ms

  BLEDevice::startAdvertising();
  Serial.println("BLE: 广播已开始! 设备名: " + String(DEVICE_NAME));
  Serial.println("BLE: 请用手机搜索 'ECG-Monitor'");
}

// ===================== 更新BLE心率数据 =====================
void updateBLEHeartRate(int bpm, int signal, int ibi) {
  if (!pHRMCharacteristic) return;

  // 标准心率测量格式 (Bluetooth SIG规范)
  uint8_t hrmData[2];
  hrmData[0] = 0x00;
  hrmData[1] = (uint8_t)constrain(bpm, 0, 255);
  pHRMCharacteristic->setValue(hrmData, 2);
  pHRMCharacteristic->notify();

  // 发送原始信号
  if (pSignalCharacteristic) {
    uint8_t signalData[2];
    signalData[0] = (uint8_t)(signal & 0xFF);
    signalData[1] = (uint8_t)((signal >> 8) & 0xFF);
    pSignalCharacteristic->setValue(signalData, 2);
    pSignalCharacteristic->notify();
  }

  // 发送IBI
  if (pIBICharacteristic) {
    uint8_t ibiData[2];
    ibiData[0] = (uint8_t)(ibi & 0xFF);
    ibiData[1] = (uint8_t)((ibi >> 8) & 0xFF);
    pIBICharacteristic->setValue(ibiData, 2);
    pIBICharacteristic->notify();
  }
}

// ===================== 更新BLE双路肌电数据 =====================
void updateBLEEMG(int a6_smooth, int a7_smooth) {
  // 肱二头肌 EMG (A6)
  if (pEMG_BicepsCharacteristic) {
    uint8_t emgData[2];
    emgData[0] = (uint8_t)(a6_smooth & 0xFF);
    emgData[1] = (uint8_t)((a6_smooth >> 8) & 0xFF);
    pEMG_BicepsCharacteristic->setValue(emgData, 2);
    pEMG_BicepsCharacteristic->notify();
  }

  // 肱三头肌 EMG (A7)
  if (pEMG_TricepsCharacteristic) {
    uint8_t emgData[2];
    emgData[0] = (uint8_t)(a7_smooth & 0xFF);
    emgData[1] = (uint8_t)((a7_smooth >> 8) & 0xFF);
    pEMG_TricepsCharacteristic->setValue(emgData, 2);
    pEMG_TricepsCharacteristic->notify();
  }
}

#endif // ENABLE_BLE
