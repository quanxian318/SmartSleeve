/*
 * WiFi_Manager.ino - WiFi STA模式 + UDP EMG数据发送
 * ESP32 连接到路由器 → 通过UDP向 RDK X5 推送双路EMG数据
 *
 * 数据格式 (CSV, 50Hz):
 *   seq,timestamp,biceps_uv,triceps_uv,bpm
 *   例: 1,123456789,512,340,72
 */

#if ENABLE_WIFI

static unsigned long udpSeq = 0;  // UDP包序号

// ===================== 初始化WiFi (STA模式) =====================
void initWiFi() {
  Serial.println("WiFi: 正在连接到 " + String(WIFI_SSID) + " ...");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  // 等待连接（最多15秒）
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 150) {
    delay(100);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println("\nWiFi: 已连接!");
    Serial.print("WiFi: IP地址 = ");
    Serial.println(WiFi.localIP());
    Serial.print("WiFi: 目标 RDK X5 = ");
    Serial.print(RDK_X5_IP);
    Serial.print(":");
    Serial.println(RDK_X5_PORT);

    // 初始化 UDP
    udpClient.begin(RDK_X5_PORT);
    Serial.println("WiFi: UDP 已就绪");
  } else {
    wifiConnected = false;
    Serial.println("\nWiFi: 连接失败! 请检查SSID和密码");
    Serial.println("WiFi: 将继续尝试重连...");
  }
}

// ===================== WiFi连接检查与重连 =====================
void udpLoop() {
  // 每5秒检查一次WiFi状态
  if (millis() - lastWiFiCheck < 5000) return;
  lastWiFiCheck = millis();

  if (WiFi.status() != WL_CONNECTED) {
    if (wifiConnected) {
      wifiConnected = false;
      Serial.println("WiFi: 连接断开，尝试重连...");
    }
    WiFi.reconnect();
    if (WiFi.status() == WL_CONNECTED) {
      wifiConnected = true;
      Serial.println("WiFi: 已重新连接! IP=" + WiFi.localIP().toString());
    }
  } else if (!wifiConnected) {
    wifiConnected = true;
    Serial.println("WiFi: 连接正常");
  }
}

// ===================== 向RDK X5发送EMG+心率数据 (UDP) =====================
// 格式: seq,timestamp,biceps_uv,triceps_uv,bpm
void sendEMGtoRDK(int a6_smooth, int a7_smooth, int bpm) {
  if (!wifiConnected) return;

  udpSeq++;

  // 构建 CSV 数据包: seq,ts,biceps,triceps,bpm
  String packet = String(udpSeq) + "," +
                  String(millis()) + "," +
                  String(a6_smooth) + "," +
                  String(a7_smooth) + "," +
                  String(bpm);

  // UDP 发送
  udpClient.beginPacket(RDK_X5_IP, RDK_X5_PORT);
  udpClient.print(packet);
  udpClient.endPacket();
}

#endif // ENABLE_WIFI
