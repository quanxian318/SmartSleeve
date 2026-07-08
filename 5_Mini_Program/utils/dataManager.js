/**
 * Shared global data manager for sensor data and connections.
 * Singleton pattern — all pages share the same WebSocket and sensor state.
 *
 * Usage:
 *   const dm = require('../../utils/dataManager');
 *   dm.subscribe((data) => { this.setData({...}); });
 *   dm.connectRDK('192.168.43.225');
 *   dm.runInference(rfInference);
 */

const app = getApp();

// Initialize global sensor data store (once)
if (!app.globalData.sensorData) {
  app.globalData.sensorData = {
    // Camera angles from RDK X5 WebSocket
    leftElbowAngle: 0,
    rightElbowAngle: 0,
    leftUpperAngle: 0,
    rightUpperAngle: 0,
    leftValid: false,
    rightValid: false,
    rdkAngle: 180,
    rdkStatus: '未连接',
    rdkIP: '192.168.43.9',
    lastSkeleton: null,

    // BLE sensor data
    emg: 0,
    ecg: 0,
    fsr: 0,
    bpm: 0,
    ibi: 0,
    bicepsBLE: 0,       // 真实BLE肱二头肌(ADC 0-1023)
    tricepsBLE: 0,      // 真实BLE肱三头肌(ADC 0-1023)
    pulseSignal: 0,     // 脉搏原始信号
    emgPercent: 0,
    deviceId: '',
    deviceType: '',
    connected: false,

    // ML predicted EMG
    predictedBiceps: 0,
    predictedTriceps: 0,
    predictedBrachioradialis: 0,
    // TCN predicted EMG from RDK WebSocket
    tcnBiceps: 0,
    tcnTriceps: 0,
    tcnBrachioradialis: 0,
    // Compensation + electrode quality
    compensationScore: 0,
    compensationLevel: 'unknown',
    bicepsQuality: 100,
    tricepsQuality: 100,
    dropoutB: false,
    dropoutT: false,
    r2Biceps: null,
    r2Triceps: null,
    qualityAlerts: [],

    // User profile for ML inference
    gender: 0,
    weight: 70,
    height: 175,
    actionId: 19
  };
}

if (!app.globalData._sensorSubscribers) {
  app.globalData._sensorSubscribers = [];
}

const dataManager = {
  /** Get current sensor data snapshot */
  getSensorData() {
    return app.globalData.sensorData;
  },

  /** Update partial sensor data and notify subscribers */
  updateSensorData(partial) {
    const data = app.globalData.sensorData;
    Object.assign(data, partial);
    // Auto-compute emgPercent only when caller didn't provide it.
    // Respects demo mode (μV-scale) and real BLE (ADC-scale) by not overriding.
    if (partial.emg !== undefined && partial.emgPercent === undefined) {
      const val = Math.max(0, data.emg - 400);
      data.emgPercent = Math.min(100, Math.max(0, (val / 2100) * 100));
    }
    // Notify all subscribers
    const subs = app.globalData._sensorSubscribers;
    for (let i = 0; i < subs.length; i++) {
      try { subs[i](data); } catch (e) { console.error('[DataMgr] Subscriber error:', e); }
    }

    // Debounced cloud sync: upload sensor snapshot periodically
    this._scheduleSync();
  },

  /** Internal: sync sensor snapshot to cloud (for doctor real-time view) */
  _scheduleSync: function () {
    var self = this;
    var now = Date.now();
    if (!this._lastSyncTime) this._lastSyncTime = 0;
    if (now - this._lastSyncTime < 5000) return; // max every 5 seconds
    this._lastSyncTime = now;

    // Only sync if patient has a doctor
    var doctorId = app.globalData.doctorId;
    if (!doctorId) {
      doctorId = wx.getStorageSync('doctorId');
      if (doctorId) app.globalData.doctorId = doctorId;
    }
    if (!doctorId) return; // no doctor bound, skip

    var data = app.globalData.sensorData;
    var db = wx.cloud.database();
    db.collection('users').where({ _openid: '{openid}' }).update({
      data: {
        lastSensorData: {
          emg: data.emg || 0,
          bpm: data.bpm || 0,
          rdkAngle: data.rdkAngle || 180,
          leftElbowAngle: data.leftElbowAngle || 0,
          fsr: data.fsr || 0
        },
        lastSensorTime: new Date()
      }
    }).then(function () {
      // silent success
    }).catch(function (err) {
      // Ignore — sensor sync is best-effort
    });
  },

  /** Subscribe to data changes. Returns unsubscribe function. */
  subscribe(fn) {
    app.globalData._sensorSubscribers.push(fn);
    return () => {
      const arr = app.globalData._sensorSubscribers;
      const idx = arr.indexOf(fn);
      if (idx >= 0) arr.splice(idx, 1);
    };
  },

  // --- WebSocket to RDK X5 ---

  /**
   * Connect WebSocket to RDK X5 vision system.
   * @param {string} ip - RDK X5 IP address
   * @returns {Promise}
   */
  connectRDK(ip) {
    var self = this;
    return new Promise(function (resolve, reject) {
      // Close existing connection if any
      if (app.globalData._socketTask) {
        try { app.globalData._socketTask.close({}); } catch (e) { /* ignore */ }
        app.globalData._socketTask = null;
      }

      if (ip) {
        self.updateSensorData({ rdkIP: ip });
      }
      var targetIP = ip || app.globalData.sensorData.rdkIP;
      var url = 'ws://' + targetIP + ':8765';
      console.log('[DataMgr] Connecting to: ' + url);
      self.updateSensorData({ rdkStatus: '正在连接...' });

      var task = wx.connectSocket({
        url: url,
        tcpNoDelay: true,
        success: function () {
          console.log('[DataMgr] WS api call OK, waiting for onOpen...');
        },
        fail: function (err) {
          console.error('[DataMgr] WS connect fail:', JSON.stringify(err));
          var msg = '连接失败: ';
          if (err.errMsg) {
            if (err.errMsg.indexOf('timeout') !== -1) {
              msg += '超时(检查IP是否正确,手机和RDK是否同WiFi)';
            } else if (err.errMsg.indexOf('refused') !== -1) {
              msg += '端口拒绝(检查RDK服务是否运行)';
            } else if (err.errMsg.indexOf('url not in domain list') !== -1 || err.errMsg.indexOf('legal') !== -1) {
              msg += '域名未授权(请开启调试模式/不校验域名)';
            } else if (err.errMsg.indexOf('network') !== -1) {
              msg += '网络不可达(手机和RDK在同一WiFi吗?)';
            } else {
              msg += err.errMsg;
            }
          }
          self.updateSensorData({ rdkStatus: msg });
          reject(err);
        }
      });

      var timeout = setTimeout(function () {
        if (app.globalData.sensorData.rdkStatus === '正在连接...') {
          console.error('[DataMgr] WS connection timed out after 10s');
          self.updateSensorData({ rdkStatus: '连接超时(请检查WiFi和IP)' });
          try { task.close({}); } catch (e) {}
        }
      }, 10000);

      task.onOpen(function () {
        clearTimeout(timeout);
        console.log('[DataMgr] WebSocket connected to RDK X5');
        app.globalData._socketTask = task;
        self.updateSensorData({ rdkStatus: '已连接' });
        resolve();
      });

      task.onMessage(function (res) {
        try {
          var v = JSON.parse(res.data);
          // 数字孪生主角度 = 实时监控页「右肘角度」那个值 (right_elbow_angle)
          // 有读数(非0)就更新, 否则保持上一帧, 避免检测丢失时手臂跳变。
          var mainAngle = app.globalData.sensorData.rdkAngle;
          var elbowAngle = Number(v.right_elbow_angle);
          if (elbowAngle) mainAngle = elbowAngle;

          self.updateSensorData({
            leftElbowAngle: Number(v.left_elbow_angle) || 0,
            rightElbowAngle: Number(v.right_elbow_angle) || 0,
            leftUpperAngle: Number(v.left_upper_angle) || 0,
            rightUpperAngle: Number(v.right_upper_angle) || 0,
            leftValid: !!v.left_valid,
            rightValid: !!v.right_valid,
            rdkAngle: Math.round(mainAngle),
            lastSkeleton: v.points || null,
            // TCN predicted EMG from RDK WebSocket (右臂, 与主角度同侧)
            tcnBiceps: Number(v.right_biceps_uv) || 0,
            tcnTriceps: Number(v.right_triceps_uv) || 0,
            tcnBrachioradialis: Number(v.right_brachioradialis_uv) || (Number(v.right_biceps_uv) || 0) * 0.7,
            // 代偿 + 电极质量
            compensationScore: Number(v.compensation_score) || 0,
            compensationLevel: v.compensation_level || 'unknown',
            bicepsQuality: Number(v.biceps_quality) || 100,
            tricepsQuality: Number(v.triceps_quality) || 100,
            dropoutB: !!v.dropout_b,
            dropoutT: !!v.dropout_t,
            r2Biceps: v.r2_biceps != null ? Number(v.r2_biceps) : null,
            r2Triceps: v.r2_triceps != null ? Number(v.r2_triceps) : null,
            qualityAlerts: v.quality_alerts || []
          });
        } catch (e) {
          console.error('[DataMgr] Parse error:', e);
        }
      });

      task.onError(function (err) {
        clearTimeout(timeout);
        console.error('[DataMgr] WS error:', JSON.stringify(err));
        self.updateSensorData({ rdkStatus: '连接出错: ' + (err.errMsg || '') });
      });

      task.onClose(function () {
        console.log('[DataMgr] WS closed');
        self.updateSensorData({ rdkStatus: '已断开' });
        app.globalData._socketTask = null;
      });
    });
  },

  /** Disconnect from RDK X5 and reset all sensor fields */
  disconnectRDK() {
    if (app.globalData._socketTask) {
      try { app.globalData._socketTask.close({}); } catch (e) { /* ignore */ }
      app.globalData._socketTask = null;
    }
    this.updateSensorData({
      rdkStatus: '未连接',
      rdkAngle: 180,
      leftElbowAngle: 0, rightElbowAngle: 0,
      leftUpperAngle: 0, rightUpperAngle: 0,
      leftValid: false, rightValid: false,
      lastSkeleton: null,
      tcnBiceps: 0, tcnTriceps: 0, tcnBrachioradialis: 0,
      compensationScore: 0, compensationLevel: 'unknown',
      bicepsQuality: 100, tricepsQuality: 100,
      dropoutB: false, dropoutT: false,
      r2Biceps: null, r2Triceps: null,
      qualityAlerts: []
    });
  },

  isRDKConnected() {
    return app.globalData.sensorData.rdkStatus === '已连接';
  },

  // --- ML Inference ---

  /**
   * Run model inference on current sensor data.
   * @param {Object} rfEngine - the rfInference module
   * @returns {{brachioradialis: number, biceps: number, triceps: number}|null}
   */
  runInference(rfEngine) {
    if (!rfEngine || !rfEngine.isLoaded()) return null;
    const d = app.globalData.sensorData;
    const bmi = d.height > 0 ? d.weight / Math.pow(d.height / 100, 2) : 22;
    // rdkAngle 约定 180°=伸直、越小越弯; 训练数据的 angle 是屈曲角 0°=伸直、越大越弯。
    // 两者方向相反, 必须转换后再喂模型, 否则预测趋势会整个反过来。
    var flexionAngle = 180 - d.rdkAngle;
    const features = [flexionAngle, d.gender, d.weight, d.height, bmi, d.actionId];
    const result = rfEngine.predict(features);
    // Output order: brachioradialis, biceps, triceps
    var brachioradialis = result[0];
    var biceps = result[1];
    var triceps = result[2];
    this.updateSensorData({
      predictedBrachioradialis: brachioradialis,
      predictedBiceps: biceps,
      predictedTriceps: triceps
    });
    return { brachioradialis, biceps, triceps };
  },

  /** Update user profile for ML inference */
  setUserProfile(gender, weight, height, actionId) {
    this.updateSensorData({ gender, weight, height, actionId });
  }
};

module.exports = dataManager;
