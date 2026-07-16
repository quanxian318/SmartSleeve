var app = getApp();
var dataManager = require('../../utils/dataManager');
var roleManager = require('../../utils/roleManager');

Page({
  data: {
    role: '',
    connected: false,
    currentMode: 'rehab',

    deviceId: '', deviceType: '', buffer: '',
    emg: 0, ecg: 0, fsr: 0, bpm: 0, ibi: 0, biceps: 0, triceps: 0,

    rdkIP: '192.168.43.9', rdkStatus: '未连接', rdkAngle: 180,
    socketTask: null, lastSkeleton: null,
    leftElbowAngle: 0, rightElbowAngle: 0,
    leftUpperAngle: 0, rightUpperAngle: 0,
    leftValid: false, rightValid: false,

    aiStatus: 'normal', aiTitle: '医生就绪', aiMessage: '等待同步...',
    aiScore: 0, aiIconColor: '#2ecc71',
    emgPercent: 0, smooth_p_u: 0.0,
    animTimer: null, mockTimer: null, t: 0,

    inviteCode: '', patientList: [], patientCount: 0, doctorLoading: false,
    patientTasks: [],
    activeTaskReps: 0, activeTaskInZone: false,

    // Patient tab switch
    patientTab: 'monitor',

    // Waveform charts
    showHeartChart: false,
    showEmgChart: false,
    emgChartMuscle: 'biceps',

    // Quality cards (from RDK WebSocket)
    compScore: '--',
    compLevelClass: 'normal',
    compLabel: '等待同步',
    bicepsQuality: 100,
    tricepsQuality: 100,
    elecBClass: 'good',
    elecTClass: 'good',
    dropoutB: false,
    dropoutT: false
  },

  onLoad(options) {
    try {
      if (!roleManager.requireRole()) return;
      this.setData({ role: app.globalData.role });
      if (options && options.doctorId && app.globalData.role === 'patient') this.handleQRBind(options.doctorId);
      if (options && options.taskId && app.globalData.role === 'patient') this.setData({ activeTaskId: options.taskId });
      // Support tab switch from external navigation
      if (options && options.tab === 'tasks' && app.globalData.role === 'patient') {
        this.setData({ patientTab: 'tasks' });
      }
      this.imgs = {};
      this.lastUpdateTime = 0;
      this._activeTaskReps = 0;
      this._activeTaskInZone = false;
      this._activeTaskLeaveZone = false;
      var that = this;
      this._unsub = dataManager.subscribe(function (data) {
      that.setData({
        rdkAngle: data.rdkAngle, rdkStatus: data.rdkStatus,
        leftElbowAngle: Number(data.leftElbowAngle || 0).toFixed(2),
        rightElbowAngle: Number(data.rightElbowAngle || 0).toFixed(2),
        leftUpperAngle: Number(data.leftUpperAngle || 0).toFixed(2),
        rightUpperAngle: Number(data.rightUpperAngle || 0).toFixed(2),
        leftValid: data.leftValid, rightValid: data.rightValid, lastSkeleton: data.lastSkeleton
      });
      that._recordDataPoint(data);
      that._updateQualityCards(data);
      that.updateAIDiagnosis(data.emg, data.bpm, data.rdkAngle, data.fsr);
      that._trackActiveTask(data.rdkAngle);
    });
    } catch (e) {
      console.error('[Index] onLoad error:', e.message, e.stack);
      wx.setStorageSync('_lastCrash', 'Index.onLoad: ' + e.message);
    }
  },

  onShow() {
    this.setData({ role: app.globalData.role });
    if (app.globalData.cloudSynced && app.globalData.role) roleManager.setupTabBar(app.globalData.role);
    if (app.globalData.role === 'doctor') {
      this.setData({ inviteCode: app.globalData.inviteCode || '' });
      this.fetchPatients();
    }
    if (app.globalData.role === 'patient') {
      // Open tasks tab if navigated from "My Training Tasks"
      if (app.globalData._openTasksTab) {
        this.setData({ patientTab: 'tasks' });
        app.globalData._openTasksTab = false;
      }
      if (app.globalData.activeTask) {
        this.setData({ activeTaskId: app.globalData.activeTask._id, activeTask: app.globalData.activeTask });
        this._activeTaskReps = 0;
        this._activeTaskInZone = false;
        this._activeTaskLeaveZone = false;
        this.setData({ activeTaskReps: 0, activeTaskInZone: false });
      }
      this.fetchPatientTasks();
      // 退出再进入时，如果之前连着BLE但数据停了，尝试重连
      if (this._wasConnected && !this._bleDataTimer) {
        this._wasConnected = false;
        this.connectBluetooth();
      }
    }
  },

  onHide() {
    // 记录当前连接状态，onShow 时用于判断是否需要重连
    this._wasConnected = this.data.connected;
    this._bleDataTimer = null;
  },

  // ========== Doctor ==========

  fetchPatients() {
    var that = this;
    this.setData({ doctorLoading: true });
    wx.cloud.callFunction({
      name: 'getDoctorPatients', data: {},
      success: function (res) {
        var r = res.result;
        if (r && r.success) that.setData({ patientList: r.patients || [], patientCount: r.count || 0, doctorLoading: false });
        else that.fetchPatientsDB();
      },
      fail: function () { that.fetchPatientsDB(); }
    });
  },

  fetchPatientsDB: function () {
    var self = this;
    var db = wx.cloud.database();
    db.collection('users').where({ doctorId: app.globalData.userInfo ? app.globalData.userInfo._openid : '' })
      .field({ _openid: true, customName: true, lastSensorTime: true }).get()
      .then(function (res) {
        self.setData({ patientList: (res.data || []).map(function (p) { return { _openid: p._openid, customName: p.customName || '患者', isOnline: p.lastSensorTime ? (Date.now() - new Date(p.lastSensorTime).getTime() < 30000) : false }; }), patientCount: res.data.length, doctorLoading: false });
      }).catch(function () { self.setData({ doctorLoading: false }); });
  },

  copyInviteCode() {
    if (!this.data.inviteCode) return;
    wx.setClipboardData({ data: this.data.inviteCode, success: function () { wx.showToast({ title: '已复制', icon: 'success' }); } });
  },

  openPatientDetail(e) {
    var id = e.currentTarget.dataset.patientid;
    var name = e.currentTarget.dataset.patientname;
    wx.navigateTo({ url: '/pages/patientDetail/patientDetail?patientId=' + id + '&patientName=' + encodeURIComponent(name || '') });
  },

  onFabTap: function () {
    wx.navigateTo({ url: '/pages/taskPublish/taskPublish' });
  },

  // ========== Patient Tasks ==========

  fetchPatientTasks: function () {
    var self = this;
    // Use cloud function to bypass DB permission: tasks are created by doctor,
    // but patient needs to read them. Cloud functions run with admin privileges.
    wx.cloud.callFunction({
      name: 'fetchMyTasks',
      data: { limit: 50 },
      success: function (res) {
        var result = res.result;
        if (result && result.success) {
          // Tasks already enriched with deadlineText and isOverdue by the cloud function
          self.setData({ patientTasks: result.tasks || [] });
        } else {
          // Fallback to direct DB query (may work if permissions are set correctly)
          self._fetchPatientTasksDirect();
        }
      },
      fail: function () {
        // Fallback to direct DB query
        self._fetchPatientTasksDirect();
      }
    });
  },

  /** Fallback: direct DB query for patient tasks */
  _fetchPatientTasksDirect: function () {
    var self = this;
    wx.cloud.database().collection('tasks')
      .where({ patientId: '{openid}' }).orderBy('createdAt', 'desc').limit(10).get()
      .then(function (res) {
        var tasks = (res.data || []).map(function (item) {
          var deadlineText = '';
          var isOverdue = false;
          if (item.deadline) {
            var dl = new Date(item.deadline);
            var now = new Date();
            var diffTime = dl.getTime() - now.getTime();
            var diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
            if (diffDays < 0) {
              deadlineText = '已逾期 ' + Math.abs(diffDays) + ' 天';
              isOverdue = true;
            } else if (diffDays === 0) {
              deadlineText = '今天截止';
            } else {
              deadlineText = '剩余 ' + diffDays + ' 天';
            }
          } else {
            deadlineText = '无截止日期';
          }
          return Object.assign({}, item, { deadlineText: deadlineText, isOverdue: isOverdue });
        });
        self.setData({ patientTasks: tasks });
      }).catch(function () {});
  },

  openPatientTask: function (e) {
    wx.navigateTo({ url: '/pages/taskDetail/taskDetail?taskId=' + e.currentTarget.dataset.taskid });
  },

  // ========== WiFi / Sensor / BLE (unchanged) ==========

  inputIP(e) { this.setData({ rdkIP: e.detail.value }); },
  connectRDK() {
    var that = this;
    if (!this.data.rdkIP) { wx.showToast({ title: '请先输入IP', icon: 'none' }); return; }
    this.setData({ rdkStatus: '正在连接...' });
    wx.showLoading({ title: 'WiFi 握手中...' });
    dataManager.connectRDK(this.data.rdkIP).then(function () { wx.hideLoading(); that.setData({ rdkStatus: '已连接' }); wx.showToast({ title: '已同步', icon: 'success' }); }).catch(function () { wx.hideLoading(); that.setData({ rdkStatus: '连接失败' }); });
  },

  /** Switch between monitoring and tasks tab on patient view */
  switchPatientTab: function (e) {
    var tab = e.currentTarget.dataset.tab;
    this.setData({ patientTab: tab });
  },

  updateAIDiagnosis(emg, bpm, angle, fsr) {
    var status = 'normal', title = '同步中', msg = '系统运行正常', color = '#2ecc71', score = 95;
    if (bpm > 120) { status = 'danger'; title = '心率过载'; msg = '请立即停止运动！'; color = '#e74c3c'; score = 10; }
    else if (this.data.currentMode === 'rehab' && angle < 130 && emg > 1200) { status = 'warn'; title = '检测到代偿'; msg = '手臂伸展不足却伴随强信号'; color = '#f1c40f'; score = 45; }
    this.setData({ aiStatus: status, aiTitle: title, aiMessage: msg, aiIconColor: color, aiScore: score });
  },

  /** Update quality card display from WebSocket data */
  _updateQualityCards: function (data) {
    var score = data.compensationScore || 0;
    var level = data.compensationLevel || 'unknown';
    var compClass = level === 'severe' ? 'danger' : (level === 'moderate' ? 'warn' : 'normal');
    var compLabel = level === 'normal' ? '正常' : (level === 'mild' ? '轻度' : (level === 'moderate' ? '中度' : (level === 'severe' ? '严重' : '--')));
    if (level === 'unknown' || level === 'insufficient_data') { compLabel = '采集中'; compClass = 'normal'; }

    var bq = data.bicepsQuality != null ? Number(data.bicepsQuality) : 100;
    var tq = data.tricepsQuality != null ? Number(data.tricepsQuality) : 100;
    var ebClass = bq < 50 ? 'danger' : (bq < 80 ? 'warn' : 'good');
    var etClass = tq < 50 ? 'danger' : (tq < 80 ? 'warn' : 'good');

    this.setData({
      compScore: Math.round(score),
      compLevelClass: compClass,
      compLabel: compLabel,
      bicepsQuality: Math.round(bq),
      tricepsQuality: Math.round(tq),
      elecBClass: ebClass,
      elecTClass: etClass,
      dropoutB: data.dropoutB || false,
      dropoutT: data.dropoutT || false
    });
  },

  /** Track active task progress: count reps when angle enters target zone */
  _trackActiveTask: function (angle) {
    var task = this.data.activeTask;
    if (!task) return;
    var min = Number(task.targetAngleMin);
    var max = Number(task.targetAngleMax);
    var inZone = angle >= min && angle <= max;

    if (inZone && !this._activeTaskInZone && this._activeTaskLeaveZone) {
      // Re-entered zone after leaving: count as one rep
      this._activeTaskReps = (this._activeTaskReps || 0) + 1;
      this._activeTaskLeaveZone = false;
      this.setData({ activeTaskReps: this._activeTaskReps });

      // Check completion
      if (this._activeTaskReps >= task.repetitions) {
        wx.showModal({
          title: '🎉 训练完成',
          content: '已完成全部 ' + task.repetitions + ' 次训练！',
          showCancel: false,
          success: function () {
            wx.navigateTo({ url: '/pages/taskDetail/taskDetail?taskId=' + task._id });
          }
        });
      }
    } else if (inZone && !this._activeTaskInZone && !this._activeTaskLeaveZone) {
      // First time entering zone for this rep
      this._activeTaskLeaveZone = false;
    } else if (!inZone && this._activeTaskInZone) {
      // Left zone — arm next rep detection
      this._activeTaskLeaveZone = true;
    }

    this._activeTaskInZone = inZone;
    this.setData({ activeTaskInZone: inZone });
  },

  processSensorData(rawEmg, rawEcg, rawFsr, rawBpm) {
    var now = Date.now(); if (now - this.lastUpdateTime < 80) return; this.lastUpdateTime = now;
    var emgVal = Math.max(0, parseInt(rawEmg) || 0);
    var ecgVal = Math.max(0, parseInt(rawEcg) || 0);
    var fsrVal = Math.max(0, parseInt(rawFsr) || 0);
    var bpmVal = Math.max(0, parseInt(rawBpm) || 0);
    this.setData({ emg: emgVal, ecg: ecgVal, fsr: fsrVal, bpm: bpmVal, emgPercent: Math.min(Math.max(((emgVal - 400) / 2100) * 100, 0), 100) });
    this.updateAIDiagnosis(emgVal, bpmVal, this.data.rdkAngle, fsrVal);
  },

  connectBluetooth() {
    wx.showLoading({ title: '正在初始化蓝牙...' });
    var that = this;
    // 先强制关闭再重新打开，避免二次进入时蓝牙适配器残留状态
    wx.closeBluetoothAdapter({
      complete: function () {
        // 等300ms确保底层释放完成
        setTimeout(function () {
          wx.openBluetoothAdapter({
            success: function () { wx.hideLoading(); that.searchDevice(); },
            fail: function (err) {
              wx.hideLoading();
              if (err.errCode === 10001) {
                wx.showModal({ title: '蓝牙未开启', content: '请在手机设置中开启蓝牙', showCancel: false });
              } else {
                wx.showToast({ title: '蓝牙初始化失败，请重试', icon: 'none' });
              }
            }
          });
        }, 300);
      }
    });
  },

  searchDevice() {
    var that = this;
    // 先停止之前的搜索（如果有）
    wx.stopBluetoothDevicesDiscovery({
      complete: function () {
        wx.startBluetoothDevicesDiscovery({
          success: function () {
            wx.onBluetoothDeviceFound(function (res) {
              var dev = res.devices[0]; if (!dev) return;
              var name = (dev.name || dev.localName || '').toUpperCase();
              if (name.includes('ECG') || name.includes('NANO') || name.includes('ESP32')) {
                wx.stopBluetoothDevicesDiscovery();
                that.connectToDevice(dev.deviceId, name);
              }
            });
          },
          fail: function () {
            wx.showToast({ title: '搜索设备失败', icon: 'none' });
          }
        });
      }
    });
  },

  connectToDevice(deviceId, name) {
    var that = this;
    wx.createBLEConnection({
      deviceId: deviceId,
      timeout: 10000,
      success: function () {
        var dtype = name.includes('ECG') ? 'ecg' : 'nano';
        that.setData({ connected: true, deviceId: deviceId, deviceType: dtype });
        // 记录数据接收时间戳，用于 onShow 判断是否需要重连
        that._bleDataTimer = Date.now();
        that.getServices(deviceId, dtype);
      },
      fail: function (err) {
        console.error('BLE连接失败', err);
        wx.showToast({ title: '连接设备失败，请靠近后重试', icon: 'none' });
      }
    });
  },

  getServices(deviceId, dtype) {
    var that = this;
    wx.getBLEDeviceServices({ deviceId: deviceId, success: function (res) {
      if (!res || !res.services || res.services.length === 0) {
        console.error('[Index] No BLE services found');
        wx.showToast({ title: '未找到设备服务', icon: 'none' });
        return;
      }
      if (dtype === 'ecg') {
        that.connectECGDevice(deviceId, res.services);
      } else {
        that.connectNanoDevice(deviceId, res.services);
      }
    }, fail: function (err) {
      console.error('[Index] 获取服务失败', err);
      wx.showToast({ title: '获取服务失败', icon: 'none' });
    } });
  },

  connectECGDevice(deviceId, services) {
    var hrService = services.find(function (s) { return s.uuid.toUpperCase().includes('180D'); });
    var customService = services.find(function (s) { return s.uuid.toUpperCase().includes('4FAFC201'); });
    var that = this;

    // 慢速数据(BPM/IBI/脉搏)批量刷新，EMG数据直接刷新保证响应速度
    this._slowCache = {};
    if (this._slowTimer) clearInterval(this._slowTimer);
    this._slowTimer = setInterval(function () {
      var c = that._slowCache;
      if (!c || !c._dirty) return;
      c._dirty = false;
      var u = {};
      if (c.bpm !== undefined) u.bpm = c.bpm;
      if (c.ibi !== undefined) u.ibi = c.ibi;
      if (c.ecg !== undefined) { u.ecg = c.ecg; u.emgPercent = Math.min(100, Math.max(0, ((c.ecg - 350) / 650) * 100)); }
      if (Object.keys(u).length > 0) that.setData(u);
    }, 100);

    wx.onBLECharacteristicValueChange(function (res) {
      that._bleDataTimer = Date.now();
      var uuid = res.characteristicId.toUpperCase();
      var bytes = new Uint8Array(res.value);
      var c = that._slowCache;
      if (uuid.includes('2A37') && bytes.length >= 2) {
        var bpm = (bytes[0] & 0x01) ? (bytes[1] | (bytes[2] << 8)) : bytes[1];
        c.bpm = bpm; c._dirty = true;
        dataManager.updateSensorData({ bpm: bpm });
      } else if (uuid.includes('B26A8') && bytes.length >= 2) {
        var signal = bytes[0] | (bytes[1] << 8);
        c.ecg = signal; c._dirty = true;
        dataManager.updateSensorData({ pulseSignal: signal });
      } else if (uuid.includes('B26A9') && bytes.length >= 2) {
        var ibi = bytes[0] | (bytes[1] << 8);
        c.ibi = ibi; c._dirty = true;
      } else if (uuid.includes('B26AA') && bytes.length >= 2) {
        var bicepsVal = bytes[0] | (bytes[1] << 8);
        that.setData({ biceps: bicepsVal });          // EMG直接刷，0延迟
        dataManager.updateSensorData({ bicepsBLE: bicepsVal });
      } else if (uuid.includes('B26AB') && bytes.length >= 2) {
        var tricepsVal = bytes[0] | (bytes[1] << 8);
        that.setData({ triceps: tricepsVal });         // EMG直接刷，0延迟
        dataManager.updateSensorData({ tricepsBLE: tricepsVal });
      }
    });
    if (hrService) {
      wx.getBLEDeviceCharacteristics({ deviceId: deviceId, serviceId: hrService.uuid, success: function (res) { var hrm = res.characteristics.find(function (c) { return c.uuid.toUpperCase().includes('2A37'); }); if (hrm) wx.notifyBLECharacteristicValueChange({ state: true, deviceId: deviceId, serviceId: hrService.uuid, characteristicId: hrm.uuid }); } });
    }
    if (customService) {
      wx.getBLEDeviceCharacteristics({ deviceId: deviceId, serviceId: customService.uuid, success: function (res) {
        res.characteristics.forEach(function (c) {
          var uid = c.uuid.toUpperCase();
          if (uid.includes('B26A8') || uid.includes('B26A9') || uid.includes('B26AA') || uid.includes('B26AB')) {
            wx.notifyBLECharacteristicValueChange({ state: true, deviceId: deviceId, serviceId: customService.uuid, characteristicId: c.uuid });
          }
        });
      }});
    }
    wx.showToast({ title: 'ECG已连接', icon: 'success' });
  },

  connectNanoDevice(deviceId, services) {
    var s = services.find(function (s) { return s.uuid.includes('FFE0'); }) || services[0]; if (!s) return;
    var that = this;
    wx.getBLEDeviceCharacteristics({ deviceId: deviceId, serviceId: s.uuid, success: function (res) { var c = res.characteristics.find(function (c) { return c.uuid.includes('FFE1'); }) || res.characteristics[0]; if (!c) return; wx.notifyBLECharacteristicValueChange({ state: true, deviceId: deviceId, serviceId: s.uuid, characteristicId: c.uuid, success: function () { wx.onBLECharacteristicValueChange(function (res) {
                    var raw = new Uint8Array(res.value);
                    // 分段拼接避免 String.fromCharCode.apply 参数过多崩溃
                    for (var bi = 0; bi < raw.length; bi++) { that.data.buffer += String.fromCharCode(raw[bi]); }
                    // 限制最大 8KB，防止无换行符时 OOM
                    if (that.data.buffer.length > 8192) { that.data.buffer = that.data.buffer.slice(-4096); }
                    var lines = that.data.buffer.split('\n');
                    if (lines.length > 1) { var vals = lines[lines.length - 2].split(','); if (vals.length >= 4) that.processSensorData(vals[0], vals[1], vals[2], vals[3]); that.data.buffer = lines[lines.length - 1]; }
                  }); } }); } });
  },

  startMockData() {
    if (this.data.mockTimer) return;
    this.setData({ connected: true });

    var that = this;
    // ── Realistic independent state ──
    var mockBpm = 72 + Math.random() * 8;
    var mockEmgTarget = 0, mockEmgCurrent = 0;
    var CYCLE = 5.5; // seconds per curl rep

    var timer = setInterval(function () {
      var t = that.data.t + 0.15;
      var cycleT = t % CYCLE;
      var phaseInCycle = cycleT / CYCLE; // 0..1

      // ════════════════════════════════════════════
      // Elbow angle: realistic curl cycle (NOT sine)
      //   0.00–0.36  concentric (bend): fast 180°→55°
      //   0.36–0.55  peak hold: ~55° + tremor
      //   0.55–0.91  eccentric (extend): slow 55°→175°
      //   0.91–1.00  rest at bottom: ~175°
      // ════════════════════════════════════════════
      var elbowAngle;
      if (phaseInCycle < 0.36) {
        var s = phaseInCycle / 0.36;
        var smooth = s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;
        elbowAngle = 180 - smooth * 125;
      } else if (phaseInCycle < 0.55) {
        elbowAngle = 55 + (Math.random() - 0.5) * 3;
      } else if (phaseInCycle < 0.91) {
        var s = (phaseInCycle - 0.55) / 0.36;
        var smooth = s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;
        elbowAngle = 55 + smooth * 120;
      } else {
        elbowAngle = 175 + (Math.random() - 0.5) * 5;
      }

      // ════════════════════════════════════════════
      // Shoulder: fixed ~90° with tiny tremor only
      // ════════════════════════════════════════════
      var shoulderAngle = 90 + (Math.random() - 0.5) * 2;

      // ════════════════════════════════════════════
      // BPM: independent random walk (NOT sine)
      // ════════════════════════════════════════════
      mockBpm += (Math.random() - 0.48) * 3.5;
      if (mockBpm > 90) mockBpm -= 1.5;
      if (mockBpm < 60) mockBpm += 1.5;
      mockBpm = mockBpm * 0.995 + 74 * 0.005;
      var bpm = Math.floor(Math.min(95, Math.max(58, mockBpm)));

      // ════════════════════════════════════════════
      // EMG: realistic muscle dynamics
      //   - concentric: 1.15x effort, eccentric: 0.75x
      //   - fast rise (0.4), slow decay (0.08)
      //   - 25% stochastic noise + occasional spikes
      // ════════════════════════════════════════════
      var angleRatio = Math.max(0, Math.min(1, (180 - elbowAngle) / 125));
      var isConcentric = (phaseInCycle < 0.36);
      var isEccentric = (phaseInCycle >= 0.55 && phaseInCycle < 0.91);
      var effort = isConcentric ? 1.15 : isEccentric ? 0.75 : 0.3;
      var targetUv = 50 + angleRatio * 650 * effort;
      var rate = targetUv > mockEmgTarget ? 0.4 : 0.08;
      mockEmgTarget = mockEmgTarget * (1 - rate) + targetUv * rate;
      mockEmgCurrent = mockEmgCurrent * 0.6 + mockEmgTarget * 0.4;
      var noiseStd = Math.max(8, mockEmgCurrent * 0.25);
      var emgUv = Math.round(Math.max(0, mockEmgCurrent + (Math.random() - 0.5) * 2 * noiseStd));
      if (Math.random() < 0.03 && isConcentric) emgUv += Math.round(Math.random() * 80);

      // Map μV to ADC range for display formula: (adc - 400) / 2100 * 100
      var emgAdc = Math.round(emgUv * 2.8);
      var emgPercent = Math.round(Math.min(100, Math.max(0, ((emgUv - 50) / 650) * 100)));

      // FSR: slight correlation with angle
      var fsr = Math.floor(500 + angleRatio * 1000 + (Math.random() - 0.5) * 300);
      // ECG: raw signal with noise
      var ecg = Math.floor(420 + (bpm - 70) * 2 + (Math.random() - 0.5) * 120);
      // BLE-style raw EMG for display
      var rawEmg = emgAdc;

      that.setData({ t: t });
      that.processSensorData(rawEmg, ecg, fsr, bpm);
      // Override emg display: processSensorData uses ADC scale, demo sends μV
      that.setData({ emg: emgUv, emgPercent: emgPercent });

      // Push to dataManager so digitalTwin + waveform charts get demo data
      dataManager.updateSensorData({
        rdkAngle: Math.round(elbowAngle),
        leftElbowAngle: Math.round(elbowAngle), rightElbowAngle: Math.round(elbowAngle),
        leftUpperAngle: Math.round(shoulderAngle), rightUpperAngle: Math.round(shoulderAngle),
        rightValid: true, leftValid: true,
        rdkStatus: '演示模式',
        emg: emgUv, bpm: bpm, fsr: fsr, emgPercent: emgPercent,
        // RF predicted (same source in demo since both derive from angle)
        predictedBiceps: emgUv,
        predictedTriceps: Math.round(emgUv * 0.25),
        predictedBrachioradialis: Math.round(emgUv * 0.55),
        // TCN predicted (RDK-side, simulated as slightly different for realism)
        tcnBiceps: Math.round(emgUv * (0.92 + Math.random() * 0.16)),
        tcnTriceps: Math.round(emgUv * 0.26),
        tcnBrachioradialis: Math.round(emgUv * 0.58)
      });
    }, 120);

    this.setData({ mockTimer: timer });
    wx.showToast({ title: '演示模式', icon: 'success' });
  },

  resetConnection() {
    if (this.data.mockTimer) clearInterval(this.data.mockTimer);
    if (this._heartChartTimer) { clearInterval(this._heartChartTimer); this._heartChartTimer = null; }
    if (this._emgChartTimer) { clearInterval(this._emgChartTimer); this._emgChartTimer = null; }
    if (this._slowTimer) { clearInterval(this._slowTimer); this._slowTimer = null; }
    this._slowCache = null;
    this._bpmHistory = []; this._predEmgHistory = {}; this._realEmgHistory = {};
    dataManager.disconnectRDK(); if (this.data.animTimer) cancelAnimationFrame(this.data.animTimer);
    wx.closeBluetoothAdapter();
    this.setData({ connected: false, deviceId: '', deviceType: '', buffer: '', emg: 0, ecg: 0, fsr: 0, bpm: 0, ibi: 0, biceps: 0, triceps: 0, rdkStatus: '未连接', rdkAngle: 180, socketTask: null, leftElbowAngle: 0, rightElbowAngle: 0, leftUpperAngle: 0, rightUpperAngle: 0, leftValid: false, rightValid: false, emgPercent: 0, smooth_p_u: 0.0, animTimer: null, mockTimer: null, aiStatus: 'normal', aiTitle: '医生就绪', aiMessage: '等待同步...', aiScore: 0, aiIconColor: '#2ecc71' });
    // Also reset demo tcn/predicted fields in dataManager so digitalTwin shows defaults
    dataManager.updateSensorData({
      predictedBiceps: 0, predictedTriceps: 0, predictedBrachioradialis: 0,
      tcnBiceps: 0, tcnTriceps: 0, tcnBrachioradialis: 0,
      emg: 0, bpm: 0, emgPercent: 0,
      bicepsBLE: 0, tricepsBLE: 0, pulseSignal: 0
    });
    wx.showToast({ title: '已重置', icon: 'none' });
  },

  handleQRBind(doctorId) {
    var that = this;
    wx.showLoading({ title: '绑定中...' });
    wx.cloud.callFunction({ name: 'joinDoctorByQR', data: { doctorId: doctorId }, success: function (res) { wx.hideLoading(); var r = res.result; if (r.success) { app.globalData.doctorId = r.doctorId; app.globalData.doctorName = r.doctorName; wx.setStorageSync('doctorId', r.doctorId); wx.setStorageSync('doctorName', r.doctorName); wx.showModal({ title: '绑定成功', content: '已绑定医生：' + r.doctorName, showCancel: false }); } else { wx.showModal({ title: '失败', content: r.error, showCancel: false }); } }, fail: function () { wx.hideLoading(); } });
  },

  onSaveRecord() {
    wx.cloud.database().collection('training_records').add({ data: { date: new Date(), maxEmg: this.data.emg, avgBpm: this.data.bpm, visionAngle: this.data.rdkAngle, mode: this.data.currentMode, score: this.data.aiScore }, success: function () { wx.showToast({ title: '已存' }); }, fail: function (err) { wx.showToast({ title: '失败', icon: 'none' }); } });
  },

  // ========== Waveform Charts ==========

  _bpmHistory: [],
  _predEmgHistory: {},  // { biceps: [...], triceps: [...], brachioradialis: [...] }
  _realEmgHistory: {},

  _initMuscleHistory: function () {
    var keys = ['biceps', 'triceps', 'brachioradialis'];
    var self = this;
    keys.forEach(function (k) {
      if (!self._predEmgHistory[k]) self._predEmgHistory[k] = [];
      if (!self._realEmgHistory[k]) self._realEmgHistory[k] = [];
    });
  },

  /** Called from dataManager subscriber to store data */
  _recordDataPoint: function (data) {
    var now = Date.now();
    var isDemo = data.rdkStatus === '演示模式';
    this._initMuscleHistory();
    if (data.bpm && data.bpm !== '--') {
      this._bpmHistory.push({ t: now, val: Number(data.bpm) || 0 });
      if (this._bpmHistory.length > 150) this._bpmHistory = this._bpmHistory.slice(-150);
    }
    var b = data.predictedBiceps != null ? Number(data.predictedBiceps) : (data.tcnBiceps || data.emg || 0);
    var t = data.predictedTriceps != null ? Number(data.predictedTriceps) : (data.tcnTriceps || (data.emg || 0) * 0.3);
    var br = data.predictedBrachioradialis != null ? Number(data.predictedBrachioradialis) : (data.tcnBrachioradialis || b * 0.7);
    this._predEmgHistory.biceps.push({ t: now, val: b });
    this._predEmgHistory.triceps.push({ t: now, val: t });
    this._predEmgHistory.brachioradialis.push({ t: now, val: br });
    this._realEmgHistory.biceps.push({ t: now, val: isDemo ? 0 : (data.emg || 0) });
    this._realEmgHistory.triceps.push({ t: now, val: isDemo ? 0 : (data.emg || 0) * 0.3 });
    this._realEmgHistory.brachioradialis.push({ t: now, val: isDemo ? 0 : (data.emg || 0) * 0.7 });
    var keys = ['biceps', 'triceps', 'brachioradialis'];
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      if (this._predEmgHistory[k].length > 150) this._predEmgHistory[k] = this._predEmgHistory[k].slice(-150);
      if (this._realEmgHistory[k].length > 150) this._realEmgHistory[k] = this._realEmgHistory[k].slice(-150);
    }
  },

  switchEmgMuscle: function (e) {
    var muscle = e.currentTarget.dataset.muscle;
    // Set BOTH sync property (immediate, for draw) and data (async, for UI)
    this._currentEmgMuscle = muscle;
    this.setData({ emgChartMuscle: muscle });
    // Stop the interval timer during transition to prevent stale draws
    if (this._emgChartTimer) { clearInterval(this._emgChartTimer); this._emgChartTimer = null; }
    // Immediately clear canvas so old waveform disappears
    this._clearEmgCanvas();
    // Cancel any pending draw to avoid race conditions on rapid switching
    if (this._muscleDrawTimer) clearTimeout(this._muscleDrawTimer);
    var self = this;
    this._muscleDrawTimer = setTimeout(function () {
      self._muscleDrawTimer = null;
      self._drawEmgChart();
      // Restart the refresh timer
      if (!self._emgChartTimer) {
        self._emgChartTimer = setInterval(function () { self._drawEmgChart(); }, 1000);
      }
    }, 250);
  },

  openHeartChart: function () {
    this.setData({ showHeartChart: true });
    // Immediately clear old canvas content
    this._clearHeartCanvas();
    var self = this;
    setTimeout(function () { self._drawHeartChart(); }, 200);
    // 实时刷新定时器
    if (this._heartChartTimer) clearInterval(this._heartChartTimer);
    this._heartChartTimer = setInterval(function () { self._drawHeartChart(); }, 1000);
  },
  closeHeartChart: function () {
    this.setData({ showHeartChart: false });
    if (this._heartChartTimer) { clearInterval(this._heartChartTimer); this._heartChartTimer = null; }
  },

  openEmgChart: function () {
    this.setData({ showEmgChart: true });
    // Sync synchronous muscle tracker from data (data may have default 'biceps')
    this._currentEmgMuscle = this.data.emgChartMuscle || 'biceps';
    // Immediately clear old canvas content
    this._clearEmgCanvas();
    var self = this;
    setTimeout(function () { self._drawEmgChart(); }, 200);
    if (this._emgChartTimer) clearInterval(this._emgChartTimer);
    this._emgChartTimer = setInterval(function () { self._drawEmgChart(); }, 1000);
  },
  closeEmgChart: function () {
    this.setData({ showEmgChart: false });
    if (this._emgChartTimer) { clearInterval(this._emgChartTimer); this._emgChartTimer = null; }
  },

  stopProp: function () {}, // prevent tap-through on modal

  /** Immediately clear the EMG waveform canvas (used on muscle switch) */
  _clearEmgCanvas: function () {
    var query = wx.createSelectorQuery();
    query.select('#emgChartCanvas').fields({ node: true, size: true }).exec(function (res) {
      if (!res || !res[0] || !res[0].node) return;
      var ctx = res[0].node.getContext('2d');
      var dpr = wx.getSystemInfoSync().pixelRatio;
      res[0].node.width = res[0].width * dpr;
      res[0].node.height = res[0].height * dpr;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, res[0].width, res[0].height);
    });
  },

  /** Immediately clear the heart waveform canvas */
  _clearHeartCanvas: function () {
    var query = wx.createSelectorQuery();
    query.select('#heartChartCanvas').fields({ node: true, size: true }).exec(function (res) {
      if (!res || !res[0] || !res[0].node) return;
      var ctx = res[0].node.getContext('2d');
      var dpr = wx.getSystemInfoSync().pixelRatio;
      res[0].node.width = res[0].width * dpr;
      res[0].node.height = res[0].height * dpr;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, res[0].width, res[0].height);
    });
  },

  _drawHeartChart: function () {
    var self = this;
    var query = wx.createSelectorQuery();
    query.select('#heartChartCanvas').fields({ node: true, size: true }).exec(function (res) {
      if (!res || !res[0] || !res[0].node) return;
      var canvas = res[0].node;
      var ctx = canvas.getContext('2d');
      var dpr = wx.getSystemInfoSync().pixelRatio;
      var w = res[0].width;
      var h = res[0].height;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.scale(dpr, dpr);

      var data = self._bpmHistory;
      if (data.length < 2) { ctx.fillText('数据不足', w / 2 - 20, h / 2); return; }

      var now = Date.now();
      var recent = data.filter(function (d) { return now - d.t < 35000; });
      if (recent.length < 2) { recent = data.slice(-30); }
      if (recent.length < 2) return;

      var vals = recent.map(function (d) { return d.val; });
      var maxV = Math.max(200, Math.max.apply(null, vals));
      var minV = 0;
      var range = maxV - minV;
      var pad = 10;

      ctx.clearRect(0, 0, w, h);

      // ── Y-axis grid & ticks (0-200) ──
      var heartTicks = [0, 50, 100, 150, 200];
      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 0.5;
      ctx.fillStyle = '#95a5a6';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'left';
      for (var ti = 0; ti < heartTicks.length; ti++) {
        var tickVal = heartTicks[ti];
        var ty = pad + (1 - tickVal / 200) * (h - pad * 2);
        ctx.beginPath();
        ctx.moveTo(pad, ty);
        ctx.lineTo(w - pad, ty);
        ctx.stroke();
        ctx.fillText(tickVal, 2, ty + 3);
      }

      ctx.strokeStyle = '#e74c3c';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (var i = 0; i < recent.length; i++) {
        var x = pad + (i / Math.max(recent.length - 1, 1)) * (w - pad * 2);
        var y = pad + (1 - (recent[i].val - minV) / range) * (h - pad * 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // X-axis label
      ctx.fillStyle = '#95a5a6';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('30s', w - 4, h - 4);
    });
  },

  _drawEmgChart: function () {
    var self = this;
    var query = wx.createSelectorQuery();
    query.select('#emgChartCanvas').fields({ node: true, size: true }).exec(function (res) {
      if (!res || !res[0] || !res[0].node) return;
      var canvas = res[0].node;
      var ctx = canvas.getContext('2d');
      var dpr = wx.getSystemInfoSync().pixelRatio;
      var w = res[0].width;
      var h = res[0].height;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.scale(dpr, dpr);

      var muscle = self._currentEmgMuscle || self.data.emgChartMuscle || 'biceps';
      self._initMuscleHistory();
      var predData = self._predEmgHistory[muscle] || [];
      var realData = self._realEmgHistory[muscle] || [];
      var now = Date.now();
      var maxT = now;
      var minT = now - 30000;

      var allVals = [];
      var predPts = []; var realPts = [];
      for (var i = 0; i < predData.length; i++) { if (predData[i].t >= minT) predPts.push(predData[i]); allVals.push(predData[i].val); }
      for (var j = 0; j < realData.length; j++) { if (realData[j].t >= minT) realPts.push(realData[j]); allVals.push(realData[j].val); }

      if (predPts.length < 2 && realPts.length < 2) {
        ctx.fillText('数据不足', w / 2 - 20, h / 2); return;
      }

      var maxV = Math.max.apply(null, allVals) || 100;
      var minV = 0;  // force Y-axis to start from 0
      // Round maxV up to a nice number for grid ticks
      var tickStep = maxV > 800 ? 200 : (maxV > 400 ? 100 : (maxV > 200 ? 50 : 25));
      var niceMax = Math.ceil(maxV / tickStep) * tickStep;
      var range = Math.max(niceMax - minV, 50);
      var padX = 8;
      var topPad = 26, bottomPad = 6;
      var chartH = h - topPad - bottomPad;

      ctx.clearRect(0, 0, w, h);

      // ── Muscle name title ──
      var muscleNames = { biceps: '肱二头肌', triceps: '肱三头肌', brachioradialis: '肱桡肌' };
      var muscleTitle = muscleNames[muscle] || muscle;
      ctx.fillStyle = '#2c3e50';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(muscleTitle + ' · 预测 vs 实测', w / 2, topPad - 6);

      // ── Y-axis grid & ticks ──
      var emgTicks = [];
      for (var tv = 0; tv <= niceMax; tv += tickStep) { emgTicks.push(tv); }
      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 0.5;
      ctx.fillStyle = '#95a5a6';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'left';
      for (var ti2 = 0; ti2 < emgTicks.length; ti2++) {
        var tickV = emgTicks[ti2];
        var ty2 = topPad + (1 - tickV / range) * chartH;
        ctx.beginPath();
        ctx.moveTo(padX, ty2);
        ctx.lineTo(w - padX, ty2);
        ctx.stroke();
        ctx.fillText(tickV, 2, ty2 + 3);
      }

      var drawLine = function (pts, color, width) {
        if (pts.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        for (var i = 0; i < pts.length; i++) {
          var x = padX + ((pts[i].t - minT) / (maxT - minT)) * (w - padX * 2);
          var y = topPad + (1 - (pts[i].val - minV) / range) * chartH;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      };

      drawLine(predPts, '#0984e3', 2);  // 预测: 蓝线
      drawLine(realPts, '#e74c3c', 2);  // 实测: 红线

      // X-axis label
      ctx.fillStyle = '#95a5a6';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('30s', w - 4, h - 4);

      // legend
      ctx.fillStyle = '#0984e3';
      ctx.fillRect(w - 120, 4, 10, 2);
      ctx.fillStyle = '#2c3e50';
      ctx.textAlign = 'left';
      ctx.font = '10px sans-serif';
      ctx.fillText('预测', w - 106, 8);
      ctx.fillStyle = '#e74c3c';
      ctx.fillRect(w - 65, 4, 10, 2);
      ctx.fillStyle = '#2c3e50';
      ctx.fillText('实测', w - 51, 8);
    });
  },

  onUnload() {
    if (this._unsub) { this._unsub(); this._unsub = null; }
    if (this.data.mockTimer) clearInterval(this.data.mockTimer);
    if (this._heartChartTimer) { clearInterval(this._heartChartTimer); this._heartChartTimer = null; }
    if (this._emgChartTimer) { clearInterval(this._emgChartTimer); this._emgChartTimer = null; }
    if (this._muscleDrawTimer) { clearTimeout(this._muscleDrawTimer); this._muscleDrawTimer = null; }
    if (this._slowTimer) { clearInterval(this._slowTimer); this._slowTimer = null; }
    if (this.data.animTimer) cancelAnimationFrame(this.data.animTimer);
    wx.closeBluetoothAdapter();
  }
});
