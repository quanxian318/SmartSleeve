var app = getApp();
var dataManager = require('../../utils/dataManager');

Page({
  data: {
    // Action selection
    actionTypes: [
      { key: 'elbow_flexion', name: '肘屈伸', icon: '💪', desc: '手臂从弯曲到伸直', min: 50, max: 140 },
      { key: 'shoulder_press', name: '肩推举', icon: '🏋️', desc: '手臂从下垂到上举', min: 30, max: 160 },
      { key: 'wrist_curl', name: '腕屈伸', icon: '🤲', desc: '手腕上下活动', min: 20, max: 80 },
      { key: 'forearm_rotation', name: '前臂旋转', icon: '🔄', desc: '前臂内外旋', min: 0, max: 120 },
      { key: 'custom', name: '自定义', icon: '✏️', desc: '自定义角度范围', min: 0, max: 180 }
    ],
    selectedAction: '',
    customName: '',

    // Recording
    recording: false,
    recordingTime: 0,
    recordingMax: 3,
    recAngle: 180,
    recEmg: 0,
    recBpm: '--',
    angleCurve: [],
    emgCurve: [],
    peakEmg: 0,
    angleMin: 180,
    angleMax: 0,

    // Result
    recorded: false,
    recordSummary: '',
    saving: false
  },

  onLoad: function () {
    this._recInterval = null;
    this._unsub = null;
  },

  onUnload: function () {
    this._stopRecording();
  },

  /** Select an action type */
  selectAction: function (e) {
    var key = e.currentTarget.dataset.key;
    this.setData({ selectedAction: key });
  },

  onCustomName: function (e) {
    this.setData({ customName: e.detail.value });
  },

  /** Start recording standard action */
  startRecord: function () {
    if (!this.data.selectedAction) {
      wx.showToast({ title: '请先选择动作类型', icon: 'none' });
      return;
    }
    if (this.data.selectedAction === 'custom' && !this.data.customName.trim()) {
      wx.showToast({ title: '请输入自定义动作名称', icon: 'none' });
      return;
    }

    var that = this;
    this.setData({
      recording: true,
      recordingTime: 0,
      recAngle: 180,
      recEmg: 0,
      recBpm: '--',
      angleCurve: [],
      emgCurve: [],
      peakEmg: 0,
      angleMin: 180,
      angleMax: 0,
      recorded: false
    });

    // Subscribe to sensor data
    this._unsub = dataManager.subscribe(function (data) {
      var angle = data.rdkAngle || 180;
      var emg = data.emg || 0;
      if (!that.data.recording) return;

      that.data.angleCurve.push(angle);
      that.data.emgCurve.push(emg);

      that.setData({
        recAngle: angle,
        recEmg: emg,
        recBpm: data.bpm || '--',
        peakEmg: Math.max(that.data.peakEmg, emg),
        angleMin: Math.min(that.data.angleMin, angle),
        angleMax: Math.max(that.data.angleMax, angle)
      });
    });

    // Countdown timer
    wx.showToast({ title: '3秒录制开始！', icon: 'none', duration: 1000 });
    this._recInterval = setInterval(function () {
      var t = that.data.recordingTime + 1;
      if (t >= that.data.recordingMax) {
        that._finishRecording();
      } else {
        that.setData({ recordingTime: t });
      }
    }, 1000);
  },

  /** Finish recording and show summary */
  _finishRecording: function () {
    if (this._recInterval) { clearInterval(this._recInterval); this._recInterval = null; }
    if (this._unsub) { this._unsub(); this._unsub = null; }

    var summary = '峰值EMG: ' + this.data.peakEmg + ' μV\n' +
      '角度范围: ' + this.data.angleMin + '° - ' + this.data.angleMax + '°\n' +
      '采样点数: ' + this.data.angleCurve.length;

    this.setData({
      recording: false,
      recorded: true,
      recordSummary: summary
    });

    wx.showToast({ title: '录制完成！', icon: 'success' });
  },

  _stopRecording: function () {
    if (this._recInterval) { clearInterval(this._recInterval); this._recInterval = null; }
    if (this._unsub) { this._unsub(); this._unsub = null; }
  },

  /** Save recorded template to cloud */
  saveTemplate: function () {
    var that = this;
    var action = this.data.actionTypes.find(function (a) { return a.key === that.data.selectedAction; });
    var name = this.data.selectedAction === 'custom'
      ? this.data.customName.trim()
      : (action ? action.name : '');

    if (!name) { wx.showToast({ title: '请输入动作名称', icon: 'none' }); return; }

    this.setData({ saving: true });

    var template = {
      actionName: name,
      actionType: this.data.selectedAction,
      targetAngleMin: action ? action.min : 0,
      targetAngleMax: action ? action.max : 180,
      standardEmgBiceps: this.data.peakEmg,
      standardEmgTriceps: Math.round(this.data.peakEmg * 0.35),
      standardAngleCurve: this.data.angleCurve,
      standardEmgCurve: this.data.emgCurve,
      repetitions: 10,
      createdAt: new Date(),
      // 补充 doctorId，防止降级写入生成孤儿数据
      doctorId: app.globalData.userInfo ? app.globalData.userInfo._openid : (wx.getStorageSync('userOpenid') || '')
    };

    wx.cloud.callFunction({
      name: 'actionTemplates',
      data: { action: 'save', data: template },
      success: function (res) {
        that.setData({ saving: false });
        if (res.result && res.result.success) {
          wx.showToast({ title: '模板已保存！', icon: 'success' });
          setTimeout(function () { wx.navigateBack(); }, 1000);
        } else {
          wx.showToast({ title: (res.result && res.result.error) || '保存失败', icon: 'none' });
        }
      },
      fail: function () {
        that.setData({ saving: false });
        // Fallback direct DB
        wx.cloud.database().collection('action_templates').add({ data: template }).then(function () {
          wx.showToast({ title: '模板已保存！', icon: 'success' });
          setTimeout(function () { wx.navigateBack(); }, 1000);
        }).catch(function (err) {
          wx.showToast({ title: '保存失败: ' + (err.errMsg || ''), icon: 'none' });
        });
      }
    });
  },

  /** Reset and record again */
  reRecord: function () {
    this._stopRecording();
    this.setData({
      recording: false,
      recorded: false,
      angleCurve: [],
      emgCurve: [],
      peakEmg: 0,
      angleMin: 180,
      angleMax: 0
    });
  }
});
