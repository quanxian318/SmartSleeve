var app = getApp();
var dataManager = require('../../utils/dataManager');
var roleManager = require('../../utils/roleManager');

Page({
  data: {
    userInfo: null,
    records: [],
    loading: true,
    isEditing: false,
    tempName: '',
    showAiResult: false,
    aiResultText: '',

    // 角色
    role: '',
    isDoctor: false,
    isPatient: false,

    // 身体数据
    isEditingProfile: false,
    gender: 0,
    age: 30,
    height: 175,
    weight: 70,
    bmi: 22.9,
    tempGender: 0,
    tempAge: 30,
    tempHeight: 175,
    tempWeight: 70,
    tempBmi: '22.9',

    // 患者绑定医生
    doctorName: '',
    doctorId: '',
    showBindInput: false,
    inviteCodeInput: '',
    bindLoading: false,

    // 医生统计
    doctorStats: { patientCount: 0, taskCount: 0 },
    inviteCode: '',
    showInviteCode: true
  },

  onLoad() {
    this.setData({
      role: app.globalData.role || '',
      isDoctor: app.globalData.role === 'doctor',
      isPatient: app.globalData.role === 'patient'
    });
    this.autoLogin();
  },

  onShow() {
    this.setData({
      role: app.globalData.role || '',
      isDoctor: app.globalData.role === 'doctor',
      isPatient: app.globalData.role === 'patient'
    });

    if (roleManager.isDoctor()) {
      this.fetchDoctorStats();
    } else if (roleManager.isPatient()) {
      this.fetchRecords();
      this.setData({
        doctorName: app.globalData.doctorName || wx.getStorageSync('doctorName') || '',
        doctorId: app.globalData.doctorId || wx.getStorageSync('doctorId') || ''
      });
    }
  },

  // ================= 1. 身份识别逻辑 =================
  autoLogin() {
    var self = this;
    var db = wx.cloud.database();
    db.collection('users').where({ _openid: '{openid}' }).get({
      success: function (res) {
        if (res.data.length > 0) {
          var user = { nickName: res.data[0].customName };
          self.setData({ userInfo: user, loading: false });
          app.globalData.userInfo = user;

          // 加载身体数据
          var d = res.data[0];
          var gender = d.gender !== undefined ? d.gender : 0;
          var age = d.age || 30;
          var height = d.height || 175;
          var weight = d.weight || 70;
          var bmi = self.calcBMI(height, weight);
          self.setData({ gender: gender, age: age, height: height, weight: weight, bmi: bmi });
          self._syncProfile(gender, weight, height);
        } else {
          self.createNewUser(db);
        }
      },
      fail: function (err) {
        console.error('数据库连接失败', err);
        self.setData({ loading: false });
      }
    });
  },

  createNewUser(db) {
    var defaultName = '微信用户';
    db.collection('users').add({
      data: {
        customName: defaultName,
        createTime: new Date(),
        gender: 0,
        height: 175,
        weight: 70,
        bmi: 22.9
      },
      success: function () {
        var user = { nickName: defaultName };
        this.setData({ userInfo: user, loading: false });
        app.globalData.userInfo = user;
      }.bind(this)
    });
  },

  // ================= 2. 名字编辑逻辑 =================
  startEdit() {
    this.setData({ 
      isEditing: true, 
      tempName: this.data.userInfo.nickName 
    });
  },

  cancelEdit() {
    this.setData({ isEditing: false });
  },

  onNameInput(e) {
    this.setData({ tempName: e.detail.value });
  },

  saveNickname() {
    const newName = this.data.tempName.trim();
    if (!newName) return;
    const db = wx.cloud.database();
    wx.showLoading({ title: '保存中' });

    db.collection('users').where({
      _openid: '{openid}' 
    }).update({
      data: { customName: newName },
      success: () => {
        wx.hideLoading();
        this.setData({ 
          isEditing: false,
          'userInfo.nickName': newName
        });
        wx.showToast({ title: '修改成功' });
      },
      fail: err => {
        wx.hideLoading();
        wx.showToast({ title: '修改失败', icon: 'none' });
      }
    });
  },

  // ================= 3. 身体数据编辑 =================

  calcBMI: function (h, w) {
    if (!h || h <= 0) return 0;
    return parseFloat((w / Math.pow(h / 100, 2)).toFixed(1));
  },

  startEditProfile: function () {
    this.setData({
      isEditingProfile: true,
      tempGender: this.data.gender,
      tempAge: this.data.age,
      tempHeight: this.data.height,
      tempWeight: this.data.weight,
      tempBmi: this.calcBMI(this.data.height, this.data.weight).toFixed(1)
    });
  },

  cancelEditProfile: function () {
    this.setData({ isEditingProfile: false });
  },

  onAgeInput: function (e) {
    this.setData({ tempAge: Number(e.detail.value) || 30 });
  },

  onGenderSwitch: function (e) {
    var val = e.currentTarget.dataset.gender;
    this.setData({ tempGender: val });
  },

  onHeightInput: function (e) {
    var h = Number(e.detail.value) || 0;
    var w = this.data.tempWeight;
    var bmiStr = (h > 0 && w > 0) ? this.calcBMI(h, w).toFixed(1) : '--';
    this.setData({ tempHeight: h, tempBmi: bmiStr });
  },

  onWeightInput: function (e) {
    var w = Number(e.detail.value) || 0;
    var h = this.data.tempHeight;
    var bmiStr = (h > 0 && w > 0) ? this.calcBMI(h, w).toFixed(1) : '--';
    this.setData({ tempWeight: w, tempBmi: bmiStr });
  },

  saveProfile: function () {
    var self = this;
    var gender = this.data.tempGender;
    var age = this.data.tempAge;
    var height = this.data.tempHeight;
    var weight = this.data.tempWeight;

    if (!age || age < 5 || age > 120) {
      wx.showToast({ title: '请输入合理年龄(5-120)', icon: 'none' });
      return;
    }
    if (!height || height < 50 || height > 250) {
      wx.showToast({ title: '请输入合理身高(50-250cm)', icon: 'none' });
      return;
    }
    if (!weight || weight < 20 || weight > 200) {
      wx.showToast({ title: '请输入合理体重(20-200kg)', icon: 'none' });
      return;
    }

    var bmi = this.calcBMI(height, weight);
    wx.showLoading({ title: '保存中' });

    var db = wx.cloud.database();
    db.collection('users').where({ _openid: '{openid}' }).update({
      data: { gender: gender, age: age, height: height, weight: weight, bmi: bmi },
      success: function () {
        wx.hideLoading();
        self.setData({
          isEditingProfile: false,
          gender: gender,
          age: age,
          height: height,
          weight: weight,
          bmi: bmi
        });
        self._syncProfile(gender, weight, height);
        wx.showToast({ title: '身体数据已同步', icon: 'success' });
      },
      fail: function (err) {
        wx.hideLoading();
        wx.showToast({ title: '保存失败', icon: 'none' });
      }
    });
  },

  _syncProfile: function (gender, weight, height) {
    var sensorData = dataManager.getSensorData();
    var actionId = sensorData.actionId || 19;
    dataManager.setUserProfile(gender, weight, height, actionId);
  },

  // ================= 4. AI 诊断逻辑 =================
  // 在 history.js 中增加/修改这些逻辑
getAiDiagnosis() {
  if (this.data.records.length < 2) {
    wx.showToast({ title: '数据不足，请多做几次锻炼吧', icon: 'none' });
    return;
  }

  const latest = this.data.records[0]; // 最近一次
  const older = this.data.records[this.data.records.length - 1]; // 最早一次
  
  let diagnosis = "";
  
  // 1. 关节活动度评估 (ROM)
  const angleDiff = latest.visionAngle - older.visionAngle;
  if (angleDiff > 10) {
    diagnosis += `📈 进步明显：您的关节活动度提升了 ${angleDiff}度。手臂伸展能力显著增强。\n\n`;
  } else {
    diagnosis += `⚖️ 状态稳定：目前的关节活动范围与初期基本持平，建议增加拉伸尝试。\n\n`;
  }

  // 2. 肌肉力量评估 (EMG)
  const emgDiff = latest.maxEmg - older.maxEmg;
  if (emgDiff > 200) {
    diagnosis += `💪 力量增强：肌电信号峰值有所提升，说明受损肌肉的神经募集能力正在恢复。\n\n`;
  }

  // 3. 协调性评价 (Sleeve + Vision 融合评价)
  if (latest.maxEmg > 1000 && latest.visionAngle < 120) {
    diagnosis += `⚠️ 姿态提醒：检测到您在小角度时发力过猛，这可能是“代偿性发力”，请注意保持肩膀放松。\n\n`;
  }

  // 4. 综合结语
  diagnosis += `👨‍⚕️ 医生结语：坚持是康复的关键。目前的趋势显示您的恢复处于${angleDiff > 5 ? '上升期' : '平稳期'}。继续加油！`;

  this.setData({
    showAiResult: true,
    aiResultText: diagnosis
  });
},

closeAiModal() {
  this.setData({ showAiResult: false });
},

/** Prevent tap-through on modal (used by catchtap in wxml) */
stop() {},

// ================= 5. 医生统计 =================

fetchDoctorStats: function () {
  var self = this;
  self.setData({ inviteCode: app.globalData.inviteCode || '' });

  wx.cloud.callFunction({
    name: 'getDoctorPatients',
    data: {},
    success: function (res) {
      var result = res.result;
      if (result.success) {
        var patients = result.patients || [];
        var totalTasks = 0;
        patients.forEach(function (p) { totalTasks += p.totalTasks || 0; });
        self.setData({
          doctorStats: {
            patientCount: result.count || 0,
            taskCount: totalTasks
          }
        });
      }
    }
  });
},

copyDoctorInviteCode: function () {
  var code = this.data.inviteCode;
  if (!code) return;
  wx.setClipboardData({
    data: code,
    success: function () { wx.showToast({ title: '邀请码已复制', icon: 'success' }); }
  });
},

generateQRCode: function () {
  wx.showModal({
    title: '提示',
    content: '小程序码需要在微信云开发控制台通过 cloud.openapi.wxacode.getUnlimited 生成。生成后可将小程序码分享给患者扫码绑定。',
    showCancel: false
  });
},

refreshInviteCode: function () {
  var self = this;
  wx.showModal({
    title: '刷新邀请码',
    content: '刷新后旧邀请码将失效，确定吗？',
    success: function (res) {
      if (res.confirm) {
        wx.cloud.callFunction({
          name: 'setUserRole',
          data: { role: 'doctor' },
          success: function (cfRes) {
            var result = cfRes.result;
            if (result.success && result.inviteCode) {
              app.globalData.inviteCode = result.inviteCode;
              wx.setStorageSync('inviteCode', result.inviteCode);
              self.setData({ inviteCode: result.inviteCode });
              wx.showToast({ title: '邀请码已刷新', icon: 'success' });
            }
          }
        });
      }
    }
  });
},

// ================= 6. 患者：绑定医生 =================

toggleBindInput: function () {
  this.setData({ showBindInput: !this.data.showBindInput, inviteCodeInput: '' });
},

onInviteCodeInput: function (e) {
  this.setData({ inviteCodeInput: e.detail.value.toUpperCase() });
},

bindToDoctor: function () {
  var self = this;
  var code = this.data.inviteCodeInput.trim();

  if (!code || code.length !== 6) {
    wx.showToast({ title: '请输入6位邀请码', icon: 'none' });
    return;
  }

  this.setData({ bindLoading: true });

  wx.cloud.callFunction({
    name: 'joinDoctor',
    data: { inviteCode: code },
    success: function (res) {
      self.setData({ bindLoading: false });
      var result = res.result;
      if (result.success) {
        app.globalData.doctorId = result.doctorId;
        app.globalData.doctorName = result.doctorName;
        wx.setStorageSync('doctorId', result.doctorId);
        wx.setStorageSync('doctorName', result.doctorName);
        self.setData({
          doctorId: result.doctorId,
          doctorName: result.doctorName,
          showBindInput: false,
          inviteCodeInput: ''
        });
        wx.showToast({ title: '绑定成功！', icon: 'success' });
      } else {
        wx.showToast({ title: result.error || '绑定失败', icon: 'none' });
      }
    },
    fail: function () {
      self.setData({ bindLoading: false });
      wx.showToast({ title: '网络错误', icon: 'none' });
    }
  });
},

// ================= 7. 患者：任务入口 =================

navigateToMyTasks: function () {
  if (roleManager.isPatient()) {
    app.globalData._openTasksTab = true;
    wx.switchTab({ url: '/pages/index/index' });
  } else if (roleManager.isDoctor()) {
    wx.switchTab({ url: '/pages/digitalTwin/digitalTwin' });
  }
},

  // ================= 4. 历史记录获取 =================
  fetchRecords() {
    const db = wx.cloud.database();
    this.setData({ loading: true });

    db.collection('training_records')
      .where({ _openid: '{openid}' })
      .orderBy('date', 'desc')
      .get({
        success: res => {
          const formatted = res.data.map(item => {
            let d = item.date;
            if (!(d instanceof Date)) d = new Date(d);
            const dStr = `${d.getMonth() + 1}月${d.getDate()}日 ${d.getHours()}:${d.getMinutes() < 10 ? '0'+d.getMinutes() : d.getMinutes()}`;
            return { ...item, dateDisplay: dStr };
          });
          
          this.setData({ records: formatted, loading: false });
        },
        fail: err => {
          this.setData({ loading: false });
          console.error("数据库查询失败", err);
        }
      });
  }
})