// pages/patientDetail/patientDetail.js
var roleManager = require('../../utils/roleManager');

Page({
  data: {
    patientId: '',
    patientName: '',
    patient: null,
    records: [],
    tasks: [],
    now: Date.now(),
    activeTab: 'records', // 'records' | 'tasks'
    loading: true,
    publishTaskTitle: '',
    publishTaskDesc: '',
    publishAngleMin: 50,
    publishAngleMax: 130,
    publishReps: 10,
    publishDeadline: '',
    publishLoading: false
  },

  onLoad(options) {
    if (!roleManager.requireRole()) return;
    if (!roleManager.isDoctor()) {
      wx.showToast({ title: '仅医生可访问', icon: 'none' });
      wx.navigateBack();
      return;
    }

    var patientId = options.patientId || '';
    var patientName = decodeURIComponent(options.patientName || '');

    this.setData({ patientId: patientId, patientName: patientName });

    // Set default deadline
    var nextWeek = new Date();
    nextWeek.setDate(nextWeek.getDate() + 7);
    this.setData({
      publishDeadline: nextWeek.getFullYear() + '-' +
        String(nextWeek.getMonth() + 1).padStart(2, '0') + '-' +
        String(nextWeek.getDate()).padStart(2, '0')
    });
  },

  onShow() {
    this.setData({ now: Date.now() });
    if (this.data.patientId) {
      this.fetchPatientData();
    }
  },

  /** Fetch patient records and tasks */
  fetchPatientData() {
    var that = this;
    var db = wx.cloud.database();
    this.setData({ loading: true });

    // Fetch user profile
    db.collection('users')
      .where({ _openid: this.data.patientId })
      .get()
      .then(function (userRes) {
        var user = userRes.data.length > 0 ? userRes.data[0] : null;
        that.setData({ patient: user, patientName: user ? (user.customName || '患者') : that.data.patientName });
      })
      .catch(function () { /* ignore */ });

    // Fetch training records
    db.collection('training_records')
      .where({ _openid: this.data.patientId })
      .orderBy('date', 'desc')
      .limit(20)
      .get()
      .then(function (recRes) {
        var formatted = recRes.data.map(function (item) {
          var d = item.date;
          if (!(d instanceof Date)) d = new Date(d);
          var dStr = (d.getMonth() + 1) + '月' + d.getDate() + '日 ' +
            d.getHours() + ':' + (d.getMinutes() < 10 ? '0' + d.getMinutes() : d.getMinutes());
          return Object.assign({}, item, { dateDisplay: dStr });
        });
        that.setData({ records: formatted });
      })
      .catch(function () { /* ignore */ });

    // Fetch tasks — query by patientId only (userInfo._openid unreliable)
    db.collection('tasks')
      .where({ patientId: this.data.patientId })
      .orderBy('createdAt', 'desc')
      .get()
      .then(function (taskRes) {
        that.setData({ tasks: taskRes.data, loading: false });
      })
      .catch(function () {
        that.setData({ loading: false });
      });
  },

  /** Switch tab */
  switchTab(e) {
    var tab = e.currentTarget.dataset.tab;
    this.setData({ activeTab: tab });
  },

  /** Open task detail */
  openTaskDetail(e) {
    var taskId = e.currentTarget.dataset.taskid;
    wx.navigateTo({ url: '/pages/taskDetail/taskDetail?taskId=' + taskId });
  },

  /** Navigate to publish page with this patient pre-selected */
  showPublishTask() {
    wx.navigateTo({
      url: '/pages/taskPublish/taskPublish?patientId=' + this.data.patientId + '&patientName=' + encodeURIComponent(this.data.patientName || '')
    });
  }
});
