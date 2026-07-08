var app = getApp();

Page({
  data: {
    patients: [],
    picked: -1,
    title: '',
    desc: '',
    angMin: 60,
    angMax: 140,
    reps: 10,
    deadline: '',
    sending: false,

    // Action templates
    templates: [],
    pickedTemplate: null,
    showTemplates: false
  },

  onLoad: function (opts) {
    var d = new Date();
    d.setDate(d.getDate() + 7);
    this.setData({
      deadline: d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate())
    });

    // Pre-select patient if passed via URL
    if (opts && opts.patientId) {
      this.setData({
        preSelectedId: opts.patientId,
        preSelectedName: decodeURIComponent(opts.patientName || '')
      });
    }

    this.loadPatients();
    this.loadTemplates();
  },

  /** Load doctor's action templates */
  loadTemplates: function () {
    var self = this;
    wx.cloud.callFunction({
      name: 'actionTemplates',
      data: { action: 'list' },
      success: function (res) {
        if (res.result && res.result.success) {
          self.setData({ templates: res.result.templates || [] });
        }
      },
      fail: function () {}
    });
  },

  /** Pick a standard action template */
  pickTemplate: function (e) {
    var idx = parseInt(e.currentTarget.dataset.idx);
    var tpl = this.data.templates[idx];
    if (!tpl) return;
    // Auto-fill task fields from template
    this.setData({
      pickedTemplate: tpl,
      showTemplates: false,
      title: tpl.actionName,
      angMin: tpl.targetAngleMin || 60,
      angMax: tpl.targetAngleMax || 140,
      reps: tpl.repetitions || 10
    });
  },

  /** Clear selected template */
  clearTemplate: function () {
    this.setData({ pickedTemplate: null });
  },

  toggleTemplates: function () {
    this.setData({ showTemplates: !this.data.showTemplates });
  },

  loadPatients: function () {
    var self = this;
    wx.cloud.callFunction({
      name: 'getDoctorPatients',
      data: {},
      success: function (res) {
        if (res.result && res.result.success) {
          self.setPatients(res.result.patients || []);
        } else {
          self.loadPatientsDB();
        }
      },
      fail: function () { self.loadPatientsDB(); }
    });
  },

  loadPatientsDB: function () {
    var self = this;
    wx.cloud.database().collection('users')
      .where({ doctorId: app.globalData.doctorId || (app.globalData.userInfo ? app.globalData.userInfo._openid : '') })
      .field({ _openid: true, customName: true })
      .get()
      .then(function (res) {
        self.setPatients((res.data || []).map(function (p) {
          return { _openid: p._openid, customName: p.customName || '患者' };
        }));
      })
      .catch(function () {});
  },

  /** Set patient list and auto-pick pre-selected patient */
  setPatients: function (list) {
    var picked = -1;
    if (this.data.preSelectedId) {
      for (var i = 0; i < list.length; i++) {
        if (list[i]._openid === this.data.preSelectedId) { picked = i; break; }
      }
    }
    this.setData({ patients: list, picked: picked });
  },

  pickPatient: function (e) {
    var i = parseInt(e.currentTarget.dataset.i);
    this.setData({ picked: this.data.picked === i ? -1 : i });
  },

  submit: function () {
    var d = this.data;
    if (d.picked < 0) { wx.showToast({ title: '请选择学员', icon: 'none' }); return; }
    if (!d.title.trim()) { wx.showToast({ title: '请输入任务标题', icon: 'none' }); return; }

    this.setData({ sending: true });
    var self = this;
    var p = d.patients[d.picked];

    var task = {
      patientId: p._openid,
      patientName: p.customName || '患者',
      title: d.title.trim(),
      description: d.desc.trim(),
      targetAngleMin: Number(d.angMin),
      targetAngleMax: Number(d.angMax),
      repetitions: Number(d.reps),
      deadline: d.deadline || null,
      status: 'pending',
      createdAt: new Date(),
      templateId: d.pickedTemplate ? d.pickedTemplate._id : null
    };

    wx.cloud.callFunction({
      name: 'manageTask',
      data: { action: 'create', data: task },
      success: function (res) {
        self.setData({ sending: false });
        if (res.result && res.result.success) {
          wx.showToast({ title: '已发布', icon: 'success' });
          setTimeout(function () { wx.navigateBack(); }, 800);
        } else {
          self.directAdd(task);
        }
      },
      fail: function () { self.setData({ sending: false }); self.directAdd(task); }
    });
  },

  directAdd: function (task) {
    // Ensure templateId is included in direct fallback
    if (this.data.pickedTemplate && !task.templateId) {
      task.templateId = this.data.pickedTemplate._id;
    }
    var self = this;
    wx.cloud.database().collection('tasks').add({ data: task }).then(function () {
      self.setData({ sending: false });
      wx.showToast({ title: '已发布', icon: 'success' });
      setTimeout(function () { wx.navigateBack(); }, 800);
    }).catch(function (e) {
      self.setData({ sending: false });
      wx.showToast({ title: '失败: ' + (e.errMsg || ''), icon: 'none' });
    });
  },

  onTitle: function (e) { this.setData({ title: e.detail.value }); },
  onDesc: function (e) { this.setData({ desc: e.detail.value }); },
  onDeadline: function (e) { this.setData({ deadline: e.detail.value }); },
  step: function (e) {
    var k = e.currentTarget.dataset.k;
    var d = parseInt(e.currentTarget.dataset.d);
    var v = Number(this.data[k]) + d;
    this.setData({ [k]: v });
  }
});

function pad(n) { return n < 10 ? '0' + n : '' + n; }
