var app = getApp();
var roleManager = require('../../utils/roleManager');

Page({
  data: {
    selectedRole: '',
    loading: false
  },

  onLoad() {
    if (roleManager.hasRole()) {
      this._goHome();
    }
  },

  selectRole(e) {
    this.setData({ selectedRole: e.currentTarget.dataset.role });
  },

  confirmRole() {
    var role = this.data.selectedRole;
    if (!role) { wx.showToast({ title: '请先选择您的身份', icon: 'none' }); return; }

    app.globalData.role = role;
    wx.setStorageSync('userRole', role);

    var that = this;
    this.setData({ loading: true });
    wx.cloud.callFunction({
      name: 'setUserRole',
      data: { role: role },
      success: function (res) {
        that.setData({ loading: false });
        if (role === 'doctor' && res.result && res.result.inviteCode) {
          app.globalData.inviteCode = res.result.inviteCode;
          wx.setStorageSync('inviteCode', res.result.inviteCode);
        }
        that._goHome();
      },
      fail: function (err) {
        that.setData({ loading: false });
        that._goHome();
      }
    });
  },

  goHome: function () {
    wx.switchTab({ url: '/pages/index/index' });
  },

  _goHome: function () {
    wx.switchTab({ url: '/pages/index/index' });
  }
});
