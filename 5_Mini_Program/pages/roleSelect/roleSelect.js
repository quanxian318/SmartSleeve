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
        var result = res.result || {};
        // 云端可能已有角色（幂等返回），以云端为准，避免本地与云端永久不一致
        var finalRole = result.role || role;
        app.globalData.role = finalRole;
        wx.setStorageSync('userRole', finalRole);
        if (finalRole === 'doctor' && result.inviteCode) {
          app.globalData.inviteCode = result.inviteCode;
          wx.setStorageSync('inviteCode', result.inviteCode);
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
