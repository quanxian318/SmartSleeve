// app.js
var roleManager = require('./utils/roleManager');

App({
  onLaunch() {
    // 全局错误捕获 — 真机调试用
    if (typeof wx !== 'undefined' && wx.onError) {
      wx.onError(function (err) {
        console.error('[App] Global error:', err);
        wx.setStorageSync('_lastCrash', String(err).substring(0, 500));
      });
    }
    if (typeof wx !== 'undefined' && wx.onUnhandledRejection) {
      wx.onUnhandledRejection(function (res) {
        console.error('[App] Unhandled rejection:', res.reason);
        wx.setStorageSync('_lastCrash', String(res.reason).substring(0, 500));
      });
    }

    try {
      if (!wx.cloud) {
        console.error('请使用 2.2.3 或以上的基础库以使用云能力')
      } else {
        wx.cloud.init({
          env: 'cloud1-d7g8tpqsscb7f752a',
          traceUser: true,
        })
      }

      const logs = wx.getStorageSync('logs') || []
      logs.unshift(Date.now())
      wx.setStorageSync('logs', logs)

      // Restore role from local storage for fast cold start
      roleManager.restoreFromStorage();

      this.checkUserLogin();
      this.syncUserFromCloud();
    } catch (e) {
      console.error('[App] onLaunch error:', e.message, e.stack);
      wx.setStorageSync('_lastCrash', 'onLaunch: ' + e.message);
    }
  },

  /** Sync user role and profile from cloud DB on launch */
  syncUserFromCloud() {
    var self = this;
    var db = wx.cloud.database();
    db.collection('users').where({ _openid: '{openid}' }).get({
      success: function (res) {
        if (res.data.length > 0) {
          var user = res.data[0];
          self.globalData.userInfo = user;

          // Only overwrite role from cloud if local role is NOT set
          // (prevents async cloud sync from wiping freshly selected role)
          if (!self.globalData.role && user.role) {
            self.globalData.role = user.role;
            self.globalData.inviteCode = user.inviteCode || null;
            self.globalData.doctorId = user.doctorId || null;
            self.globalData.doctorName = user.doctorName || null;
            wx.setStorageSync('userRole', user.role);
            if (user.inviteCode) wx.setStorageSync('inviteCode', user.inviteCode);
            if (user.doctorId) wx.setStorageSync('doctorId', user.doctorId);
            if (user.doctorName) wx.setStorageSync('doctorName', user.doctorName);
          }

          // If role was set locally but cloud doesn't have it yet, push to cloud
          if (self.globalData.role && !user.role) {
            db.collection('users').where({ _openid: '{openid}' }).update({
              data: { role: self.globalData.role }
            }).catch(function () {});
          }

          // Sync age to sensor profile
          if (user.age) self.globalData.userAge = user.age;
          self.globalData.cloudSynced = true;
        } else {
          // New user — no users record yet, will be created by history page
        }
        self.globalData.cloudSynced = true;
      },
      fail: function (err) {
        console.error('[App] Cloud sync failed:', err);
        self.globalData.cloudSynced = false;
      }
    });
  },

  checkUserLogin() {
    wx.getSetting({
      success: res => {
        if (res.authSetting['scope.userInfo']) {
          wx.getUserInfo({
            success: res => {
              this.globalData.userInfo = res.userInfo
              if (this.userInfoReadyCallback) {
                this.userInfoReadyCallback(res)
              }
            }
          })
        }
      }
    })
  },

  globalData: {
    userInfo: null,
    // Role system
    role: null,          // 'doctor' | 'patient' | null
    inviteCode: null,    // doctor's invite code
    doctorId: null,      // patient's bound doctor openid
    doctorName: null,    // patient's bound doctor name
    needsRoleSelect: false, // true if user hasn't selected a role yet
    cloudSynced: false,
    // Device / sensor
    connectedDevice: null,
    sensorData: null,
    _sensorSubscribers: [],
    _socketTask: null,
    // Shared training state (synced between taskDetail and digitalTwin)
    trainingState: null
  }
})
