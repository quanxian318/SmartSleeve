/**
 * Role management utility.
 *
 * Centralized role logic shared across all pages.
 * Reads role from app.globalData, synced from cloud DB on launch.
 *
 * Usage:
 *   const rm = require('../../utils/roleManager');
 *   if (rm.isDoctor()) { ... }
 *   rm.setupTabBar('doctor');
 */

var roleManager = {
  /** Get App instance safely (lazy — handles early require before App() is ready) */
  _app: function () {
    return getApp();
  },

  /** Get current user role from globalData */
  getRole: function () {
    var a = getApp(); if (!a || !a.globalData) return null;
    return a.globalData.role || null;
  },

  /** Check if current user is a doctor */
  isDoctor: function () {
    var a = getApp();
    return a && a.globalData && a.globalData.role === 'doctor';
  },

  /** Check if current user is a patient */
  isPatient: function () {
    var a = getApp();
    return a && a.globalData && a.globalData.role === 'patient';
  },

  /** Check if role has been set */
  hasRole: function () {
    var a = getApp();
    var stored = wx.getStorageSync('userRole');
    return (a && a.globalData && !!a.globalData.role) || !!stored;
  },

  /** Get doctor's invite code */
  getInviteCode: function () {
    var a = getApp();
    return (a && a.globalData) ? (a.globalData.inviteCode || null) : null;
  },

  /** Get patient's bound doctor id */
  getDoctorId: function () {
    var a = getApp();
    return (a && a.globalData) ? (a.globalData.doctorId || null) : null;
  },

  /** Get patient's bound doctor name */
  getDoctorName: function () {
    var a = getApp();
    return (a && a.globalData) ? (a.globalData.doctorName || null) : null;
  },

  /** Persist role and related data to both globalData and local storage. */
  setRoleData: function (role, extra) {
    var app = getApp();
    if (app && app.globalData) {
      app.globalData.role = role;
      if (extra) {
        if (extra.inviteCode !== undefined) app.globalData.inviteCode = extra.inviteCode;
        if (extra.doctorId !== undefined) app.globalData.doctorId = extra.doctorId;
        if (extra.doctorName !== undefined) app.globalData.doctorName = extra.doctorName;
      }
    }
    wx.setStorageSync('userRole', role);
    if (extra) {
      if (extra.inviteCode !== undefined) wx.setStorageSync('inviteCode', extra.inviteCode);
      if (extra.doctorId !== undefined) wx.setStorageSync('doctorId', extra.doctorId);
      if (extra.doctorName !== undefined) wx.setStorageSync('doctorName', extra.doctorName);
    }
  },

  /** Restore role data from local storage (used on cold start before cloud sync) */
  restoreFromStorage: function () {
    var app = getApp();
    var role = wx.getStorageSync('userRole');
    if (role && app && app.globalData) {
      app.globalData.role = role;
      app.globalData.inviteCode = wx.getStorageSync('inviteCode') || null;
      app.globalData.doctorId = wx.getStorageSync('doctorId') || null;
      app.globalData.doctorName = wx.getStorageSync('doctorName') || null;
    }
  },

  /** Navigate to the appropriate home page for the current role. */
  navigateToRoleHome: function () {
    wx.switchTab({ url: '/pages/index/index' });
  },

  /** Dynamically configure the tab bar for the current role. */
  setupTabBar: function (role) {
    if (role === 'doctor') {
      wx.setTabBarItem({ index: 0, text: '患者管理' });
      wx.setTabBarItem({ index: 1, text: '任务中心' });
      wx.setTabBarItem({ index: 2, text: '我的' });
    } else {
      wx.setTabBarItem({ index: 0, text: '实时监控' });
      wx.setTabBarItem({ index: 1, text: '数字孪生' });
      wx.setTabBarItem({ index: 2, text: '个人中心' });
    }
  },

  /** Page guard — call in onLoad of role-sensitive pages. */
  requireRole: function () {
    var app = getApp();
    var stored = wx.getStorageSync('userRole');
    var hasRole = (app && app.globalData && !!app.globalData.role) || !!stored;
    if (!hasRole) {
      wx.reLaunch({ url: '/pages/roleSelect/roleSelect' });
      return false;
    }
    // If globalData lost the role but storage has it, restore
    if (app && app.globalData && !app.globalData.role && stored) {
      app.globalData.role = stored;
    }
    return true;
  }
};

module.exports = roleManager;
