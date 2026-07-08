// pages/taskDetail/taskDetail.js
var roleManager = require('../../utils/roleManager');
var dataManager = require('../../utils/dataManager');
var scorer = require('../../utils/scorer');

var app = getApp();

Page({
  data: {
    role: '',
    taskId: '',
    task: null,
    loading: true,
    completing: false,
    deadlineText: '',
    isOverdue: false,

    // Training state
    training: false,
    currentRep: 0,
    currentAngle: 0,
    inTargetZone: false,
    repStarted: false,
    trainingBpm: '--',
    trainingEmg: 0,
    emgPercent: 0,

    // Angle gauge
    gaugeRotation: 0,

    // Standard template for scoring
    standardTemplate: null
  },

  onLoad: function (options) {
    if (!roleManager.requireRole()) return;

    var taskId = options.taskId || '';
    this.setData({
      taskId: taskId,
      role: app.globalData.role
    });

    if (taskId) {
      this.fetchTask();
    }
  },

  onShow: function () {
    // Refresh task data when returning to this page
    if (this.data.taskId) {
      this.fetchTask();
    }

    // Restore training UI if training is active on another page
    var ts = app.globalData.trainingState;
    if (ts && ts.active && ts.taskId === this.data.taskId) {
      this.setData({
        training: true,
        currentRep: ts.currentRep || 0,
        currentAngle: 0,
        inTargetZone: false,
        repStarted: true
      });
      // Re-subscribe if needed
      if (!this._unsubSensor) {
        this._reattachSensorForTraining();
      }
    }
  },

  onUnload: function () {
    this.stopTraining();
  },

  onHide: function () {
    // Keep training running — do NOT stop
  },

  /** Fetch task details */
  fetchTask: function () {
    var that = this;
    this.setData({ loading: true });

    var db = wx.cloud.database();
    db.collection('tasks')
      .doc(this.data.taskId)
      .get()
      .then(function (res) {
        if (!res || !res.data) {
          that.setData({ loading: false });
          wx.showToast({ title: '任务不存在或已删除', icon: 'none' });
          setTimeout(function () { wx.navigateBack(); }, 1500);
          return;
        }
        var task = res.data;
        // Compute deadline display
        var deadlineText = '';
        var isOverdue = false;
        if (task.deadline) {
          var dl = new Date(task.deadline);
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

        that.setData({
          task: task,
          loading: false,
          deadlineText: deadlineText,
          isOverdue: isOverdue
        });

        // Load standard template if task has one
        if (task.templateId) {
          that._loadTemplate(task.templateId);
        }
      })
      .catch(function (err) {
        console.error('[taskDetail] Fetch error:', err);
        that.setData({ loading: false });
        wx.showToast({ title: '加载失败', icon: 'none' });
      });
  },

  /** Load standard action template for scoring comparison */
  _loadTemplate: function (templateId) {
    var that = this;
    wx.cloud.callFunction({
      name: 'actionTemplates',
      data: { action: 'get', templateId: templateId },
      success: function (res) {
        if (res.result && res.result.success && res.result.template) {
          that.setData({ standardTemplate: res.result.template });
        }
      },
      fail: function () {}
    });
  },

  // ============= Training Mode =============

  /** Start training on digital twin (3D visual feedback) */
  startTraining: function () {
    var task = this.data.task;
    if (!task) return;

    // Store active task and init training state
    app.globalData.activeTask = {
      _id: task._id,
      title: task.title,
      targetAngleMin: task.targetAngleMin,
      targetAngleMax: task.targetAngleMax,
      repetitions: task.repetitions,
      templateId: task.templateId || null
    };

    // Init shared training state with autoStart
    app.globalData.trainingState = {
      taskId: task._id,
      active: false,
      autoStart: true,
      currentRep: 0,
      startedAt: Date.now()
    };

    wx.switchTab({ url: '/pages/digitalTwin/digitalTwin' });
  },

  /** Quick training on this page (number-only, no 3D) */
  startQuickTraining: function () {
    var task = this.data.task;
    if (!task) return;

    app.globalData.activeTask = {
      _id: task._id,
      title: task.title,
      targetAngleMin: task.targetAngleMin,
      targetAngleMax: task.targetAngleMax,
      repetitions: task.repetitions,
      templateId: task.templateId || null
    };

    // Init shared training state
    app.globalData.trainingState = {
      taskId: task._id,
      active: true,
      autoStart: false,
      currentRep: 0,
      startedAt: Date.now()
    };

    var that = this;
    var targetMin = Number(task.targetAngleMin);
    var targetMax = Number(task.targetAngleMax);

    this.setData({
      training: true,
      currentRep: 0,
      currentAngle: 0,
      inTargetZone: false,
      repStarted: false,
      trainingBpm: '--',
      trainingEmg: 0,
      emgPercent: 0
    });

    this._unsubSensor = dataManager.subscribe(function (sensorData) {
      var angle = sensorData.rdkAngle || 180;
      var bpm = sensorData.bpm || 0;
      var emg = sensorData.emg || 0;
      var inZone = angle >= targetMin && angle <= targetMax;
      var wasInZone = that.data.inTargetZone;

      that.setData({
        currentAngle: angle,
        inTargetZone: inZone,
        trainingBpm: bpm || '--',
        trainingEmg: emg,
        emgPercent: sensorData.emgPercent || 0
      });

      if (inZone && !wasInZone && that.data.repStarted) {
        var newRep = that.data.currentRep + 1;
        that.setData({ currentRep: newRep, repStarted: false });
        // Sync to shared state
        if (app.globalData.trainingState) {
          app.globalData.trainingState.currentRep = newRep;
        }
        wx.vibrateShort({ type: 'light' });
        if (newRep >= task.repetitions) {
          that.setData({ training: false });
          if (app.globalData.trainingState) {
            app.globalData.trainingState.active = false;
          }
          var scoreResult = scorer.score({
            maxEmg: sensorData.emg || 0,
            completedReps: newRep,
            targetReps: task.repetitions,
            angleMin: that.data._trainingAngleMin || targetMin,
            angleMax: that.data._trainingAngleMax || targetMax,
            angleCurve: that.data._trainingAngleCurve || [],
            taskAngleMin: targetMin,
            taskAngleMax: targetMax
          }, that.data.standardTemplate);

          wx.showModal({
            title: '🎉 训练完成',
            content: '全部 ' + task.repetitions + ' 次已完成！\n\n综合评分: ' + scoreResult.total + '分 ' + scoreResult.level.emoji + '\n' + scoreResult.level.text,
            confirmText: '标记完成',
            cancelText: '继续训练',
            success: function (modalRes) {
              if (modalRes.confirm) { that.completeTask(scoreResult); }
              else { that.setData({ training: true }); }
            }
          });
        }
      } else if (!inZone && wasInZone) {
        that.setData({ repStarted: true });
      } else if (inZone && !wasInZone && !that.data.repStarted && that.data.currentRep === 0) {
        that.setData({ repStarted: true });
      }

      // Track angle range
      if (!that.data._trainingAngleCurve) that.data._trainingAngleCurve = [];
      that.data._trainingAngleCurve.push(angle);
      if (that.data._trainingAngleMin === undefined || angle < that.data._trainingAngleMin) {
        that.data._trainingAngleMin = angle;
      }
      if (that.data._trainingAngleMax === undefined || angle > that.data._trainingAngleMax) {
        that.data._trainingAngleMax = angle;
      }
    });

    wx.showToast({ title: '快速训练开始', icon: 'success', duration: 1200 });
  },

  /** Re-attach sensor subscription for training (when returning from another tab) */
  _reattachSensorForTraining: function () {
    var that = this;
    var task = this.data.task;
    if (!task || this._unsubSensor) return;
    var targetMin = Number(task.targetAngleMin);
    var targetMax = Number(task.targetAngleMax);

    this._unsubSensor = dataManager.subscribe(function (sensorData) {
      var angle = sensorData.rdkAngle || 180;
      var inZone = angle >= targetMin && angle <= targetMax;
      var wasInZone = that.data.inTargetZone;

      that.setData({
        currentAngle: angle,
        inTargetZone: inZone,
        trainingBpm: sensorData.bpm || '--',
        trainingEmg: sensorData.emg || 0,
        emgPercent: sensorData.emgPercent || 0
      });

      if (inZone && !wasInZone && that.data.repStarted) {
        var newRep = (that.data.currentRep || 0) + 1;
        that.setData({ currentRep: newRep, repStarted: false });
        if (app.globalData.trainingState) {
          app.globalData.trainingState.currentRep = newRep;
        }
        wx.vibrateShort({ type: 'light' });
        if (newRep >= task.repetitions) {
          that.setData({ training: false });
          if (app.globalData.trainingState) {
            app.globalData.trainingState.active = false;
          }
          wx.showModal({
            title: '🎉 训练完成',
            content: '全部 ' + task.repetitions + ' 次已完成！是否标记完成？',
            confirmText: '标记完成',
            cancelText: '继续训练',
            success: function (modalRes) {
              if (modalRes.confirm) { that.completeTask(); }
              else { that.setData({ training: true }); }
            }
          });
        }
      } else if (!inZone && wasInZone) {
        that.setData({ repStarted: true });
      }
    });
  },

  /** Navigate to digital twin page with active task context */
  gotoDigitalTwin: function () {
    var task = this.data.task;
    if (!task) return;
    // Store active task on globalData so digitalTwin picks it up
    app.globalData.activeTask = {
      _id: task._id,
      title: task.title,
      targetAngleMin: task.targetAngleMin,
      targetAngleMax: task.targetAngleMax,
      repetitions: task.repetitions,
      templateId: task.templateId || null
    };
    // If training is NOT already active, set autoStart so twin starts fresh
    // If training IS active, twin will detect and resume
    if (!app.globalData.trainingState || !app.globalData.trainingState.active) {
      app.globalData.trainingState = {
        taskId: task._id,
        active: false,
        autoStart: true,
        currentRep: 0,
        startedAt: Date.now()
      };
    }
    wx.switchTab({ url: '/pages/digitalTwin/digitalTwin' });
  },

  /** Stop training and unsubscribe from sensor data */
  stopTraining: function () {
    var wasTraining = this.data.training;
    if (this._unsubSensor) {
      this._unsubSensor();
      this._unsubSensor = null;
    }
    this.setData({
      training: false,
      currentRep: 0,
      currentAngle: 0,
      inTargetZone: false,
      repStarted: false
    });
    // Clear shared training state
    if (app.globalData.trainingState) {
      app.globalData.trainingState.active = false;
      app.globalData.trainingState.currentRep = 0;
    }
    if (wasTraining) {
      wx.showToast({ title: '训练已停止', icon: 'none' });
    }
  },

  // ============= Task Actions =============

  /** Patient: manually mark task as complete */
  completeTask: function (scoreResult) {
    var that = this;

    var content = '请确认您已按照要求完成了训练动作。';
    if (scoreResult) {
      content = '综合评分: ' + scoreResult.total + '分 ' + scoreResult.level.emoji + '\n' + scoreResult.level.text + '\n\n' + content;
    }

    wx.showModal({
      title: '确认完成',
      content: content,
      success: function (modalRes) {
        if (!modalRes.confirm) return;

        // Stop training if running
        that.stopTraining();
        that.setData({ completing: true });

        // Get current sensor data snapshot
        var sensorData = dataManager.getSensorData();
        var completionData = {
          maxEmg: sensorData.emg || 0,
          avgAngle: sensorData.rdkAngle || 0,
          bpm: sensorData.bpm || 0,
          score: scoreResult || null
        };

        wx.cloud.callFunction({
          name: 'manageTask',
          data: {
            action: 'complete',
            taskId: that.data.taskId,
            data: completionData
          },
          success: function (res) {
            that.setData({ completing: false });
            var result = res.result;
            if (result.success) {
              // Clear active task
              app.globalData.activeTask = null;
              wx.showToast({ title: '任务完成！', icon: 'success' });
              that.fetchTask(); // refresh to show updated status
            } else {
              wx.showToast({ title: result.error || '操作失败', icon: 'none' });
            }
          },
          fail: function () {
            that.setData({ completing: false });
            wx.showToast({ title: '网络错误', icon: 'none' });
          }
        });
      }
    });
  },

  /** Navigate back */
  goBack: function () {
    wx.navigateBack();
  },

  /** Doctor: delete this task */
  deleteTask: function () {
    var that = this;

    wx.showModal({
      title: '确认删除',
      content: '删除后患者将无法看到此任务，确定要删除吗？',
      success: function (modalRes) {
        if (!modalRes.confirm) return;

        wx.cloud.callFunction({
          name: 'manageTask',
          data: {
            action: 'delete',
            taskId: that.data.taskId
          },
          success: function (res) {
            var result = res.result;
            if (result.success) {
              wx.showToast({ title: '已删除', icon: 'success' });
              setTimeout(function () { wx.navigateBack(); }, 800);
            } else {
              wx.showToast({ title: result.error || '删除失败', icon: 'none' });
            }
          },
          fail: function () {
            wx.showToast({ title: '网络错误', icon: 'none' });
          }
        });
      }
    });
  }
});
