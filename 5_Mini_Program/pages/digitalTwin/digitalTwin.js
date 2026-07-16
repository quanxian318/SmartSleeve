/**
 * 数字孪生 - Digital Twin page
 *
 * 3D arm visualization with real-time muscle activation display.
 * Uses Three.js WebGL canvas, RDK X5 camera angles, and ML-predicted EMG.
 *
 * Role-aware: patients see the 3D canvas, doctors see task management.
 */

var app = getApp();
var dataManager = require('../../utils/dataManager');
var rfInference = require('../../utils/rfInference');
var threeAdapter = require('./threeAdapter');
var armModelBuilder = require('./armModel');
var TouchOrbitControls = require('./orbitControls');
var muscleMat = require('./muscleMaterials');

var scorer = require('../../utils/scorer');
var THREE;

Page({
  data: {
    // Role
    isDoctor: false,
    isPatient: false,

    // 3D model data (patient only)
    rdkStatus: '未连接',
    rdkAngle: 180,
    rightUpperAngle: 90,
    rightValid: false,
    predictedBiceps: 0,
    predictedTriceps: 0,
    predictedBrachioradialis: 0,
    selectedEmgValue: '0',
    showMuscleDetail: false,
    selectedMuscle: '',
    selectedMuscleName: '',
    selectedMuscleEn: '',
    activationPercent: 0,
    activationColor: '#2ecc71',
    activationLevel: '静息',
    modelLoading: false,
    modelReady: false,
    glLoadError: '',       // 3D 加载错误信息，空字符串表示正常
    // EMG input source
    emgSource: 'rf',            // 'tcn' | 'rf' | 'measured'
    emgSourceLabel: 'RF',
    showSourceMenu: false,
    bpm: '--',
    emg: 0,

    // Doctor: task management
    doctorTab: 'twin',        // 'twin' | 'tasks'
    doctorTasks: [],
    taskLoading: false,

    // Training mode (when active task is set)
    hasActiveTask: false,
    trainingActive: false,
    twinReps: 0,
    twinInZone: false,
    twinTask: null,
    twinGuidance: '',
    twinGuidanceType: 'hint'
  },

  // ============ Lifecycle ============

  onLoad: function () {
    this._three = {
      renderer: null, scene: null, camera: null, armModel: null, controls: null,
      animId: 0, canvasWidth: 0, canvasHeight: 0, lastTime: 0
    };
    this._unsub = null;
    this._lastTap = { x: 0, y: 0, time: 0 };
    this._lastInference = 0;
    this._demoInterval = null;
    this._twinReps = 0;
    this._twinPrevInZone = false;
    this._twinLeaveZone = false;

    var role = app.globalData.role;
    this.setData({
      isDoctor: role === 'doctor',
      isPatient: role === 'patient'
    });
  },

  onReady: function () {
    this._safeInitWebGL();
  },

  onShow: function () {
    var self = this;
    var role = app.globalData.role;
    this.setData({
      isDoctor: role === 'doctor',
      isPatient: role === 'patient'
    });

    // WebGL 和传感器订阅：医生和患者通用
    if (!this._three.renderer && !this.data.glLoadError) {
      this._safeInitWebGL();
    }

    this._unsub = dataManager.subscribe(function (data) {
      var b = data.predictedBiceps != null ? data.predictedBiceps : (data.tcnBiceps || data.emg || 0);
      var t = data.predictedTriceps != null ? data.predictedTriceps : (data.tcnTriceps || (data.emg || 0) * 0.3);
      var br = data.predictedBrachioradialis != null ? data.predictedBrachioradialis : (data.tcnBrachioradialis || b * 0.7);
      self.setData({
        rdkStatus: data.rdkStatus,
        rdkAngle: data.rdkAngle,
        rightUpperAngle: data.rightUpperAngle || data.leftUpperAngle || 90,
        rightValid: data.rightValid || data.leftValid,
        predictedBiceps: Number(b).toFixed(1),
        predictedTriceps: Number(t).toFixed(1),
        predictedBrachioradialis: Number(br).toFixed(1),
        bpm: data.bpm || '--',
        emg: data.emg || 0
      });
      if (self.data.trainingActive) {
        self._detectTwinRep(data.rdkAngle);
        self._updateGuidance(data.rdkAngle);
      }
    });

    if (this._three.renderer && !this._three.animId) {
      this.startAnimationLoop();
    }
    if (!rfInference.isLoaded() && !this.data.modelLoading) {
      this.ensureModelLoaded();
    }

    if (role === 'patient') {
      this._checkActiveTask();
    } else if (role === 'doctor') {
      this.fetchDoctorTasks();
    }
  },

  onHide: function () {
    if (this._unsub) { this._unsub(); this._unsub = null; }
    this.stopAnimationLoop();
  },

  onUnload: function () {
    this.stopAnimationLoop();
    if (this._unsub) { this._unsub(); this._unsub = null; }
    var state = this._three;
    if (state.renderer) {
      state.renderer.dispose();
      if (state.renderer.forceContextLoss) state.renderer.forceContextLoss();
      state.renderer = null;
    }
    if (state.scene) { disposeScene(state.scene); state.scene = null; }
  },

  // ============ WebGL Initialization ============

  /** 安全初始化 WebGL：捕获 npm 未构建等错误，显示友好提示 */
  _safeInitWebGL: function () {
    var self = this;
    try {
      this.initWebGL();
    } catch (e) {
      console.error('[DigitalTwin] WebGL init failed:', e.message);
      var msg = e.message || '';
      if (msg.indexOf('threejs-miniprogram') !== -1 || msg.indexOf('is not defined') !== -1) {
        self.setData({ glLoadError: 'threejs-miniprogram 未构建\n请在开发者工具中点击\n「工具 → 构建 npm」' });
      } else if (msg.indexOf('pako') !== -1) {
        self.setData({ glLoadError: 'pako 未构建\n请在开发者工具中点击\n「工具 → 构建 npm」' });
      } else {
        self.setData({ glLoadError: '3D 引擎启动失败\n' + msg.substring(0, 60) });
      }
    }
  },

  initWebGL: function () {
    var self = this;
    var query = wx.createSelectorQuery();
    query.select('#glCanvas')
      .fields({ node: true, size: true })
      .exec(function (res) {
        if (!res || !res[0] || !res[0].node) {
          console.error('[DigitalTwin] Canvas node not found');
          return;
        }
        var canvas = res[0].node;
        var width = res[0].width;
        var height = res[0].height;
        self._three.canvasWidth = width;
        self._three.canvasHeight = height;
        var renderer = threeAdapter.createWebGLRenderer(canvas, width, height);
        THREE = threeAdapter.THREE;
        self._three.lastTime = Date.now();
        self._three.renderer = renderer;
        var scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a30);
        scene.fog = new THREE.Fog(0x1a1a30, 12, 45);
        self._three.scene = scene;
        self.addLighting(scene);
        self._loadArmGLB(function (armModel) {
          if (!armModel || !armModel.rootGroup) {
            console.error('[DigitalTwin] Arm model unavailable');
            self.setData({ glLoadError: '3D 模型未能加载\n请确认 arm_model_clean.glb.gz\n已上传至云存储' });
            return;
          }
          scene.add(armModel.rootGroup);
          armModel.rootGroup.updateMatrixWorld(true);
          var bCenter = new THREE.Vector3();
          var box = new THREE.Box3();
          armModel.rootGroup.traverse(function (obj) {
            if (obj.isMesh && obj.geometry) {
              obj.geometry.computeBoundingBox();
              if (obj.geometry.boundingBox) {
                var gb = obj.geometry.boundingBox.clone();
                gb.applyMatrix4(obj.matrixWorld);
                box.union(gb);
              }
            }
          });
          var modelSize = new THREE.Vector3();
          box.getSize(modelSize);
          box.getCenter(bCenter);
          // 居中模型（不做 Blender Z-up 转换，arm_model_clean 已是 Y-up）
          armModel.rootGroup.position.set(-bCenter.x, -bCenter.y, -bCenter.z);
          armModel.rootGroup.updateMatrixWorld(true);
          armModel.rootGroup.traverse(function (obj) {
            if (obj.isSkinnedMesh && obj.skeleton) {
              obj.bind(obj.skeleton, obj.matrixWorld);
            }
          });
          self._three.armModel = armModel;
          var orbitTarget = new THREE.Vector3(0, 0, 0);
          self._three.orbitTarget = orbitTarget;
          var maxDim = Math.max(modelSize.x, modelSize.y, modelSize.z, 1);
          var camDist = maxDim * 1.5;
          self._three.cameraDist = camDist;
          if (!Number.isFinite(camDist) || camDist <= 0) camDist = 20;
          var camera = new THREE.PerspectiveCamera(45, width / Math.max(height, 1), 0.1, Math.max(camDist * 5, 50));
          camera.position.set(camDist * 0.3, camDist * 0.35, camDist * 0.7);
          camera.lookAt(orbitTarget);
          self._three.camera = camera;
          var controls = new TouchOrbitControls(camera, orbitTarget);
          controls.minDistance = maxDim * 0.4;
          controls.maxDistance = maxDim * 4.0;
          self._three.controls = controls;
          self.startAnimationLoop();
        });
      });
  },

  _loadArmGLB: function (callback) {
    var self = this;
    var fs = wx.getFileSystemManager();
    var cachePath = wx.env.USER_DATA_PATH + '/arm_model_clean.glb';
    try {
      var cached = fs.readFileSync(cachePath);
      console.log('[DigitalTwin] Arm GLB cache hit: ' + cached.byteLength + ' bytes');
      var buffer = self._maybeDecompress(cached);
      armModelBuilder.loadArmModel(buffer).then(function (model) {
        callback(model);
      }).catch(function (err) {
        console.error('[DigitalTwin] Arm GLB parse error (cached):', err.message);
        try { fs.unlinkSync(cachePath); } catch (_) {}
        self._downloadArmGLB(fs, cachePath, callback);
      });
    } catch (e) {
      console.log('[DigitalTwin] Arm GLB cache miss:', e.message);
      self._downloadArmGLB(fs, cachePath, callback);
    }
  },

  _downloadArmGLB: function (fs, cachePath, callback) {
    var self = this;
    var fileID = 'cloud://cloud1-d7g8tpqsscb7f752a.636c-cloud1-d7g8tpqsscb7f752a-1435955022/arm_model_clean.glb.gz';

    console.log('[DigitalTwin] Getting temp URL for arm GLB...');
    wx.cloud.getTempFileURL({
      fileList: [fileID],
      success: function (urlRes) {
        var url = urlRes.fileList && urlRes.fileList[0] && urlRes.fileList[0].tempFileURL;
        if (!url) {
          // Try direct hosting URL
          self._downloadViaWx('https://636c-cloud1-d7g8tpqsscb7f752a-1435955022.tcb.qcloud.la/arm_model_clean.glb.gz', fs, cachePath, callback);
          return;
        }
        // Use wx.downloadFile — most reliable cross-platform
        console.log('[DigitalTwin] Downloading via wx.downloadFile...');
        wx.downloadFile({
          url: url,
          success: function (res) {
            if (res.statusCode !== 200) { callback(null); return; }
            try {
              var compressed = fs.readFileSync(res.tempFilePath);
              console.log('[DigitalTwin] Arm GLB downloaded: ' + compressed.byteLength + ' bytes');
              try { fs.writeFileSync(cachePath, compressed); } catch (e) {}
              var buffer = self._maybeDecompress(compressed);
              armModelBuilder.loadArmModel(buffer).then(function (model) { callback(model); })
                .catch(function (err) { console.error('[DigitalTwin] Parse error:', err.message); callback(null); });
            } catch (err) { console.error('[DigitalTwin] Read error:', err.message); callback(null); }
          },
          fail: function (err) {
            console.error('[DigitalTwin] wx.downloadFile failed:', JSON.stringify(err));
            self._downloadViaWx('https://636c-cloud1-d7g8tpqsscb7f752a-1435955022.tcb.qcloud.la/arm_model_clean.glb.gz', fs, cachePath, callback);
          }
        });
      },
      fail: function (err) {
        console.error('[DigitalTwin] getTempFileURL failed:', JSON.stringify(err));
        self._downloadViaWx('https://636c-cloud1-d7g8tpqsscb7f752a-1435955022.tcb.qcloud.la/arm_model_clean.glb.gz', fs, cachePath, callback);
      }
    });
  },

  /** Download via wx.downloadFile with direct URL */
  _downloadViaWx: function (url, fs, cachePath, callback) {
    var self = this;
    console.log('[DigitalTwin] Trying direct URL via wx.downloadFile...');
    wx.downloadFile({
      url: url,
      success: function (res) {
        if (res.statusCode !== 200) { callback(null); return; }
        try {
          var compressed = fs.readFileSync(res.tempFilePath);
          try { fs.writeFileSync(cachePath, compressed); } catch (e) {}
          var buffer = self._maybeDecompress(compressed);
          armModelBuilder.loadArmModel(buffer).then(function (model) { callback(model); })
            .catch(function () { callback(null); });
        } catch (err) { callback(null); }
      },
      fail: function () { callback(null); }
    });
  },

  _maybeDecompress: function (buffer) {
    var arr = new Uint8Array(buffer);
    if (arr[0] === 0x1f && arr[1] === 0x8b) {
      try {
        var pako = require('pako');
        var result = pako.ungzip(arr);
        return result.buffer;
      } catch (e) {
        console.error('[DigitalTwin] GLB decompress failed:', e.message);
        throw new Error('3D模型解压失败，请清除缓存后重试');
      }
    }
    return buffer;
  },

  addLighting: function (scene) {
    scene.add(new THREE.AmbientLight(0xffeedd, 0.5));
    var keyLight = new THREE.DirectionalLight(0xfff5ee, 1.0);
    keyLight.position.set(8, 6, 8);
    scene.add(keyLight);
    var rimLight = new THREE.DirectionalLight(0xaaccff, 0.35);
    rimLight.position.set(-3, 2, -5);
    scene.add(rimLight);
    var fillLight = new THREE.DirectionalLight(0xddddff, 0.3);
    fillLight.position.set(-5, 2, 4);
    scene.add(fillLight);
    scene.add(new THREE.HemisphereLight(0xffeedd, 0x332222, 0.25));
  },

  // ============ Animation Loop ============

  startAnimationLoop: function () {
    if (this._three.animId) return;
    var self = this;
    var state = this._three;
    var errCount = 0;
    function render() {
      if (!state.renderer || !state.scene || !state.camera) return;
      try {
        var now = Date.now();
        var dt = Math.min((now - state.lastTime) / 1000, 0.1);
        state.lastTime = now;
        if (state.controls) state.controls.update();
        self.updateArmPose(dt);
        self.updateInference();
        state.renderer.render(state.scene, state.camera);
        errCount = 0; // 连续成功则重置
      } catch (e) {
        errCount++;
        console.error('[DigitalTwin] Render loop error (' + errCount + '):', e.message);
        // 连续 10 帧失败则停止，避免死循环；偶发错误则继续
        if (errCount >= 10) {
          console.error('[DigitalTwin] Render loop stopped after 10 consecutive errors');
          state.animId = 0;
          return;
        }
      }
      // 始终调度下一帧，确保单次错误不会杀死循环
      if (typeof wx !== 'undefined' && wx.requestAnimationFrame) {
        state.animId = wx.requestAnimationFrame(render);
      } else if (typeof requestAnimationFrame !== 'undefined') {
        state.animId = requestAnimationFrame(render);
      } else {
        state.animId = setTimeout(render, 16);
      }
    }
    render();
  },

  stopAnimationLoop: function () {
    var state = this._three;
    if (state.animId) {
      // 使用与本项目 startAnimationLoop 一致的取消方式
      if (typeof wx !== 'undefined' && wx.cancelAnimationFrame) {
        wx.cancelAnimationFrame(state.animId);
      } else if (typeof cancelAnimationFrame !== 'undefined') {
        cancelAnimationFrame(state.animId);
      } else {
        clearTimeout(state.animId);
      }
      state.animId = 0;
    }
  },

  // ============ Arm Pose Update ============

  updateArmPose: function (dt) {
    var armModel = this._three.armModel;
    if (!armModel) return;
    var sensorData = dataManager.getSensorData();
    var angle = sensorData.rdkAngle || 180;
    var elbowRad = ((180 - angle) / 180) * Math.PI;
    if (armModel.hasSkinning && armModel.forearmBone) {
      armModel.forearmBone.rotation.x = -elbowRad;
    } else if (armModel.elbowPivot) {
      armModel.elbowPivot.rotation.x = -elbowRad;
    }
    if (sensorData.rightUpperAngle || sensorData.leftUpperAngle) {
      var shoulderAngle = sensorData.rightUpperAngle || sensorData.leftUpperAngle;
      var shoulderOffset = ((shoulderAngle - 90) / 90) * 0.3;
      armModel.shoulderPivot.rotation.x = shoulderOffset;
    }
  },

  // ============ ML Inference ============

  updateInference: function () {
    try { this._updateInferenceImpl(); }
    catch (e) { console.error('[DigitalTwin] updateInference error:', e.message); }
  },

  _updateInferenceImpl: function () {
    var now = Date.now();
    if (now - this._lastInference < 50) return;
    this._lastInference = now;

    var armModel = this._three.armModel;
    if (!armModel || !armModel.materials || !armModel.materials.biceps || !armModel.materials.triceps) return;

    var sensorData = dataManager.getSensorData();

    // 优先使用 BLE 实测数据（如果有），否则回退到当前输入源
    var useBleData = (sensorData.bicepsBLE > 420) || (sensorData.tricepsBLE > 420);

    // 根据输入源获取 EMG 值
    var bEmg, tEmg, brachEmg;
    var source = this.data.emgSource;

    if (useBleData) {
      // BLE 实测: 将 ADC 转换为 μV
      bEmg = this._bleToMicrovolt(sensorData.bicepsBLE || 0);
      tEmg = this._bleToMicrovolt(sensorData.tricepsBLE || 0);
      brachEmg = bEmg * 0.7;
    } else if (source === 'tcn') {
      // TCN: RDK WebSocket 推送的预测值
      bEmg = sensorData.tcnBiceps || 0;
      tEmg = sensorData.tcnTriceps || 0;
      brachEmg = sensorData.tcnBrachioradialis || bEmg * 0.7;
    } else if (source === 'measured') {
      // 实测: BLE 采集的原始 EMG
      bEmg = sensorData.emg || 0;
      tEmg = sensorData.emg ? sensorData.emg * 0.3 : 0;  // 三头肌约为二头肌的30%
      brachEmg = bEmg * 0.7;
    } else {
      // RF: 小程序本地 RandomForest
      var result = dataManager.runInference(rfInference);
      if (!result) return;
      bEmg = result.biceps;
      tEmg = result.triceps;
      brachEmg = result.brachioradialis;
    }

    var bicepsInfo = muscleMat.updateMuscleMaterial(armModel.materials.biceps, bEmg, 'biceps');
    var tricepsInfo = muscleMat.updateMuscleMaterial(armModel.materials.triceps, tEmg, 'triceps');
    var brachInfo = null;
    if (armModel.materials.brachioradialis) {
      brachInfo = muscleMat.updateMuscleMaterial(armModel.materials.brachioradialis, brachEmg, 'brachioradialis');
    }
    var shoulderEmg = (sensorData && (sensorData.rightUpperAngle || sensorData.leftUpperAngle))
      ? Math.abs((sensorData.rightUpperAngle || sensorData.leftUpperAngle) - 90) / 90 * 800 : 0;
    if (armModel.materials.deltoid) {
      muscleMat.updateMuscleMaterial(armModel.materials.deltoid, shoulderEmg, 'deltoid');
    }
    if (this.data.showMuscleDetail) {
      var sel = this.data.selectedMuscle;
      var info, col, emgVal;
      if (sel === 'biceps') { info = bicepsInfo; col = info.color; emgVal = bEmg; }
      else if (sel === 'triceps') { info = tricepsInfo; col = info.color; emgVal = tEmg; }
      else if (sel === 'deltoid') { info = muscleMat.emgToActivationLevel(shoulderEmg, 'deltoid'); col = muscleMat.emgToColor(shoulderEmg, 'deltoid'); emgVal = shoulderEmg; }
      else if (sel === 'brachioradialis') { info = brachInfo || muscleMat.emgToActivationLevel(0, 'brachioradialis'); col = brachInfo ? brachInfo.color : muscleMat.emgToColor(0, 'brachioradialis'); emgVal = brachEmg; }
      else { info = bicepsInfo; col = info.color; emgVal = bEmg; }
      this.setData({
        predictedBiceps: bEmg.toFixed(1),
        predictedTriceps: tEmg.toFixed(1),
        predictedBrachioradialis: brachEmg.toFixed(1),
        selectedEmgValue: emgVal.toFixed(1),
        activationPercent: info.percent,
        activationColor: '#' + col.getHexString(),
        activationLevel: muscleMat.getLevelDisplayName(info.level)
      });
    }
  },

  /** Convert BLE ADC raw value (0-1023) to microvolts (~0-800 μV) */
  _bleToMicrovolt: function (adc) {
    if (!adc || adc <= 420) return 0;
    return Math.round((adc - 420) / 600 * 800);
  },

  // ============ Model Loading ============

  ensureModelLoaded: function () {
    if (rfInference.isLoaded()) { this.setData({ modelReady: true }); return; }
    var self = this;
    this.setData({ modelLoading: true });
    wx.showLoading({ title: '加载AI模型...', mask: true });
    var fs = wx.getFileSystemManager();
    var cachePath = wx.env.USER_DATA_PATH + '/model.bin';
    try {
      var cached = fs.readFileSync(cachePath);
      var arr = new Uint8Array(cached);
      if (arr[0] === 0x1f && arr[1] === 0x8b) cached = self._decompressGzip(cached);
      rfInference.loadModel(cached);
      wx.hideLoading();
      self.setData({ modelLoading: false, modelReady: true });
      wx.showToast({ title: 'AI模型就绪(缓存)', icon: 'success', duration: 1500 });
    } catch (e) {
      try { fs.unlinkSync(cachePath); } catch (_) {}
      self._downloadModel(fs, cachePath);
    }
  },

  _downloadModel: function (fs, cachePath) {
    var self = this;
    var fileID = 'cloud://cloud1-d7g8tpqsscb7f752a.636c-cloud1-d7g8tpqsscb7f752a-1435955022/model.bin.gz';
    var hostingUrl = 'https://636c-cloud1-d7g8tpqsscb7f752a-1435955022.tcb.qcloud.la/model.bin.gz';

    console.log('[DigitalTwin] Getting temp URL for AI model...');
    wx.cloud.getTempFileURL({
      fileList: [fileID],
      success: function (urlRes) {
        var url = urlRes.fileList && urlRes.fileList[0] && urlRes.fileList[0].tempFileURL;
        var downloadUrl = url || hostingUrl;
        wx.request({
          url: downloadUrl,
          responseType: 'arraybuffer',
          success: function (res) {
            if (res.statusCode !== 200) { self._downloadTryHosting(fs, cachePath); return; }
            try {
              var compressed = res.data;
              try { fs.writeFileSync(cachePath, compressed, 'binary'); } catch (e) {}
              var buffer = self._decompressGzip(compressed);
              rfInference.loadModel(buffer);
              wx.hideLoading();
              self.setData({ modelLoading: false, modelReady: true });
              wx.showToast({ title: 'AI模型就绪', icon: 'success', duration: 1500 });
            } catch (err) { self._downloadTryHosting(fs, cachePath); }
          },
          fail: function () { self._downloadTryHosting(fs, cachePath); }
        });
      },
      fail: function () {
        wx.request({
          url: hostingUrl,
          responseType: 'arraybuffer',
          success: function (res) {
            if (res.statusCode !== 200) { self._downloadTryHosting(fs, cachePath); return; }
            try {
              var compressed = res.data;
              try { fs.writeFileSync(cachePath, compressed, 'binary'); } catch (e) {}
              var buffer = self._decompressGzip(compressed);
              rfInference.loadModel(buffer);
              wx.hideLoading();
              self.setData({ modelLoading: false, modelReady: true });
              wx.showToast({ title: 'AI模型就绪', icon: 'success', duration: 1500 });
            } catch (err) { self._downloadTryHosting(fs, cachePath); }
          },
          fail: function () { self._downloadTryHosting(fs, cachePath); }
        });
      }
    });
  },

  _downloadTryHosting: function (fs, cachePath) {
    var self = this;
    var hostingUrl = 'https://636c-cloud1-d7g8tpqsscb7f752a-1435955022.tcb.qcloud.la/model.bin.gz';
    wx.request({
      url: hostingUrl,
      responseType: 'arraybuffer',
      success: function (res) {
        if (res.statusCode !== 200) { self._onDownloadFail('HTTP ' + res.statusCode); return; }
        try {
          var compressed = res.data;
          try { fs.writeFileSync(cachePath, compressed, 'binary'); } catch (e) {}
          var buffer = self._decompressGzip(compressed);
          rfInference.loadModel(buffer);
          wx.hideLoading();
          self.setData({ modelLoading: false, modelReady: true });
          wx.showToast({ title: 'AI模型就绪', icon: 'success', duration: 1500 });
        } catch (err) { self._onDownloadFail('解析失败: ' + err.message); }
      },
      fail: function (err) { self._onDownloadFail('下载失败: ' + (err.errMsg || '')); }
    });
  },

  _onDownloadFail: function (msg) {
    wx.hideLoading();
    this.setData({ modelLoading: false });
    wx.showModal({ title: '模型加载失败', content: msg + '\n\n将使用演示模式', showCancel: false });
  },

  _decompressGzip: function (compressedBuffer) {
    try {
      var pako = require('pako');
      var result = pako.ungzip(new Uint8Array(compressedBuffer));
      return result.buffer;
    } catch (e) {
      console.error('[DigitalTwin] Gzip decompress failed:', e.message);
      throw new Error('模型数据解压失败，请清除缓存后重试');
    }
  },

  // ============ Touch Events ============

  onTouchStart: function (e) {
    var touches = (e.touches || []).map(function (t) { return { x: t.x, y: t.y }; });
    if (this._three.controls) this._three.controls.handleTouchStart(touches);
    if (touches.length === 1) this._lastTap = { x: touches[0].x, y: touches[0].y, time: Date.now() };
    else this._lastTap = null;
  },

  onTouchMove: function (e) {
    var touches = (e.touches || []).map(function (t) { return { x: t.x, y: t.y }; });
    if (this._three.controls) this._three.controls.handleTouchMove(touches);
  },

  onTouchEnd: function (e) {
    var touches = (e.touches || []).map(function (t) { return { x: t.x, y: t.y }; });
    if (this._three.controls) this._three.controls.handleTouchEnd(touches);
    if (touches.length === 0 && this._lastTap && this._lastTap.time) {
      var elapsed = Date.now() - this._lastTap.time;
      var changed = e.changedTouches[0];
      var moved = changed ? Math.hypot(changed.x - this._lastTap.x, changed.y - this._lastTap.y) : 999;
      if (elapsed < 300 && moved < 15) this.handleTap(changed || { x: this._lastTap.x, y: this._lastTap.y });
      this._lastTap = null;
    }
  },

  handleTap: function (touch) {
    var state = this._three;
    if (!state.camera || !state.armModel) return;
    var allMuscleMeshes = [];
    var muscles = state.armModel.muscles;
    for (var key in muscles) {
      if (muscles.hasOwnProperty(key)) {
        var val = muscles[key];
        if (Array.isArray(val)) {
          allMuscleMeshes = allMuscleMeshes.concat(val);
        } else if (val) {
          allMuscleMeshes.push(val);
        }
      }
    }
    var hit = TouchOrbitControls.raycastMuscles(touch.x, touch.y,
      state.canvasWidth || 375, state.canvasHeight || 600, state.camera, allMuscleMeshes);
    if (hit) {
      var sensorData = dataManager.getSensorData();
      var source = this.data.emgSource;
      var emgVal;
      if (source === 'tcn') {
        if (hit === 'biceps') emgVal = sensorData.tcnBiceps || 0;
        else if (hit === 'triceps') emgVal = sensorData.tcnTriceps || 0;
        else if (hit === 'brachioradialis') emgVal = sensorData.tcnBrachioradialis || (sensorData.tcnBiceps || 0) * 0.7;
        else emgVal = 0;
      } else if (source === 'measured') {
        if (hit === 'biceps') emgVal = sensorData.emg || 0;
        else if (hit === 'triceps') emgVal = (sensorData.emg || 0) * 0.3;
        else if (hit === 'brachioradialis') emgVal = (sensorData.emg || 0) * 0.7;
        else emgVal = 0;
      } else {
        if (hit === 'biceps') emgVal = Number(sensorData.predictedBiceps) || 0;
        else if (hit === 'triceps') emgVal = Number(sensorData.predictedTriceps) || 0;
        else if (hit === 'brachioradialis') emgVal = Number(sensorData.predictedBrachioradialis) || (Number(sensorData.predictedBiceps) || 0) * 0.7;
        else emgVal = 0;
      }
      if (hit === 'deltoid') emgVal = Math.abs((sensorData.rightUpperAngle || sensorData.leftUpperAngle)
        ? ((sensorData.rightUpperAngle || sensorData.leftUpperAngle) - 90) / 90 * 800 : 0);
      var info = muscleMat.emgToActivationLevel(emgVal, hit);
      var color = muscleMat.emgToColor(emgVal, hit);
      var muscleNode = muscles[hit];
      // 兼容 Group 节点和数组两种存储格式
      var name = hit;
      var enName = '';
      if (muscleNode) {
        var ud = muscleNode.userData;
        if (ud && ud.displayName) { name = ud.displayName; enName = ud.enName || ''; }
        else if (Array.isArray(muscleNode) && muscleNode.length > 0) {
          var firstM = muscleNode[0];
          if (firstM && firstM.userData && firstM.userData.displayName) {
            name = firstM.userData.displayName;
            enName = firstM.userData.enName || '';
          }
        }
      }
      this.setData({
        showMuscleDetail: true, selectedMuscle: hit, selectedMuscleName: name, selectedMuscleEn: enName,
        predictedBiceps: sensorData.predictedBiceps.toFixed(1),
        predictedTriceps: sensorData.predictedTriceps.toFixed(1),
        predictedBrachioradialis: sensorData.predictedBrachioradialis ? sensorData.predictedBrachioradialis.toFixed(1) : '0',
        selectedEmgValue: emgVal.toFixed(1), activationPercent: info.percent,
        activationColor: '#' + color.getHexString(), activationLevel: muscleMat.getLevelDisplayName(info.level)
      });
    } else {
      if (this.data.showMuscleDetail) this.closeMuscleDetail();
    }
  },

  closeMuscleDetail: function () { this.setData({ showMuscleDetail: false }); },

  // ============ Input Source Switching ============

  toggleSourceMenu: function () {
    this.setData({ showSourceMenu: !this.data.showSourceMenu });
  },

  selectSource: function (e) {
    var source = e.currentTarget.dataset.source;
    var labels = { tcn: 'TCN', rf: 'RF', measured: '实测' };
    this.setData({
      emgSource: source,
      emgSourceLabel: labels[source] || source,
      showSourceMenu: false
    });
    wx.showToast({ title: '切换到 ' + labels[source], icon: 'none', duration: 1000 });
  },

  resetCamera: function () {
    var controls = this._three.controls;
    var state = this._three;
    if (!controls || !state.camera || !state.cameraDist) return;
    var target = this._three.orbitTarget || new THREE.Vector3(0, 0, 0);
    var d = state.cameraDist;
    controls.reset(new THREE.Vector3(d * 0.6, d * 0.15, d * 0.8), target);
  },

  // ============ Training Mode ============

  /** Check for active task from globalData and set up training context */
  _checkActiveTask: function () {
    var task = app.globalData.activeTask;
    var ts = app.globalData.trainingState;

    if (task && task._id) {
      // Sync rep count from shared state
      var savedReps = (ts && ts.taskId === task._id) ? (ts.currentRep || 0) : 0;
      var isActive = ts && ts.active && ts.taskId === task._id;
      this._twinReps = savedReps;
      this._twinPrevInZone = false;
      this._twinLeaveZone = false;

      // Load standard template if task has one
      if (task.templateId && !this._twinTemplate) {
        this._twinTemplate = null;
        this._loadTwinTemplate(task.templateId);
      }

      if (isActive) {
        // Training already running — resume with synced reps
        this._twinLeaveZone = true; // arm for next rep detection
        this.setData({
          hasActiveTask: true,
          twinTask: task,
          twinReps: savedReps,
          twinInZone: false,
          trainingActive: true
        });
      } else if (ts && ts.autoStart && ts.taskId === task._id) {
        // Coming from taskDetail "开始训练" — auto-start
        ts.autoStart = false;
        this.setData({
          hasActiveTask: true,
          twinTask: task,
          twinReps: savedReps,
          twinInZone: false,
          trainingActive: false
        });
        var self = this;
        setTimeout(function () {
          self.startTwinTraining();
        }, 800);
      } else {
        // Task exists but training not active yet
        this.setData({
          hasActiveTask: true,
          twinTask: task,
          twinReps: savedReps,
          twinInZone: false,
          trainingActive: false
        });
      }
    } else {
      this.setData({
        hasActiveTask: false,
        twinTask: null,
        trainingActive: false
      });
    }
  },

  /** Load standard action template for scoring */
  _loadTwinTemplate: function (templateId) {
    var self = this;
    wx.cloud.callFunction({
      name: 'actionTemplates',
      data: { action: 'get', templateId: templateId },
      success: function (res) {
        if (res.result && res.result.success) {
          self._twinTemplate = res.result.template;
        }
      },
      fail: function () {}
    });
  },

  /** Start training mode on digital twin — activate rep counting */
  startTwinTraining: function () {
    if (!this.data.hasActiveTask) return;
    var task = this.data.twinTask;
    // Init or update shared training state
    app.globalData.trainingState = {
      taskId: task._id,
      active: true,
      autoStart: false,
      currentRep: this._twinReps || 0,
      startedAt: Date.now()
    };
    this._twinPrevInZone = false;
    this._twinLeaveZone = true;
    this.setData({
      trainingActive: true,
      twinInZone: false
    });
    wx.showToast({ title: '训练开始！', icon: 'success', duration: 1200 });
  },

  /** Stop training mode */
  stopTwinTraining: function () {
    if (app.globalData.trainingState) {
      app.globalData.trainingState.active = false;
    }
    this.setData({ trainingActive: false });
    wx.showToast({ title: '训练已停止', icon: 'none' });
  },

  /** Detect a rep based on angle entering and leaving the target zone */
  _detectTwinRep: function (angle) {
    var task = this.data.twinTask;
    if (!task) return;
    var inZone = angle >= task.targetAngleMin && angle <= task.targetAngleMax;

    if (inZone && !this._twinPrevInZone && this._twinLeaveZone) {
      // Re-entered zone after leaving — count a rep
      this._twinLeaveZone = false;
      var newRep = this._twinReps + 1;
      this._twinReps = newRep;
      this.setData({ twinReps: newRep });

      // Sync to shared state
      if (app.globalData.trainingState) {
        app.globalData.trainingState.currentRep = newRep;
      }

      wx.vibrateShort({ type: 'light' });
      if (newRep >= task.repetitions) {
        this.setData({ trainingActive: false });
        wx.showModal({
          title: '🎉 训练完成',
          content: '全部 ' + task.repetitions + ' 次已完成！是否标记任务完成？',
          confirmText: '标记完成',
          cancelText: '稍后',
          success: function (modalRes) {
            if (modalRes.confirm) {
              // Navigate to taskDetail to mark complete
              wx.navigateTo({ url: '/pages/taskDetail/taskDetail?taskId=' + task._id });
            }
          }
        });
      }
    } else if (!inZone && this._twinPrevInZone) {
      // Left target zone — arm the next rep detection
      this._twinLeaveZone = true;
    }

    this._twinPrevInZone = inZone;
    this.setData({ twinInZone: inZone });
  },

  /** Update real-time guidance text */
  _updateGuidance: function (angle) {
    var task = this.data.twinTask;
    if (!task) return;
    // Build angle history
    if (!this._twinAngleHistory) this._twinAngleHistory = [];
    this._twinAngleHistory.push(angle);
    if (this._twinAngleHistory.length > 30) this._twinAngleHistory.shift();

    var result;
    if (this._twinTemplate) {
      result = scorer.getDetailedGuidance(angle, this._twinAngleHistory, this._twinTemplate, task.targetAngleMin, task.targetAngleMax);
    } else {
      result = { text: scorer.getGuidance(angle, task.targetAngleMin, task.targetAngleMax), type: angle >= task.targetAngleMin && angle <= task.targetAngleMax ? 'good' : 'hint' };
    }
    if (result.text !== this.data.twinGuidance) {
      this.setData({ twinGuidance: result.text, twinGuidanceType: result.type });
    }
  },

  // ============ Doctor: Task Management ============

  /** 医生端切换子 Tab */
  switchDoctorTab: function (e) {
    var tab = e.currentTarget.dataset.tab;
    this.setData({ doctorTab: tab });
  },

  fetchDoctorTasks: function () {
    var self = this;
    this.setData({ taskLoading: true });
    var db = wx.cloud.database();
    db.collection('tasks')
      .where({ doctorId: app.globalData.userInfo ? app.globalData.userInfo._openid : '' })
      .orderBy('createdAt', 'desc')
      .limit(50).get()
      .then(function (res) { self.setData({ doctorTasks: res.data, taskLoading: false }); })
      .catch(function () { self.setData({ taskLoading: false }); });
  },

  openDoctorTask: function (e) {
    wx.navigateTo({ url: '/pages/taskDetail/taskDetail?taskId=' + e.currentTarget.dataset.taskid });
  },

  /** Navigate to standard action recording page */
  goRecordAction: function () {
    wx.navigateTo({ url: '/pages/actionRecord/actionRecord' });
  }
});

// ============ Helpers ============

function disposeScene(scene) {
  scene.traverse(function (obj) {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(function (m) { m.dispose(); });
      else obj.material.dispose();
    }
  });
}

