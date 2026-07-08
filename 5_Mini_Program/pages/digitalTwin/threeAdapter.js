/**
 * Three.js adapter for WeChat Mini Program WebGL canvas.
 *
 * Uses threejs-miniprogram (bundled Three.js r108) with the
 * createScopedThreejs(canvas) API — the only way to get a working
 * THREE namespace in the mini program environment.
 */

var scopedTHREE = null;

/**
 * Initialize the scoped Three.js instance bound to a canvas.
 * Must be called once, after the canvas node is available.
 * @param {Object} canvas - from wx.createSelectorQuery select '#glCanvas'
 * @returns {Object} THREE namespace
 */
function initThree(canvas) {
  if (scopedTHREE) return scopedTHREE;

  var createScopedThreejs = require('threejs-miniprogram').createScopedThreejs;
  scopedTHREE = createScopedThreejs(canvas);
  return scopedTHREE;
}

/**
 * Create a WebGLRenderer bound to the given canvas.
 * @param {Object} canvas - from wx.createSelectorQuery select '#glCanvas'
 * @param {number} width
 * @param {number} height
 * @returns {THREE.WebGLRenderer}
 */
function createWebGLRenderer(canvas, width, height) {
  var THREE = initThree(canvas);

  var pixelRatio = 1;
  try {
    var info = wx.getWindowInfo ? wx.getWindowInfo() : wx.getSystemInfoSync();
    pixelRatio = info.pixelRatio || 1;
  } catch (e) { /* ignore */ }

  var renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: false
  });
  renderer.setPixelRatio(Math.min(pixelRatio, 2));
  renderer.setSize(width, height);
  renderer.setClearColor(0x1a1a30, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap || THREE.PCFShadowMap || 1;

  // outputEncoding in r108, outputColorSpace in r152+
  if (renderer.outputColorSpace !== undefined) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  } else if (renderer.outputEncoding !== undefined) {
    renderer.outputEncoding = THREE.sRGBEncoding;
  }

  return renderer;
}

module.exports = {
  initThree: initThree,
  createWebGLRenderer: createWebGLRenderer,
  get THREE() { return scopedTHREE; }
};
