/**
 * Touch-based orbit controls for WeChat Mini Program WebGL canvas.
 *
 * Features:
 *   - Single finger drag → rotate around target (with inertia damping)
 *   - Two finger pinch → zoom in/out
 *   - Tap detection (short touch, minimal movement)
 *   - Raycasting for muscle selection
 */

var THREE;

function ensureThree() {
  if (!THREE) {
    try {
      THREE = require('./threeAdapter').THREE || require('three');
    } catch (e) {
      THREE = require('three');
    }
  }
  return THREE;
}

var TAP_MAX_DURATION = 300;   // ms
var TAP_MAX_MOVEMENT = 10;    // px
var DAMPING = 0.85;
var ROTATE_SPEED = 0.005;
var ZOOM_SPEED = 0.01;

function TouchOrbitControls(camera, target) {
  ensureThree();

  this.camera = camera;
  this.target = target ? target.clone() : new THREE.Vector3(0, -2, 0);
  this.enabled = true;

  // Spherical coordinates
  var offset = new THREE.Vector3().copy(camera.position).sub(this.target);
  this.spherical = new THREE.Spherical();
  this.spherical.setFromVector3(offset);
  this.sphericalDelta = new THREE.Spherical();

  // Config
  this.minDistance = 3;
  this.maxDistance = 15;
  this.minPolarAngle = 0.2;
  this.maxPolarAngle = Math.PI - 0.2;

  // Internal state
  this._state = { type: 'NONE' };
  this._touches = { one: null, two: null };
  this._prevDistance = 0;
}

/**
 * Reset camera to default view.
 * @param {THREE.Vector3} position
 * @param {THREE.Vector3} target
 */
TouchOrbitControls.prototype.reset = function (position, target) {
  this.target.copy(target);
  var offset = new THREE.Vector3().copy(position).sub(this.target);
  this.spherical.setFromVector3(offset);
  this.sphericalDelta.set(0, 0, 0);
  this._state.type = 'NONE';
  this.camera.position.copy(position);
  this.camera.lookAt(this.target);
};

TouchOrbitControls.prototype.handleTouchStart = function (touches) {
  if (touches.length === 1) {
    this._state.type = 'ROTATE';
    this._touches.one = { x: touches[0].x, y: touches[0].y };
    this.sphericalDelta.set(0, 0, 0);
  } else if (touches.length >= 2) {
    this._state.type = 'ZOOM';
    this._touches.two = [
      { x: touches[0].x, y: touches[0].y },
      { x: touches[1].x, y: touches[1].y }
    ];
    this._prevDistance = Math.hypot(
      touches[0].x - touches[1].x,
      touches[0].y - touches[1].y
    );
  }
};

TouchOrbitControls.prototype.handleTouchMove = function (touches) {
  if (!this.enabled) return;

  if (this._state.type === 'ROTATE' && touches.length === 1) {
    var dx = touches[0].x - this._touches.one.x;
    var dy = touches[0].y - this._touches.one.y;
    this.sphericalDelta.theta -= dx * ROTATE_SPEED;
    this.sphericalDelta.phi -= dy * ROTATE_SPEED;
    this._touches.one = { x: touches[0].x, y: touches[0].y };
  }

  if (this._state.type === 'ZOOM' && touches.length >= 2) {
    var dist = Math.hypot(
      touches[0].x - touches[1].x,
      touches[0].y - touches[1].y
    );
    var delta = this._prevDistance - dist;
    this.spherical.radius += delta * ZOOM_SPEED;
    this.spherical.radius = Math.max(this.minDistance,
      Math.min(this.maxDistance, this.spherical.radius));
    this._prevDistance = dist;
  }
};

TouchOrbitControls.prototype.handleTouchEnd = function (touches) {
  if (touches.length === 0) {
    this._state.type = 'NONE';
  } else if (touches.length === 1) {
    // Switch from ZOOM back to ROTATE
    this._state.type = 'ROTATE';
    this._touches.one = { x: touches[0].x, y: touches[0].y };
    this.sphericalDelta.set(0, 0, 0);
  }
};

/**
 * Call each frame before rendering. Applies rotation and damping.
 */
TouchOrbitControls.prototype.update = function () {
  if (this._state.type === 'ROTATE') {
    this.spherical.theta -= this.sphericalDelta.theta;
    this.spherical.phi -= this.sphericalDelta.phi;

    // Clamp polar angle
    this.spherical.phi = Math.max(this.minPolarAngle,
      Math.min(this.maxPolarAngle, this.spherical.phi));

    // Damping
    this.sphericalDelta.theta *= DAMPING;
    this.sphericalDelta.phi *= DAMPING;
  }

  var offset = new THREE.Vector3().setFromSpherical(this.spherical);
  this.camera.position.copy(this.target).add(offset);
  this.camera.lookAt(this.target);
};

/**
 * Static method: raycast against muscle meshes.
 * @param {number} touchX - canvas-relative x
 * @param {number} touchY - canvas-relative y
 * @param {number} w - canvas width
 * @param {number} h - canvas height
 * @param {THREE.Camera} camera
 * @param {THREE.Mesh[]} muscleMeshes
 * @returns {string|null} muscle name ('biceps'/'triceps') or null
 */
TouchOrbitControls.raycastMuscles = function (touchX, touchY, w, h, camera, muscleMeshes) {
  ensureThree();

  var mouse = new THREE.Vector2();
  mouse.x = (touchX / w) * 2 - 1;
  mouse.y = -(touchY / h) * 2 + 1;

  var raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);

  // Collect all actual Mesh children from muscle groups for raycasting
  var allTargets = [];
  var targets = Array.isArray(muscleMeshes) ? muscleMeshes : Object.values(muscleMeshes);
  targets.forEach(function (node) {
    if (node.isMesh) {
      allTargets.push(node);
    } else {
      node.traverse(function (child) {
        if (child.isMesh) allTargets.push(child);
      });
    }
  });

  var intersects = raycaster.intersectObjects(allTargets, false);

  if (intersects.length > 0) {
    var obj = intersects[0].object;
    // Walk up the parent chain to find the muscle userData
    var cursor = obj;
    while (cursor) {
      if (cursor.userData && cursor.userData.muscle) {
        return cursor.userData.muscle;
      }
      cursor = cursor.parent;
    }
  }
  return null;
};

module.exports = TouchOrbitControls;
