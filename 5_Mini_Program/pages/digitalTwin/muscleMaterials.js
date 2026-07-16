/**
 * Muscle color mapping based on EMG activation values.
 *
 * EMG value ranges (ML predicted values):
 *   Rest:   0-30 μV
 *   Light:  30-100 μV
 *   Medium: 100-200 μV
 *   High:   200+ μV
 *
 * Per-muscle sensitivity (scale applied to EMG before threshold comparison —
 * higher = shows color sooner):
 *   triceps: 2.0x (antagonist, naturally lower activation)
 *   default: 1.0x
 *
 * Color gradient:
 *   Green (#2ecc71) → Yellow (#f1c40f) → Orange (#e67e22) → Red (#e74c3c)
 */

var THREE;

function ensureThree() {
  if (!THREE) {
    var ta = require('./threeAdapter');
    THREE = ta.THREE;
    if (!THREE) {
      throw new Error('[muscleMaterials] THREE not initialized — call threeAdapter.initThree() first');
    }
  }
  return THREE;
}

var EMG_THRESHOLDS = {
  LOW: 30,
  MEDIUM: 100,
  HIGH: 200
};

var MUSCLE_SCALES = {
  biceps: 1.0,
  triceps: 2.0,
  deltoid: 1.0,
  brachioradialis: 1.0
};

var COLORS = {
  REST:   null,
  LOW:    null,
  MEDIUM: null,
  HIGH:   null,
  PEAK:   null
};

function initColors() {
  if (COLORS.REST) return;
  var T = ensureThree();
  COLORS.REST   = new T.Color(0x2ecc71);
  COLORS.LOW    = new T.Color(0x27ae60);
  COLORS.MEDIUM = new T.Color(0xf1c40f);
  COLORS.HIGH   = new T.Color(0xe67e22);
  COLORS.PEAK   = new T.Color(0xe74c3c);
}

function getScale(muscle) {
  return (muscle && MUSCLE_SCALES[muscle]) ? MUSCLE_SCALES[muscle] : 1.0;
}

/**
 * @param {number} emgValue
 * @param {string=} muscle - muscle key for per-muscle sensitivity
 * @returns {{level: string, percent: number}}
 */
function emgToActivationLevel(emgValue, muscle) {
  var scale = getScale(muscle);
  var val = emgValue * scale;

  var level, percent;
  if (val < EMG_THRESHOLDS.LOW) {
    level = 'rest';
    percent = (val / EMG_THRESHOLDS.LOW) * 25;
  } else if (val < EMG_THRESHOLDS.MEDIUM) {
    level = 'low';
    percent = 25 + (val - EMG_THRESHOLDS.LOW) /
      (EMG_THRESHOLDS.MEDIUM - EMG_THRESHOLDS.LOW) * 25;
  } else if (val < EMG_THRESHOLDS.HIGH) {
    level = 'medium';
    percent = 50 + (val - EMG_THRESHOLDS.MEDIUM) /
      (EMG_THRESHOLDS.HIGH - EMG_THRESHOLDS.MEDIUM) * 25;
  } else {
    level = 'high';
    percent = 75 + Math.min(
      (val - EMG_THRESHOLDS.HIGH) / 200 * 25, 25);
  }
  return { level: level, percent: Math.min(100, Math.max(0, percent)) };
}

/**
 * @param {number} emgValue
 * @param {string=} muscle
 * @returns {THREE.Color}
 */
function emgToColor(emgValue, muscle) {
  initColors();
  var val = emgValue * getScale(muscle);

  if (val < EMG_THRESHOLDS.LOW) {
    var t = val / EMG_THRESHOLDS.LOW;
    return COLORS.REST.clone().lerp(COLORS.LOW, t);
  }
  if (val < EMG_THRESHOLDS.MEDIUM) {
    var t2 = (val - EMG_THRESHOLDS.LOW) /
      (EMG_THRESHOLDS.MEDIUM - EMG_THRESHOLDS.LOW);
    return COLORS.LOW.clone().lerp(COLORS.MEDIUM, t2);
  }
  if (val < EMG_THRESHOLDS.HIGH) {
    var t3 = (val - EMG_THRESHOLDS.MEDIUM) /
      (EMG_THRESHOLDS.HIGH - EMG_THRESHOLDS.MEDIUM);
    return COLORS.MEDIUM.clone().lerp(COLORS.HIGH, t3);
  }
  return COLORS.PEAK.clone();
}

/**
 * @param {THREE.MeshStandardMaterial} material
 * @param {number} emgValue
 * @param {string=} muscle
 * @returns {{color: THREE.Color, level: string, percent: number}}
 */
function updateMuscleMaterial(material, emgValue, muscle) {
  var color = emgToColor(emgValue, muscle);
  var info = emgToActivationLevel(emgValue, muscle);

  material.color.copy(color);
  material.emissive.copy(color);
  material.emissiveIntensity = Math.min(info.percent / 100 * 0.7, 0.7);
  material.opacity = 0.6 + (info.percent / 100) * 0.35;

  return { color: color, level: info.level, percent: info.percent };
}

function getLevelDisplayName(level) {
  var map = {
    'rest': '静息',
    'low': '轻度激活',
    'medium': '中度激活',
    'high': '高度激活'
  };
  return map[level] || level;
}

module.exports = {
  emgToColor: emgToColor,
  emgToActivationLevel: emgToActivationLevel,
  updateMuscleMaterial: updateMuscleMaterial,
  getLevelDisplayName: getLevelDisplayName,
  EMG_THRESHOLDS: EMG_THRESHOLDS
};
