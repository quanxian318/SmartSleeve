/**
 * Training quality scoring engine.
 *
 * Compares patient training data against a standard template (from doctor)
 * and produces a 0-100 composite score with detailed breakdown.
 *
 * Usage:
 *   var scorer = require('../../utils/scorer');
 *   var result = scorer.score(patientData, standardTemplate);
 *   // { total: 85, emg: 35, angle: 28, smoothness: 15, completion: 7, breakdown: [...] }
 */

/** Main scoring function */
function score(patientData, standardTemplate) {
  if (!standardTemplate) {
    return fallbackScore(patientData);
  }

  var emgScore = scoreEmgSimilarity(patientData, standardTemplate);
  var angleScore = scoreAngleMatch(patientData, standardTemplate);
  var smoothScore = scoreSmoothness(patientData);
  var completionScore = scoreCompletion(patientData);

  return {
    total: emgScore + angleScore + smoothScore + completionScore,
    emg: emgScore,
    angle: angleScore,
    smoothness: smoothScore,
    completion: completionScore,
    breakdown: [
      { label: '肌电相似度', score: emgScore, max: 40 },
      { label: '角度匹配度', score: angleScore, max: 30 },
      { label: '动作平滑度', score: smoothScore, max: 20 },
      { label: '训练完成度', score: completionScore, max: 10 }
    ],
    level: getLevel(emgScore + angleScore + smoothScore + completionScore)
  };
}

/** Score EMG similarity (0-40 points) */
function scoreEmgSimilarity(patient, standard) {
  var stdBiceps = standard.standardEmgBiceps || 800;
  var stdTriceps = standard.standardEmgTriceps || 300;
  var patBiceps = patient.maxEmgBiceps || patient.maxEmg || 0;
  var patTriceps = patient.maxEmgTriceps || 0;

  var bicepsDev = Math.abs(patBiceps - stdBiceps) / Math.max(stdBiceps, 1);
  var tricepsDev = stdTriceps > 0 ? Math.abs(patTriceps - stdTriceps) / Math.max(stdTriceps, 1) : 0;
  var avgDev = (bicepsDev + tricepsDev) / 2;

  if (avgDev < 0.15) return 40;
  if (avgDev < 0.30) return 30;
  if (avgDev < 0.50) return 15;
  return 5;
}

/** Score angle range match (0-30 points) */
function scoreAngleMatch(patient, standard) {
  var stdMin = standard.targetAngleMin || 50;
  var stdMax = standard.targetAngleMax || 140;
  var patMin = patient.angleMin || patient.avgAngle || 180;
  var patMax = patient.angleMax || patient.avgAngle || 0;

  // How much of the patient's range overlaps with the standard range
  var overlapMin = Math.max(stdMin, patMin);
  var overlapMax = Math.min(stdMax, patMax);
  var overlap = Math.max(0, overlapMax - overlapMin);
  var stdRange = stdMax - stdMin;
  var coverage = stdRange > 0 ? overlap / stdRange : 0;

  if (coverage > 0.9) return 30;
  if (coverage > 0.7) return 22;
  if (coverage > 0.5) return 15;
  if (coverage > 0.3) return 8;
  return 3;
}

/** Score movement smoothness (0-20 points) */
function scoreSmoothness(patient) {
  var curve = patient.angleCurve;
  if (!curve || curve.length < 3) return 10; // not enough data

  var jerks = 0;
  for (var i = 2; i < curve.length; i++) {
    var d1 = curve[i - 1] - curve[i - 2];
    var d2 = curve[i] - curve[i - 1];
    if (Math.abs(d2 - d1) > 15) jerks++; // sudden change > 15 degrees
  }

  var jerkRatio = jerks / curve.length;
  if (jerkRatio < 0.05) return 20;
  if (jerkRatio < 0.1) return 15;
  if (jerkRatio < 0.2) return 10;
  return 5;
}

/** Score completion rate (0-10 points) */
function scoreCompletion(patient) {
  var target = patient.targetReps || 10;
  var actual = patient.completedReps || 0;
  var rate = target > 0 ? actual / target : 0;
  return Math.round(Math.min(rate, 1) * 10);
}

/** Fallback score when no standard template (use task target only) */
function fallbackScore(patient) {
  var angleScore = 20;
  var taskMin = patient.taskAngleMin || 0;
  var taskMax = patient.taskAngleMax || 180;
  var patMin = patient.angleMin || patient.avgAngle || 180;
  var patMax = patient.angleMax || patient.avgAngle || 0;
  var overlap = Math.max(0, Math.min(taskMax, patMax) - Math.max(taskMin, patMin));
  var coverage = (taskMax - taskMin) > 0 ? overlap / (taskMax - taskMin) : 0;
  if (coverage > 0.9) angleScore = 30;
  else if (coverage > 0.7) angleScore = 22;
  else if (coverage > 0.5) angleScore = 15;

  var smoothScore = scoreSmoothness(patient);
  var completionScore = scoreCompletion(patient);
  var total = angleScore + smoothScore + completionScore;

  return {
    total: total,
    emg: 0,
    angle: angleScore,
    smoothness: smoothScore,
    completion: completionScore,
    breakdown: [
      { label: '角度匹配度', score: angleScore, max: 30 },
      { label: '动作平滑度', score: smoothScore, max: 20 },
      { label: '训练完成度', score: completionScore, max: 10 }
    ],
    level: getLevel(total),
    noTemplate: true
  };
}

/** Real-time guidance during training */
function getGuidance(currentAngle, targetMin, targetMax) {
  if (currentAngle < targetMin - 20) return '↑ 大幅伸展';
  if (currentAngle < targetMin - 5) return '↗ 再伸展一点';
  if (currentAngle > targetMax + 20) return '↓ 大幅收回';
  if (currentAngle > targetMax + 5) return '↘ 稍微收回';
  if (currentAngle >= targetMin && currentAngle <= targetMax) return '✓ 角度完美';
  return '';
}

function getLevel(score) {
  if (score >= 90) return { text: '优秀', color: '#2ecc71', emoji: '🌟🌟🌟' };
  if (score >= 75) return { text: '良好', color: '#27ae60', emoji: '🌟🌟' };
  if (score >= 60) return { text: '一般', color: '#f39c12', emoji: '🌟' };
  return { text: '需改进', color: '#e74c3c', emoji: '💪' };
}

/**
 * Detailed real-time guidance using standard template data.
 * Returns { text, type: 'good'|'hint'|'warn' }
 */
function getDetailedGuidance(currentAngle, angleHistory, standardTemplate, targetMin, targetMax) {
  // 1. Zone check (highest priority)
  if (currentAngle < targetMin - 20) return { text: '↑ 大幅伸展手臂', type: 'warn' };
  if (currentAngle < targetMin - 5) return { text: '↗ 再伸展一点就到位了', type: 'hint' };
  if (currentAngle > targetMax + 20) return { text: '↓ 手臂弯太多了，伸开', type: 'warn' };
  if (currentAngle > targetMax + 5) return { text: '↘ 稍微收一点，角度偏大', type: 'hint' };
  if (currentAngle >= targetMin && currentAngle <= targetMax) return { text: '✓ 角度完美，保持稳定！', type: 'good' };

  // 2. Speed check using standard curve (if available)
  if (standardTemplate && standardTemplate.standardAngleCurve && standardTemplate.standardAngleCurve.length > 3) {
    var stdCurve = standardTemplate.standardAngleCurve;
    // Compare patient's recent speed with standard speed
    if (angleHistory && angleHistory.length >= 3) {
      var patSpeed = Math.abs(angleHistory[angleHistory.length - 1] - angleHistory[angleHistory.length - 3]) / 2;
      // Find matching position in standard curve
      var closestIdx = 0;
      var closestDist = Infinity;
      for (var i = 0; i < stdCurve.length - 2; i++) {
        var dist = Math.abs(stdCurve[i] - currentAngle);
        if (dist < closestDist) { closestDist = dist; closestIdx = i; }
      }
      if (closestIdx < stdCurve.length - 2) {
        var stdSpeed = Math.abs(stdCurve[closestIdx + 2] - stdCurve[closestIdx]) / 2;
        if (stdSpeed > 1) {
          if (patSpeed < stdSpeed * 0.3) return { text: '🐢 动作太慢，加快节奏', type: 'hint' };
          if (patSpeed > stdSpeed * 2.5) return { text: '🐇 慢一点，控制好节奏', type: 'hint' };
        }
      }
    }
  }

  // 3. Smoothness check
  if (angleHistory && angleHistory.length >= 4) {
    var len = angleHistory.length;
    var jerk = Math.abs(angleHistory[len-1] - 2*angleHistory[len-2] + angleHistory[len-3]);
    if (jerk > 12) return { text: '⚠ 动作抖动，保持流畅', type: 'warn' };
  }

  return { text: '继续做动作...', type: 'hint' };
}

module.exports = { score, getGuidance, getDetailedGuidance, getLevel };
