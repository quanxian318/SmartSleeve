// cloudfunctions/getDoctorPatients/index.js
const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();

/**
 * Get all patients for a doctor with summary stats.
 * Avoids N+1 client-side queries by doing everything server-side.
 */
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext();
  const doctorOpenid = wxContext.OPENID;

  try {
    // 1. Get all patients assigned to this doctor
    const patientRes = await db.collection('users')
      .where({
        doctorId: doctorOpenid,
        role: 'patient'
      })
      .field({
        _openid: true,
        customName: true,
        gender: true,
        height: true,
        weight: true,
        bmi: true,
        lastSensorData: true,
        lastSensorTime: true
      })
      .get();

    const patients = patientRes.data;
    if (patients.length === 0) {
      return { success: true, patients: [], count: 0 };
    }

    // 2. For each patient, get their latest training record and pending task count
    const enriched = [];

    for (const patient of patients) {
      // Latest training record
      let latestRecord = null;
      try {
        const recordRes = await db.collection('training_records')
          .where({ _openid: patient._openid })
          .orderBy('date', 'desc')
          .limit(1)
          .get();
        if (recordRes.data.length > 0) {
          latestRecord = recordRes.data[0];
        }
      } catch (e) {
        // Ignore record fetch errors for individual patients
      }

      // Pending task count
      let pendingTasks = 0;
      let totalTasks = 0;
      try {
        const taskRes = await db.collection('tasks')
          .where({
            patientId: patient._openid,
            doctorId: doctorOpenid
          })
          .get();
        totalTasks = taskRes.data.length;
        pendingTasks = taskRes.data.filter(t => t.status === 'pending').length;
      } catch (e) {
        // Ignore task fetch errors
      }

      enriched.push({
        _openid: patient._openid,
        customName: patient.customName || '患者',
        gender: patient.gender,
        height: patient.height,
        weight: patient.weight,
        bmi: patient.bmi,
        lastSensorData: patient.lastSensorData || null,
        lastSensorTime: patient.lastSensorTime || null,
        isOnline: patient.lastSensorTime
          ? (Date.now() - new Date(patient.lastSensorTime).getTime() < 30000)
          : false,
        latestRecord: latestRecord ? {
          date: latestRecord.date,
          maxEmg: latestRecord.maxEmg || 0,
          avgBpm: latestRecord.avgBpm || 0,
          visionAngle: latestRecord.visionAngle || 180
        } : null,
        pendingTasks: pendingTasks,
        totalTasks: totalTasks
      });
    }

    return {
      success: true,
      patients: enriched,
      count: enriched.length
    };
  } catch (e) {
    console.error('[getDoctorPatients] Error:', e);
    return { success: false, error: e.message, patients: [], count: 0 };
  }
};
