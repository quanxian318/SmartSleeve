// cloudfunctions/joinDoctor/index.js
const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();

/**
 * Patient binds to a doctor via invite code.
 * Input: { inviteCode: string }
 */
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext();
  const patientOpenid = wxContext.OPENID;
  const { inviteCode } = event;

  if (!inviteCode || typeof inviteCode !== 'string' || inviteCode.length !== 6) {
    return { success: false, error: '请输入有效的6位邀请码' };
  }

  const code = inviteCode.toUpperCase();

  try {
    // Find the doctor with this invite code
    const doctorRes = await db.collection('users')
      .where({
        inviteCode: code,
        role: 'doctor'
      })
      .get();

    if (doctorRes.data.length === 0) {
      return { success: false, error: '邀请码无效，请检查后重试' };
    }

    const doctor = doctorRes.data[0];

    // Prevent self-binding
    if (doctor._openid === patientOpenid) {
      return { success: false, error: '不能绑定自己' };
    }

    // Find or create patient record
    const patientRes = await db.collection('users')
      .where({ _openid: patientOpenid })
      .get();

    let patient;
    if (patientRes.data.length === 0) {
      // Auto-create patient record (cloud may not have it yet if setUserRole wasn't deployed)
      console.log('[joinDoctor] Creating patient record for:', patientOpenid);
      await db.collection('users').add({
        data: {
          _openid: patientOpenid,
          customName: '患者',
          createTime: new Date(),
          role: 'patient',
          doctorId: doctor._openid,
          doctorName: doctor.customName || '医生',
          gender: 0,
          height: 175,
          weight: 70,
          bmi: 22.9
        }
      });
      return {
        success: true,
        doctorId: doctor._openid,
        doctorName: doctor.customName || '医生'
      };
    }

    patient = patientRes.data[0];
    if (patient.doctorId) {
      // Already bound — check if it's the same doctor
      if (patient.doctorId === doctor._openid) {
        return { success: true, message: '已绑定该医生', doctorId: doctor._openid, doctorName: doctor.customName };
      }
      return { success: false, error: '您已绑定其他医生，如需更换请联系当前医生解除绑定' };
    }

    // Perform the binding — also ensure role is set
    await db.collection('users').where({ _openid: patientOpenid }).update({
      data: {
        role: 'patient',
        doctorId: doctor._openid,
        doctorName: doctor.customName || '医生'
      }
    });

    return {
      success: true,
      doctorId: doctor._openid,
      doctorName: doctor.customName || '医生'
    };
  } catch (e) {
    console.error('[joinDoctor] Error:', e);
    return { success: false, error: e.message };
  }
};
