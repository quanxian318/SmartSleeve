// cloudfunctions/joinDoctorByQR/index.js
const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();

/**
 * Patient binds to a doctor via QR code scan.
 * Input: { doctorId: string } — parsed from mini-program code query
 */
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext();
  const patientOpenid = wxContext.OPENID;
  const { doctorId } = event;

  if (!doctorId) {
    return { success: false, error: '缺少医生标识' };
  }

  try {
    // Verify the doctor exists
    const doctorRes = await db.collection('users')
      .where({
        _openid: doctorId,
        role: 'doctor'
      })
      .get();

    if (doctorRes.data.length === 0) {
      return { success: false, error: '医生不存在或已注销' };
    }

    const doctor = doctorRes.data[0];

    // Prevent self-binding
    if (doctor._openid === patientOpenid) {
      return { success: false, error: '不能绑定自己' };
    }

    // Check if patient already has a doctor
    const patientRes = await db.collection('users')
      .where({ _openid: patientOpenid })
      .get();

    if (patientRes.data.length === 0) {
      return { success: false, error: '用户记录不存在' };
    }

    const patient = patientRes.data[0];
    if (patient.doctorId) {
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
    console.error('[joinDoctorByQR] Error:', e);
    return { success: false, error: e.message };
  }
};
