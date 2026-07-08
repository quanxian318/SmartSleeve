const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();

exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext();
  const callerOpenid = wxContext.OPENID;
  const { action, data, templateId } = event;

  try {
    switch (action) {
      case 'save':
        return await handleSave(callerOpenid, data);
      case 'list':
        return await handleList(callerOpenid);
      case 'get':
        return await handleGet(templateId);
      case 'delete':
        return await handleDelete(callerOpenid, templateId);
      default:
        return { success: false, error: '未知 action' };
    }
  } catch (e) {
    return { success: false, error: e.message };
  }
};

async function handleSave(doctorOpenid, data) {
  if (!data || !data.actionName) {
    return { success: false, error: '缺少动作名称' };
  }
  const template = {
    _openid: doctorOpenid,
    doctorId: doctorOpenid,
    actionName: data.actionName,
    actionType: data.actionType || 'custom',
    targetAngleMin: data.targetAngleMin || 50,
    targetAngleMax: data.targetAngleMax || 140,
    standardEmgBiceps: data.standardEmgBiceps || 800,
    standardEmgTriceps: data.standardEmgTriceps || 300,
    standardAngleCurve: data.standardAngleCurve || [],
    standardEmgCurve: data.standardEmgCurve || [],
    repetitions: data.repetitions || 10,
    createdAt: new Date()
  };
  const result = await db.collection('action_templates').add({ data: template });
  return { success: true, templateId: result._id };
}

async function handleList(doctorOpenid) {
  const res = await db.collection('action_templates')
    .where({ doctorId: doctorOpenid })
    .orderBy('createdAt', 'desc')
    .field({ actionName: true, actionType: true, targetAngleMin: true, targetAngleMax: true, repetitions: true, createdAt: true })
    .limit(50)
    .get();
  return { success: true, templates: res.data };
}

async function handleGet(templateId) {
  if (!templateId) return { success: false, error: '缺少模板ID' };
  const res = await db.collection('action_templates').doc(templateId).get();
  return { success: true, template: res.data };
}

async function handleDelete(doctorOpenid, templateId) {
  if (!templateId) return { success: false, error: '缺少模板ID' };
  const res = await db.collection('action_templates').doc(templateId).get();
  if (!res.data || res.data.doctorId !== doctorOpenid) {
    return { success: false, error: '无权删除' };
  }
  await db.collection('action_templates').doc(templateId).remove();
  return { success: true };
}
