// cloudfunctions/manageTask/index.js
const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();

/**
 * Task CRUD operations with role-based access control.
 *
 * Actions:
 *   create   — doctor creates a task for a patient
 *   update   — doctor updates a task
 *   delete   — doctor deletes a task
 *   complete — patient marks a task as completed
 */
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext();
  const callerOpenid = wxContext.OPENID;
  const { action, taskId, data } = event;

  if (!action) {
    return { success: false, error: '缺少 action 参数' };
  }

  try {
    switch (action) {
      case 'create':
        return await handleCreate(callerOpenid, data);
      case 'update':
        return await handleUpdate(callerOpenid, taskId, data);
      case 'delete':
        return await handleDelete(callerOpenid, taskId);
      case 'complete':
        return await handleComplete(callerOpenid, taskId, data);
      default:
        return { success: false, error: '未知 action: ' + action };
    }
  } catch (e) {
    console.error('[manageTask] Error:', e);
    return { success: false, error: e.message };
  }
};

/**
 * Doctor creates a task for a patient.
 * data: { patientId, title, description, targetAngleMin, targetAngleMax, repetitions, deadline }
 */
async function handleCreate(doctorOpenid, data) {
  if (!data || !data.patientId) {
    return { success: false, error: '请选择目标患者' };
  }

  // Verify the doctor owns this patient (don't require role field — it may not be set yet)
  const patientRes = await db.collection('users')
    .where({
      _openid: data.patientId,
      doctorId: doctorOpenid
    })
    .get();

  if (patientRes.data.length === 0) {
    return { success: false, error: '该患者不是您的学员，请让患者先绑定您的邀请码' };
  }

  // Ensure patient has role set
  const patient = patientRes.data[0];
  if (!patient.role || patient.role !== 'patient') {
    await db.collection('users').where({ _openid: data.patientId }).update({
      data: { role: 'patient' }
    });
  }

  const task = {
    _openid: doctorOpenid,
    doctorId: doctorOpenid,
    patientId: data.patientId,
    patientName: patient.customName || '患者',
    title: data.title || '训练任务',
    description: data.description || '',
    targetAngleMin: Number(data.targetAngleMin) || 30,
    targetAngleMax: Number(data.targetAngleMax) || 150,
    repetitions: Number(data.repetitions) || 10,
    deadline: data.deadline ? new Date(data.deadline) : null,
    status: 'pending',
    templateId: data.templateId || null,
    createdAt: new Date(),
    completedAt: null,
    completedData: null
  };

  if (!task.title.trim()) {
    return { success: false, error: '请输入任务标题' };
  }

  const result = await db.collection('tasks').add({ data: task });
  return { success: true, taskId: result._id, task: task };
}

/**
 * Doctor updates a task.
 */
async function handleUpdate(doctorOpenid, taskId, data) {
  if (!taskId) {
    return { success: false, error: '缺少任务ID' };
  }

  // Verify doctor owns this task
  const taskRes = await db.collection('tasks').doc(taskId).get();
  if (!taskRes || !taskRes.data || (Array.isArray(taskRes.data) && taskRes.data.length === 0)) {
    return { success: false, error: '任务不存在' };
  }
  if (taskRes.data.doctorId !== doctorOpenid) {
    return { success: false, error: '无权修改此任务' };
  }

  const updateData = {};
  if (data.title !== undefined) updateData.title = data.title;
  if (data.description !== undefined) updateData.description = data.description;
  if (data.targetAngleMin !== undefined) updateData.targetAngleMin = Number(data.targetAngleMin);
  if (data.targetAngleMax !== undefined) updateData.targetAngleMax = Number(data.targetAngleMax);
  if (data.repetitions !== undefined) updateData.repetitions = Number(data.repetitions);
  if (data.deadline !== undefined) updateData.deadline = data.deadline ? new Date(data.deadline) : null;

  await db.collection('tasks').doc(taskId).update({ data: updateData });
  return { success: true, message: '任务已更新' };
}

/**
 * Doctor deletes a task.
 */
async function handleDelete(doctorOpenid, taskId) {
  if (!taskId) {
    return { success: false, error: '缺少任务ID' };
  }

  const taskRes = await db.collection('tasks').doc(taskId).get();
  if (!taskRes || !taskRes.data || (Array.isArray(taskRes.data) && taskRes.data.length === 0)) {
    return { success: false, error: '任务不存在' };
  }
  if (taskRes.data.doctorId !== doctorOpenid) {
    return { success: false, error: '无权删除此任务' };
  }

  await db.collection('tasks').doc(taskId).remove();
  return { success: true, message: '任务已删除' };
}

/**
 * Patient marks a task as complete.
 * data: { maxEmg, avgAngle, bpm }
 */
async function handleComplete(patientOpenid, taskId, data) {
  if (!taskId) {
    return { success: false, error: '缺少任务ID' };
  }

  const taskRes = await db.collection('tasks').doc(taskId).get();
  if (!taskRes || !taskRes.data || (Array.isArray(taskRes.data) && taskRes.data.length === 0)) {
    return { success: false, error: '任务不存在' };
  }

  const task = taskRes.data;
  if (task.patientId !== patientOpenid) {
    return { success: false, error: '此任务不属于您' };
  }
  if (task.status === 'completed') {
    return { success: false, error: '任务已完成，无需重复操作' };
  }

  await db.collection('tasks').doc(taskId).update({
    data: {
      status: 'completed',
      completedAt: new Date(),
      completedData: {
        maxEmg: data && data.maxEmg ? Number(data.maxEmg) : 0,
        avgAngle: data && data.avgAngle ? Number(data.avgAngle) : 0,
        bpm: data && data.bpm ? Number(data.bpm) : 0
      }
    }
  });

  return { success: true, message: '任务已完成！' };
}
