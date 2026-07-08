// cloudfunctions/fetchMyTasks/index.js
const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();

/**
 * Patient gets their own tasks.
 * Cloud functions bypass client-side DB permission restrictions,
 * so patients can read tasks created by their doctor.
 */
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext();
  const patientOpenid = wxContext.OPENID;
  const { status, limit } = event;

  try {
    // Build query: find tasks where patientId matches this user
    const where = { patientId: patientOpenid };

    // Optional status filter
    if (status && (status === 'pending' || status === 'completed')) {
      where.status = status;
    }

    const taskRes = await db.collection('tasks')
      .where(where)
      .orderBy('createdAt', 'desc')
      .limit(limit || 50)
      .get();

    // Enrich with deadline computation
    const now = new Date();
    const tasks = taskRes.data.map(task => {
      let deadlineText = '';
      let isOverdue = false;
      if (task.deadline) {
        const dl = new Date(task.deadline);
        const diffTime = dl.getTime() - now.getTime();
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        if (diffDays < 0) {
          deadlineText = '已逾期 ' + Math.abs(diffDays) + ' 天';
          isOverdue = true;
        } else if (diffDays === 0) {
          deadlineText = '今天截止';
        } else {
          deadlineText = '剩余 ' + diffDays + ' 天';
        }
      } else {
        deadlineText = '无截止日期';
      }
      return Object.assign({}, task, { deadlineText, isOverdue });
    });

    return {
      success: true,
      tasks: tasks,
      count: tasks.length,
      pendingCount: tasks.filter(t => t.status === 'pending').length
    };
  } catch (e) {
    console.error('[fetchMyTasks] Error:', e);
    return { success: false, error: e.message, tasks: [], count: 0 };
  }
};
