// cloudfunctions/setUserRole/index.js
const cloud = require('wx-server-sdk');
cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV });

const db = cloud.database();

/**
 * Set user role on first launch.
 * For doctors: generates a unique 6-char invite code.
 * Auto-creates user record if it doesn't exist yet.
 */
exports.main = async (event, context) => {
  const wxContext = cloud.getWXContext();
  const openid = wxContext.OPENID;
  const { role } = event;

  if (!role || (role !== 'doctor' && role !== 'patient')) {
    return { success: false, error: '无效的角色类型，必须是 doctor 或 patient' };
  }

  try {
    // Check if user record exists
    const userRes = await db.collection('users')
      .where({ _openid: openid })
      .get();

    let user;

    if (userRes.data.length === 0) {
      // User record doesn't exist — create one now
      console.log('[setUserRole] Creating new user record for:', openid);
      await db.collection('users').add({
        data: {
          _openid: openid,
          customName: '微信用户',
          createTime: new Date(),
          role: role,
          inviteCode: role === 'doctor' ? await generateUniqueCode(db) : '',
          gender: 0,
          height: 175,
          weight: 70,
          bmi: 22.9
        }
      });

      const result = { success: true, role: role };
      if (role === 'doctor') {
        // Re-read to get the inviteCode
        const newUser = await db.collection('users').where({ _openid: openid }).get();
        result.inviteCode = newUser.data[0].inviteCode;
      }
      return result;
    }

    user = userRes.data[0];

    // If role is already set, return success with existing data (idempotent)
    if (user.role) {
      console.log('[setUserRole] Role already set:', user.role);
      return {
        success: true,
        role: user.role,
        inviteCode: user.inviteCode || null,
        message: '角色已设置'
      };
    }

    if (role === 'doctor') {
      const code = await generateUniqueCode(db);

      await db.collection('users').where({ _openid: openid }).update({
        data: {
          role: 'doctor',
          inviteCode: code
        }
      });

      return { success: true, role: 'doctor', inviteCode: code };
    } else {
      await db.collection('users').where({ _openid: openid }).update({
        data: {
          role: 'patient'
        }
      });

      return { success: true, role: 'patient' };
    }
  } catch (e) {
    console.error('[setUserRole] Error:', e);
    return { success: false, error: e.message };
  }
};

/**
 * Generate a 6-character alphanumeric code, ensuring uniqueness.
 */
async function generateUniqueCode(db) {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let attempts = 0;

  while (attempts < 20) {
    let code = '';
    for (let i = 0; i < 6; i++) {
      code += chars[Math.floor(Math.random() * chars.length)];
    }

    const res = await db.collection('users')
      .where({ inviteCode: code })
      .get();

    if (res.data.length === 0) {
      return code;
    }
    attempts++;
  }

  throw new Error('无法生成唯一邀请码，请重试');
}
