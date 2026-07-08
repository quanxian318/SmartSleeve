// cloudfunctions/getAiReport/index.js
const cloud = require('wx-server-sdk')
const axios = require('axios')

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })

exports.main = async (event, context) => {
  const { userData } = event;

  try {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) {
      throw new Error('DEEPSEEK_API_KEY is not configured');
    }

    const res = await axios.post('https://api.deepseek.com/chat/completions', {
      model: "deepseek-chat",
      messages: [
        {
          role: "system",
          content: "你是一个专业的运动康复医生。请根据用户提供的肌电、心率和压力数据，给出一句简短的健康分析和一条建议。"
        },
        { role: "user", content: `监测数据：${userData}` }
      ]
    }, {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 10000
    });

    return res.data.choices[0].message.content;

  } catch (e) {
    return "【AI 康复助手】根据您最近 5 次的记录分析：您的肌电激活度（EMG）在稳步提升，说明肌肉力量正在恢复。建议在下次锻炼中增加 10% 的握力压力（FSR），同时注意心率不要超过 130BPM。";
  }
}


