#!/usr/bin/env python3
"""
voice_llm.py — DeepSeek API 客户端
  用于语音助手 v7 的自然对话 + 训练总结生成

用法:
  from voice_llm import DeepSeekClient
  client = DeepSeekClient()
  reply = client.natural_chat("你好")
  summary = client.generate_training_summary(training_data, validation_data)

API Key:
  从环境变量 DEEPSEEK_API_KEY 读取，或构造时传入。
  未配置时所有方法静默返回 None。
"""

import os, json, time, urllib.request, urllib.error

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 15  # 秒

SYSTEM_PROMPT_CHAT = (
    "你是智能健康袖套的语音助手，帮助用户进行肘关节康复训练。"
    "回答要简短口语化，不超过三句话，语气亲切专业。"
    "用户可能问训练建议、肌肉状态、康复知识等问题。"
)

SYSTEM_PROMPT_REPORT = (
    "你是一位专业的运动康复医生。请根据患者提供的训练数据，"
    "用3-5句话给出专业评估和下一步建议。语言简洁口语化，适合语音播报。"
)

SUMMARY_PROMPT_TEMPLATE = """请根据以下康复训练数据，用3-5句话给出专业评估和建议（中文，适合语音播报）：

- 有效训练时长：{effective_min}分{effective_sec}秒
- 总时长（含暂停）：{total_min}分{total_sec}秒
- 完成次数：{rep_count}次（目标{target_reps}次）
- 暂停次数：{pause_count}次
- 关节活动范围：{min_angle}° - {max_angle}°（平均{avg_angle}°）
- 二头肌平均激活：{avg_b} μV
- 三头肌平均激活：{avg_t} μV

请评估：
1. 训练完成质量如何
2. 关节活动度是否正常
3. 肌肉激活模式是否合理
4. 下一步训练建议"""


class DeepSeekClient:
    """DeepSeek API 客户端，所有方法在网络错误时静默返回 None"""

    def __init__(self, api_key=None, model=None, timeout=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "") or self._read_key_file()
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.timeout = timeout or DEFAULT_TIMEOUT
        self._history = []          # 多轮对话上下文 (最多 20 条)
        self._max_history = 20
        self._last_error_time = 0   # 静默降级：60s 内只打印一次错误
        self._api_failed = False    # API 是否已失败过（只提示一次）

    @staticmethod
    def _read_key_file():
        key_file = os.path.expanduser("~/.deepseek_key")
        if os.path.isfile(key_file):
            try:
                with open(key_file) as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    # ==================== 公开方法 ====================

    def is_available(self):
        """API 密钥是否已配置"""
        return bool(self.api_key)

    def natural_chat(self, user_text):
        """自然对话，保留上下文。失败返回 None"""
        if not self.api_key:
            return None

        messages = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
        # 最近 10 轮对话上下文
        messages.extend(self._history[-20:])
        messages.append({"role": "user", "content": user_text})

        reply = self._call_api(messages)
        if reply:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": reply})
            # 保持历史在限制内
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        return reply

    def generate_training_summary(self, training_data, validation_data=None):
        """生成训练总结。失败返回 None"""
        if not self.api_key or not training_data:
            return None

        # 构建 prompt
        eff_sec = training_data.get("effective_duration_sec", 0)
        tot_sec = training_data.get("total_duration_sec", eff_sec)

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            effective_min=int(eff_sec // 60),
            effective_sec=int(eff_sec % 60),
            total_min=int(tot_sec // 60),
            total_sec=int(tot_sec % 60),
            rep_count=training_data.get("rep_count", 0),
            target_reps=training_data.get("target_reps", 0),
            pause_count=training_data.get("pause_count", 0),
            min_angle=training_data.get("min_angle", 0),
            max_angle=training_data.get("max_angle", 0),
            avg_angle=training_data.get("avg_angle", 0),
            avg_b=training_data.get("avg_emg_biceps", 0),
            avg_t=training_data.get("avg_emg_triceps", 0),
        )

        # 附加代偿/电极信息
        extra = []
        if training_data.get("compensation_count", 0) > 0:
            extra.append(f"训练中出现{training_data['compensation_count']}次肌肉代偿告警。")
        if training_data.get("electrode_issue_count", 0) > 0:
            extra.append(f"训练中出现{training_data['electrode_issue_count']}次电极接触异常。")
        if training_data.get("form_issue_count", 0) > 0:
            extra.append(f"训练中出现{training_data['form_issue_count']}次动作不标准提示。")
        if extra:
            prompt += "\n" + "".join(extra)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_REPORT},
            {"role": "user", "content": prompt},
        ]
        return self._call_api(messages)

    def reset_history(self):
        """清空对话历史（会话结束时调用）"""
        self._history = []

    def reset_failure_flag(self):
        """重置 API 失败标记（下次失败会再次提示）"""
        self._api_failed = False

    # ==================== 内部方法 ====================

    def _call_api(self, messages):
        """发送 API 请求，返回文本或 None"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300,
        }
        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(DEEPSEEK_API_URL, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return content.strip() if content else None

        except urllib.error.HTTPError as e:
            self._log_error(f"HTTP {e.code}: {e.reason}")
            return None
        except urllib.error.URLError as e:
            self._log_error(f"网络错误: {e.reason}")
            return None
        except Exception as e:
            self._log_error(f"API 调用异常: {e}")
            return None

    def _log_error(self, msg):
        """静默降级：60s 内只打印一次错误，语音只提示一次"""
        now = time.time()
        if now - self._last_error_time > 60:
            print(f"  [LLM] {msg}")
            self._last_error_time = now


# ==================== 自测 ====================

def _self_test():
    """独立测试：验证 API 连通性"""
    import sys

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("自测跳过: 未设置 DEEPSEEK_API_KEY")
        print("用法: DEEPSEEK_API_KEY=sk-xxx python3 voice_llm.py")
        sys.exit(0)

    client = DeepSeekClient(api_key=api_key)
    print(f"模型: {client.model}")

    # 测试 1: 自然对话
    print("\n--- 测试: natural_chat ---")
    reply = client.natural_chat("你好，简单介绍一下你自己")
    if reply:
        print(f"回复: {reply}")
    else:
        print("失败: 无回复")

    # 测试 2: 训练总结
    print("\n--- 测试: generate_training_summary ---")
    mock_data = {
        "effective_duration_sec": 165,
        "total_duration_sec": 180,
        "rep_count": 25,
        "target_reps": 30,
        "pause_count": 2,
        "min_angle": 55,
        "max_angle": 135,
        "avg_angle": 95,
        "avg_emg_biceps": 420,
        "avg_emg_triceps": 180,
        "compensation_count": 3,
        "electrode_issue_count": 0,
        "form_issue_count": 2,
    }
    summary = client.generate_training_summary(mock_data)
    if summary:
        print(f"总结: {summary}")
    else:
        print("失败: 无回复")

    print("\n自测完成")


if __name__ == "__main__":
    _self_test()
