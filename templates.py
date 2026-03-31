def build_reply_suggestion(text, risky_spans, docs, route):
    if route == "escalate":
        return {
            "action": "升级人工主管处理",
            "reply": "您好，已收到您的反馈，我们将优先安排专员核查并尽快联系您处理。"
        }

    if route == "manual_review":
        return {
            "action": "进入人工复核队列",
            "reply": "您好，您的问题已提交处理，我们会尽快为您核实并回复。"
        }

    return {
        "action": "自动回复",
        "reply": "您好，已收到您的问题，您可以先参考相关帮助信息，我们也会继续为您跟进。"
    }
