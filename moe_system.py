import openai
import os


class MoEAgentSystem:
    def __init__(self, api_key, base_url, model_name="gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.experts_config = {
            "MATH": {"name": "数学专家", "prompt": "你是一个精通逻辑运算的数学老师，请分步骤解答数学题。"},
            "CODE": {"name": "代码专家", "prompt": "你是一个高级程序员，请提供清晰的代码实现及注释。"},
            "SUMMARY": {"name": "摘要专家", "prompt": "你是一个信息提取专家，请用简洁的要点总结用户提供的文本内容。"},
            "QA": {"name": "百科专家", "prompt": "你是一个知识广博的助手，请用准确平实的语言回答常识性问题。"}
        }

    def router(self, user_input):
        """路由模块：判断任务类型"""
        print(f"[Router] 正在分析输入意图...")
        router_prompt = f"""
        请分析以下用户输入，并将其归类为以下四种类型之一：
        - MATH: 涉及计算、逻辑证明、数学题。
        - CODE: 涉及编程、脚本、代码查错、算法。
        - SUMMARY: 涉及长文本压缩、提取摘要、要点罗列。
        - QA: 涉及常识问答、闲聊、定义查询。

        只需返回一个单词（MATH, CODE, SUMMARY, QA），不要说其他任何话。
        输入内容：{user_input}
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": router_prompt}]
        )
        choice = response.choices[0].message.content.strip().upper()
        # 简单清洗，防止模型输出包含标点
        for key in self.experts_config.keys():
            if key in choice:
                return key
        return "QA"

    def run_task(self, user_input):
        """执行流程：Input -> Router -> Expert -> Output"""
        # 1. 路由选择
        expert_key = self.router(user_input)
        expert = self.experts_config[expert_key]
        print(f"[System] 已分配给专家: {expert['name']} ({expert_key})")

        # 2. 调用专家
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": expert["prompt"]},
                {"role": "user", "content": user_input}
            ]
        )
        return expert_key, expert["name"], response.choices[0].message.content


# --- 运行测试 ---
if __name__ == "__main__":
    # 配置区：请在此处填写你的 API Key 和 Base URL
    MY_API_KEY = "your_api key"
    MY_BASE_URL = "https://api.deepseek.com"  # 如果用DeepSeek请换成 https://api.deepseek.com
    MY_MODEL = "deepseek-chat"  # 如果用DeepSeek请换成 deepseek-chat

    # 初始化系统
    moe = MoEAgentSystem(api_key=MY_API_KEY, base_url=MY_BASE_URL, model_name=MY_MODEL)

    # 2. 初始化系统
    moe = MoEAgentSystem(api_key=MY_API_KEY, base_url=MY_BASE_URL, model_name=MY_MODEL)

    print("========================================")
    print("   欢迎使用 MoE 多专家 Agent 系统")
    print("   输入 'quit' 或 'exit' 退出程序")
    print("========================================\n")

    # 3. 开启交互循环
    while True:
        # 获取用户输入
        user_query = input("用户 >>> ").strip()

        # 检查退出条件
        if user_query.lower() in ['quit', 'exit', '退出', 'q']:
            print("系统已退出，再见！")
            break

        # 处理空输入
        if not user_query:
            continue

        try:
            # 运行 MoE 逻辑
            print("-" * 30)
            expert_key, expert_name, response = moe.run_task(user_query)

            # 输出结果
            print(f"\n[路由结果]: 任务已分配给 -> {expert_name} ({expert_key})")
            print(f"[专家回答]: \n{response}")
            print("-" * 30 + "\n")

        except Exception as e:
            print(f"\n[错误]: 处理请求时发生异常: {e}\n")
