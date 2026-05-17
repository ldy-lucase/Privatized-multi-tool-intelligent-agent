"""
Deep Agent Orchestrator — 用 create_deep_agent 实现复杂任务拆解与执行

模型：阿里云百炼 (DashScope) API
需要设置环境变量 DASHSCOPE_API_KEY，或在下面直接填写 api_key。
"""

from deepagents import (
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends.local_shell import LocalShellBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# 1. 模型 — 阿里云百炼 (DashScope)
# ============================================================

# 可用模型: qwen-max, qwen-plus, qwen-turbo, qwen2.5-72b-instruct 等
# 填好 api_key 或设置环境变量 DASHSCOPE_API_KEY

model = ChatOpenAI(
    model="qwen3.6-flash",
    temperature=0.1,
    api_key="sk-ba6431e8e5354c82b0929ae2a2fc5e57",          # ← 你的 DashScope API Key
    # 或从环境变量读取（推荐）：
    # api_key="${DASHSCOPE_API_KEY}",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# 2. 配置 — 排除不需要的内置工具
# ============================================================

# 远程大模型能力强，不需要排除太多工具
# 只排除对本 demo 多余的即可
register_harness_profile(
    "openai",  # ChatOpenAI 的 provider 标识
    HarnessProfile(
        excluded_tools=frozenset(
            {
                "write_todos",
                "task",  # 本 demo 不需要子代理
            }
        )
    ),
)

# ============================================================
# 3. 后端 — 本地 shell（同时支持文件操作 + 命令执行）
# ============================================================

# LocalShellBackend 直接在本机执行命令，同时支持文件操作
backend = LocalShellBackend()

# ============================================================
# 4. 构建 Agent
# ============================================================

SYSTEM_PROMPT = """你是任务分解型 AI 助手。对于复杂请求：

## 执行流程
1. 拆解为用户请求为 2-3 个步骤
2. 用 ls 列出目录内容（路径用绝对路径 /home/yun/deepagents/）
3. 用 execute(command="python3 -c '...'") 执行 python 代码处理数据
4. 用 write_file 写入最终结果
5. 用 read_file 验证结果

## 规则
- execute 的参数名是 command，不是 python3
- 所有 .py 文件处理用一条 python3 -c 命令完成，不要逐个 read_file
- 文件路径用绝对路径 /home/yun/deepagents/
- 用 glob 查找文件：execute(command="python3 -c \"import glob; print('\\n'.join(glob.glob('/home/yun/deepagents/**/*.py', recursive=True)))\"")
- 步骤完成后直接做下一步，不要停顿
- 全部完成后给用户一个完整总结"""

def build_agent():
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        middleware=[],
    )


# ============================================================
# 5. 运行
# ============================================================

if __name__ == "__main__":
    agent = build_agent()

    print("=== Agent 图结构 ===")
    for nid, node in agent.get_graph().nodes.items():
        print(f"  {nid}")
    print()

    questions = [
        "帮我写一个 Python 脚本，读取 /home/yun/deepagents/ 下所有 .py 文件，统计总行数，然后把结果保存到 /home/yun/deepagents/stats.txt",
        "再帮我分析一下刚才的 /home/yun/deepagents/stats.txt 内容，生成一个简洁的报告",
    ]

    for i, q in enumerate(questions):
        print(f"\n{'=' * 60}")
        print(f"[用户] {q}")
        print(f"{'=' * 60}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"configurable": {"thread_id": "deep-agent-demo"}},
        )

        for msg in result["messages"]:
            role = getattr(msg, "type", "").upper()
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if role == "AI" and content:
                print(f"\n[助手] {content[:500]}")
            elif role == "TOOL":
                name = getattr(msg, "name", "?")
                print(f"\n  🛠  [{name}] {content[:120]}...")
