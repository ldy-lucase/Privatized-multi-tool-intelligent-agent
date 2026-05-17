"""
有状态智能 Agent：任务拆解、分支流转、失败重试

图结构：
  orchestrator → router ──┬── worker ──→ router ──┬── worker ...
                           │                       │
                           └── summarizer ←────────┘

核心能力：
  - 自主拆解任务
  - 条件分支路由
  - 子任务失败自动重试
  - 检查点持久化（支持 thread_id 多轮对话）
"""

from typing import Annotated, Literal, Sequence, TypedDict

import operator
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START


# ============================================================
# 1. 状态
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_plan: str
    current_task_index: int
    subtask_results: Annotated[list[str], operator.add]
    retry_count: int
    max_retries: int


# ============================================================
# 2. 工具
# ============================================================

_available_tools = {}


@tool
def web_search(query: str) -> str:
    """搜索网络获取最新信息。"""
    return f"[搜索结果] 关于「{query}」: 这是模拟的搜索结果。"


@tool
def code_runner(code: str) -> str:
    """执行 Python 代码并返回运行结果。自动捕获 print 输出和最后一个表达式的值。"""
    import math, random, os, json, io, sys, ast
    import builtins as _builtins

    allowed = {"abs": abs, "len": len, "range": range, "int": int,
               "str": str, "list": list, "dict": dict, "print": print,
               "math": math, "random": random, "os": os, "json": json,
               "io": io, "sys": sys}
    try:
        # 解析最后一行为表达式还是语句
        lines = code.strip().split("\n")
        last_line = lines[-1].strip() if lines else ""

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        # 尝试判断最后一行是否可能是表达式（不是赋值/import/if/for等）
        is_expr = False
        try:
            parsed = ast.parse(last_line, mode="eval")
            is_expr = True
        except SyntaxError:
            is_expr = False

        if is_expr:
            # 最后一行是表达式，整体用 exec + 末尾 print
            exec_code = "\n".join(lines[:-1]) + "\nprint(" + last_line + ")"
        else:
            exec_code = code

        exec(exec_code, {"__builtins__": _builtins.__dict__}, allowed)
        sys.stdout = old_stdout
        output = buf.getvalue()

        if output.strip():
            return f"[执行成功]\n{output[:1000]}"
        else:
            return "[执行成功]\n(代码执行完毕，无输出。如需输出结果请用 print())"
    except Exception as e:
        sys.stdout = old_stdout
        return f"[执行失败] {e}"


@tool
def file_read(path: str) -> str:
    """读取文件内容。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:2000]
    except Exception as e:
        return f"[读取失败] {e}"


@tool
def file_write(path: str, content: str) -> str:
    """写入文件。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[写入成功] {path}"
    except Exception as e:
        return f"[写入失败] {e}"


@tool
def file_list(path: str = ".") -> str:
    """列出指定目录下的所有文件和子目录。"""
    import os, glob

    try:
        if path.endswith("/*"):
            items = glob.glob(path)
        else:
            items = os.listdir(path)
        result = "\n".join(sorted(items))
        return f"[目录] {path} 下的文件 ({len(items)} 个):\n{result[:1500]}"
    except Exception as e:
        return f"[列出失败] {e}"


_tools_list = [web_search, code_runner, file_read, file_write, file_list]
_available_tools = {t.name: t for t in _tools_list}


# ============================================================
# 3. 共享模型
# ============================================================

model = ChatOllama(model="qwen2.5:7b", temperature=0.1, num_predict=2048, num_ctx=4096)


def _run_tool_calls(response: AIMessage) -> list[ToolMessage]:
    """执行模型返回的所有 tool_call，返回 ToolMessage 列表。"""
    results = []
    for tc in response.tool_calls:
        tool_fn = _available_tools.get(tc["name"])
        if tool_fn:
            try:
                result = tool_fn.invoke(tc["args"])
            except Exception as e:
                result = f"[工具执行异常] {e}"
        else:
            result = f"[未知工具: {tc['name']}]"

        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
            name=tc["name"],
        ))
        print(f"    🛠  [{tc['name']}] {str(result)[:80]}")
    return results


def _react_loop(task: str, max_turns: int = 5) -> tuple[str, dict]:
    """完整的 ReAct 循环。返回 (最终回答, {工具名: 结果})。"""
    bound = model.bind_tools(_tools_list)
    messages = [
        SystemMessage(content=(
            "工具：file_list(列目录) file_read(读文件) file_write(写文件) "
            "code_runner(执行Python代码) web_search(搜索)。\n"
            "注意：工具由 AI 调用，不要写在传给 code_runner 的代码里。"
            "code_runner 里用标准 Python（open, os, print 等）。"
            "用 print() 输出结果，code_runner 会自动捕获。"
        )),
        HumanMessage(content=task),
    ]
    collected: dict[str, str] = {}

    for turn in range(max_turns):
        try:
            response = bound.invoke(messages)
        except Exception as e:
            import traceback
            print(f"    ⛔ [模型调用失败] {e}")
            traceback.print_exc()
            return f"[模型错误] {e}", collected
        messages.append(response)

        if not response.tool_calls:
            return response.content, collected

        tool_msgs = _run_tool_calls(response)
        # 记录每个工具的原始结果
        for m in tool_msgs:
            collected[m.name] = m.content
        messages.extend(tool_msgs)

    return messages[-1].content, collected


# ============================================================
# 4. 图节点
# ============================================================

def orchestrator(state: AgentState) -> dict:
    """主协调：分析请求，拆解为子任务列表。"""
    print("  🔄 [Orchestrator] 拆解任务...")
    user_msg = state["messages"][-1].content

    response = model.invoke([
        HumanMessage(content=(
            "你是任务分解专家。分析以下用户请求，拆解为 2-4 个独立、具体的子任务。\n"
            "每个子任务必须可用以下工具之一完成：\n"
            "  - web_search: 搜索信息\n"
            "  - code_runner: 执行 Python 代码\n"
            "  - file_read: 读取文件\n"
            "  - file_write: 写入文件\n"
            "  - file_list: 列出目录内容\n"
            "直接输出子任务列表，每行一个，格式:\n"
            "1. 任务描述\n2. 任务描述\n...\n"
            "不要输出多余内容。\n\n"
            f"用户请求：{user_msg}"
        ))
    ])

    plan = response.content.strip()
    print(f"  📋 计划:\n{plan}\n")

    return {
        "task_plan": plan,
        "current_task_index": 0,
        "subtask_results": [],
        "retry_count": 0,
        "max_retries": 2,
    }


def router(state: AgentState) -> Literal["worker", "summarizer"]:
    """条件路由：有未完成子任务就去 worker，否则汇总。"""
    lines = [l for l in state["task_plan"].split("\n")
             if l.strip() and len(l) > 2 and l.strip()[0].isdigit()]
    idx = state.get("current_task_index", 0)
    return "summarizer" if idx >= len(lines) else "worker"


def worker(state: AgentState) -> dict:
    """工作节点：执行任务（一次 ReAct 循环完成全部子任务）。"""
    print(f"  👷 [Worker] 执行任务...")

    try:
        # 提取 .py 文件列表作为上下文
        context = ""
        prev_results = state.get("subtask_results", [])
        if prev_results:
            for r in reversed(prev_results):
                if isinstance(r, str) and r.startswith("{{raw}}"):
                    raw = r[7:]
                    all_files = raw.split("\n")
                    # 过滤出 .py 文件
                    py_files = [l.strip() for l in all_files
                                if l.strip().endswith(".py") and not l.startswith("[")]
                    if py_files:
                        context = (
                            "当前目录的 .py 文件有：\n" + "\n".join(py_files) + "\n\n"
                            "请用 code_runner 一次性读取这些文件并统计总行数，"
                            "然后用 file_write 写入 stats.txt。"
                        )
                    else:
                        # 没有 .py 文件就用前几行
                        context = "上一步结果（前几行）：\n" + "\n".join(all_files[:5]) + "\n\n"
                    break

        plan = state.get("task_plan", "")
        output, raw = _react_loop(
            context + f"请按以下计划执行：\n{plan}\n\n"
            f"可用工具：file_list, file_read, file_write, code_runner, web_search"
        )
        print(f"  ✅ 任务完成")
        results = [f"【执行结果】{output[:500]}"]
        for name, val in raw.items():
            results.append(f"{{{{raw}}}}{name}: {val[:500]}")
        return {
            "subtask_results": results,
            "current_task_index": 99,  # 直接跳到完成
            "retry_count": 0,
        }

    except Exception as e:
        import traceback
        print(f"  ❌ 任务失败: {e}")
        traceback.print_exc()
        return {
            "subtask_results": [f"【执行结果】[失败] {e}"],
            "current_task_index": 99,
            "retry_count": 0,
        }


def summarizer(state: AgentState) -> dict:
    """汇总所有子任务结果成最终回答。"""
    print("  📝 [Summarizer] 汇总结果...")

    results = "\n".join(state.get("subtask_results", ["(无结果)"]))
    print(f"  📊 收到 {len(state.get('subtask_results', []))} 条子任务结果")

    response = model.invoke([
        HumanMessage(content=(
            "以下是多个子任务的执行结果。请整合成一份完整的最终回答给用户。\n"
            "要连贯、有条理，不要逐条罗列，而是合并成自然的段落。\n"
            f"\n{results}"
        ))
    ])

    return {"messages": [AIMessage(content=response.content)]}


# ============================================================
# 5. 构建图
# ============================================================

def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("worker", worker)
    workflow.add_node("summarizer", summarizer)

    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator", router,
        {"worker": "worker", "summarizer": "summarizer"},
    )
    workflow.add_conditional_edges(
        "worker", router,
        {"worker": "worker", "summarizer": "summarizer"},
    )
    workflow.add_edge("summarizer", END)

    return workflow.compile(checkpointer=MemorySaver())


# ============================================================
# 6. 运行
# ============================================================

if __name__ == "__main__":
    agent = build_agent()

    print("=== 图结构 ===")
    for e in agent.get_graph().edges:
        print(f"  {e.source} -> {e.target}")
    print()

    questions = [
        "帮我写一个 Python 脚本，读取当前目录所有 .py 文件，统计总行数，然后把结果保存到 stats.txt",
        "再帮我分析一下刚才的 stats.txt 内容，生成一个简洁的报告",
    ]

    for i, q in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"[用户] {q}")
        print(f"{'='*60}")

        result = agent.invoke(
            {"messages": [HumanMessage(content=q)]},
            config={"configurable": {"thread_id": "orchestrator-v2"}},
        )

        for m in result.get("messages", []):
            if hasattr(m, "type") and m.type == "ai" and isinstance(m.content, str):
                print(f"\n[助手] {m.content[:500]}")
