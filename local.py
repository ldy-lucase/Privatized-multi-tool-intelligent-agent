"""Deep Agents with local Ollama model + 自定义工具."""

from langchain_ollama import ChatOllama
from deepagents import create_deep_agent


# ============================================================
# 工具 1: 中英翻译
# ============================================================
def translate(text: str, target_lang: str = "中文") -> str:
    """将文本翻译成指定语言。

    Args:
        text: 要翻译的文本。
        target_lang: 目标语言，如"中文"、"英文"、"日文"、"法文"等。

    Returns:
        翻译后的文本。
    """
    # 实际场景这里可以调 Google / DeepL API
    # 模拟真实翻译结果
    translations = {
        "中文": f"你好，\"{text}\" 的翻译结果如上。",
        "英文": f"Hello, the translation of \"{text}\" is as above.",
        "日文": f"こんにちは、「{text}」の翻訳結果は上記の通りです。",
        "法文": f"Bonjour, la traduction de \"{text}\" est comme ci-dessus.",
    }
    return translations.get(target_lang, f"[{target_lang}] {text}")


# ============================================================
# 工具 2: 文章摘要
# ============================================================
def summarize(text: str, max_length: int = 200) -> str:
    """对长文本生成摘要。

    Args:
        text: 需要摘要的原文。
        max_length: 摘要最大字数，默认 200。

    Returns:
        生成的摘要文本。
    """
    words = text[:500]
    return f"[摘要] {words[:max_length]}……（共 {len(text)} 字）"


# ============================================================
# 工具 3: 数学计算
# ============================================================
def calculate(expression: str) -> str:
    """执行数学计算并返回结果。

    支持的运算：+ - * /、括号、幂运算**、数学函数 sqrt() sin() cos() 等。

    Args:
        expression: 数学表达式，如 "3.14 * 2 ** 10" 或 "sqrt(144) + 8 * (3 + 2)"。

    Returns:
        计算结果。
    """
    import math

    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 工具 4: 代码解释
# ============================================================
def explain_code(code: str, language: str = "python") -> str:
    """对代码片段进行逐行解释，说明其功能和逻辑。

    Args:
        code: 需要解释的代码内容。
        language: 编程语言，如 python / javascript / java / go / rust 等。

    Returns:
        代码解释结果。
    """
    lines = code.strip().split("\n")
    return (
        f"[代码解释 - {language}]\n"
        f"代码共 {len(lines)} 行\n"
        f"---\n"
        f"这段代码的功能是：{lines[0][:60] if lines else ''}……\n"
        f"（实际场景可调 LLM 或静态分析引擎来生成解释）"
    )


# ============================================================
# 工具 5: SQL 生成
# ============================================================
def generate_sql(query_description: str, db_type: str = "postgresql") -> str:
    """根据自然语言描述生成 SQL 查询语句。

    Args:
        query_description: 自然语言描述，如"查询过去7天注册的用户，按注册时间倒序"。
        db_type: 数据库类型，如 postgresql / mysql / sqlite / bigquery 等。

    Returns:
        生成的 SQL 语句。
    """
    return (
        f"[SQL 生成 - {db_type}]\n"
        f"描述: {query_description}\n"
        f"---\n"
        f"-- 实际场景可调 LLM 或 SQL 模版引擎来生成\n"
        f"SELECT * FROM users\n"
        f"WHERE created_at >= NOW() - INTERVAL '7 days'\n"
        f"ORDER BY created_at DESC;"
    )


def main() -> None:
    # 1. 构建本地模型
    model = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.1,
        num_predict=4096,
    )

    # 2. 创建 Agent（传入 5 个自定义工具）
    agent = create_deep_agent(
        model=model,
        tools=[
            translate,
            summarize,
            calculate,
            explain_code,
            generate_sql,
        ],
        system_prompt=(
            "你是一个多功能 AI 助手，拥有翻译、摘要、计算、代码解释和 SQL 生成能力。\n"
            "当用户请求涉及这些功能时，请调用对应的工具。\n"
            "用简洁的中文回答。"
        ),
    )

    # 3. 多轮对话测试
    questions = [
        "帮我翻译「Hello, what is the weather like today?」成中文",
        "计算 (3.14 * 2 ** 10) + sqrt(144) 的结果",
        "解释一下这段代码：def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
        "生成一条 SQL，查询库存低于10的商品",
        "写一个 Python 函数，判断一个字符串是否是回文。",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print(f"[用户] {q}")
        print("=" * 60)

        result = agent.invoke({"messages": q})

        for msg in result["messages"]:
            role = msg.type.upper()
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if role == "AI":
                print(f"\n[助手]")
                print(content)
                print()
            elif role == "TOOL":
                print(f"\n  🛠  [{msg.name}]")
                print(f"  {content[:300]}...")
            elif role == "HUMAN":
                pass  # 用户输入已打印


if __name__ == "__main__":
    main()
