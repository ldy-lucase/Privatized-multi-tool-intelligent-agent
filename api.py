"""FastAPI backend: 将 Deep Agents Agent 封装为 HTTP API。"""

import math
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

# ============================================================
# 自定义工具（与 local.py 一致）
# ============================================================


def translate(text: str, target_lang: str = "中文") -> str:
    """将文本翻译成指定语言。

    Args:
        text: 要翻译的文本。
        target_lang: 目标语言，如"中文"、"英文"、"日文"、"法文"等。

    Returns:
        翻译后的文本。
    """
    import json as _json
    import urllib.request as _req

    body = _json.dumps({
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": f"将以下文本翻译成{target_lang}，只输出翻译结果不要多余内容：{text}"}],
        "stream": False,
        "options": {"num_predict": 128, "temperature": 0.1},
    }).encode()
    try:
        resp = _req.urlopen(_req.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=15)
        data = _json.loads(resp.read().decode())
        return data["message"]["content"].strip()
    except Exception as e:
        return f"[翻译错误] {e}"


def summarize(text: str, max_length: int = 200) -> str:
    """对长文本生成摘要。

    Args:
        text: 需要摘要的原文。
        max_length: 摘要最大字数，默认 200。

    Returns:
        生成的摘要文本。
    """
    return f"[摘要] {text[:max_length]}……（共 {len(text)} 字）"


def calculate(expression: str) -> str:
    """执行数学计算并返回结果。支持 + - * / () ** sqrt() sin() cos() tan() log()。

    Args:
        expression: 数学表达式，如 "3.14 * 2 ** 10"。

    Returns:
        计算结果。
    """
    allowed = {
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
        r = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"{expression} = {r}"
    except Exception as e:
        return f"计算错误: {e}"


def explain_code(code: str, language: str = "python") -> str:
    """对代码片段进行逐行解释，说明其功能和逻辑。

    Args:
        code: 需要解释的代码内容。
        language: 编程语言。

    Returns:
        代码解释结果。
    """
    lines = code.strip().split("\n")
    return f"[代码解释 - {language}]\n代码共 {len(lines)} 行\n功能：{lines[0][:60]}……"


def generate_sql(query_description: str, db_type: str = "postgresql") -> str:
    """根据自然语言描述生成 SQL 查询语句。

    Args:
        query_description: 自然语言描述。
        db_type: 数据库类型。

    Returns:
        生成的 SQL 语句。
    """
    return (
        f"[SQL 生成 - {db_type}]\n"
        f"描述: {query_description}\n"
        f"---\n"
        f"SELECT * FROM users\n"
        f"WHERE created_at >= NOW() - INTERVAL '7 days'\n"
        f"ORDER BY created_at DESC;"
    )


# ============================================================
# 全局：模型 + Agent（只初始化一次）
# ============================================================

_model: ChatOllama | None = None
_agent = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _agent  # noqa: PLW0603
    _model = ChatOllama(model="qwen2.5:3b", temperature=0.1, num_predict=4096)
    _agent = create_agent(
        model=_model,
        tools=[translate, summarize, calculate, explain_code, generate_sql],
        system_prompt=(
            "你是一个多功能 AI 助手，拥有翻译、摘要、计算、代码解释和 SQL 生成能力。\n"
            "当用户请求涉及这些功能时，请调用对应的工具。\n"
            "用简洁的中文回答。"
        ),
        middleware=[TodoListMiddleware()],
    )
    yield


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Deep Agents Local API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存会话存储: {session_id: session_data}
# session_data = {
#     "messages": [...],
#     "title": str,          # 第一句用户消息（会话标题）
#     "created_at": float,   # 创建时间戳
#     "updated_at": float,   # 最后更新时间戳
# }
sessions: dict[str, dict] = {}


# --- Pydantic 请求/响应模型 ---


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


class HistoryItem(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryItem]


class SessionInfo(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: float
    updated_at: float


class SessionsResponse(BaseModel):
    sessions: list[SessionInfo]


# --- 端点 ---


@app.get("/")
def root() -> dict:
    return {"service": "Deep Agents Local API", "status": "running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """发送一条消息给 Agent，返回 AI 回复。"""
    if _agent is None:
        raise HTTPException(503, "Agent not initialized")

    # 创建或复用会话
    sid = req.session_id or str(uuid.uuid4())
    now = time.time()
    if sid not in sessions:
        sessions[sid] = {
            "messages": [],
            "title": req.message[:40] + ("…" if len(req.message) > 40 else ""),
            "created_at": now,
            "updated_at": now,
        }

    session_data = sessions[sid]

    # 添加用户消息到会话
    session_data["messages"].append({"role": "human", "content": req.message})
    session_data["updated_at"] = time.time()

    # 将会话历史转为 LangChain 消息对象
    lc_messages = _to_langchain_messages(session_data["messages"])

    # 调用 Agent
    result = _agent.invoke({"messages": lc_messages})
    updated = result["messages"]

    # 提取新增的 AI 回复
    ai_reply = ""
    for m in updated:
        if m.type == "ai" and hasattr(m, "content") and m.content:
            ai_reply = m.content

    # 更新会话存储（同步新增的消息）
    session_data["messages"] = _from_langchain_messages(updated)
    session_data["updated_at"] = time.time()

    return ChatResponse(session_id=sid, response=ai_reply)


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    """获取指定会话的完整聊天记录。"""
    if session_id not in sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    messages = sessions[session_id]["messages"]
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryItem(role=m["role"], content=m["content"]) for m in messages],
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str) -> dict:
    """清空指定会话。"""
    if session_id not in sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/sessions", response_model=SessionsResponse)
def list_sessions() -> SessionsResponse:
    """列出所有活跃会话（按更新时间倒序）。"""
    items = []
    for sid, data in sessions.items():
        msgs = data["messages"]
        items.append(SessionInfo(
            session_id=sid,
            title=data.get("title", msgs[0]["content"][:40] if msgs else "空会话"),
            message_count=len(msgs),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        ))
    items.sort(key=lambda x: x.updated_at, reverse=True)
    return SessionsResponse(sessions=items)


# ============================================================
# 消息格式转换
# ============================================================


def _to_langchain_messages(session: list[dict]) -> list:
    """将内部 dict 格式转为 LangChain 消息对象列表。"""
    mapping = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
    }
    return [mapping.get(m["role"], HumanMessage)(content=m["content"]) for m in session]


def _from_langchain_messages(lc_messages: list) -> list[dict]:
    """将 LangChain 消息对象列表转为内部 dict 格式。"""
    role_map = {
        "human": "human",
        "ai": "ai",
        "system": "system",
        "tool": "tool",
    }
    result = []
    for m in lc_messages:
        role = role_map.get(m.type, m.type)
        content = m.content if isinstance(m.content, str) else str(m.content)
        # 跳过空的 tool 消息
        if role == "tool" and not content.strip():
            continue

        entry: dict = {"role": role, "content": content}

        # AI 消息附加工具调用信息
        if role == "ai" and hasattr(m, "tool_calls") and m.tool_calls:
            tool_names = []
            for tc in m.tool_calls:
                name = tc.get("name", "?") if isinstance(tc, dict) else tc.name if hasattr(tc, "name") else "?"
                tool_names.append(name)
            if tool_names:
                entry["tool_calls"] = tool_names

        result.append(entry)
    return result


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
