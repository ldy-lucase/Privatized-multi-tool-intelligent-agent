"""Streamlit 前端：Deep Agents Web 聊天界面 — 带停止思考按钮。"""

import time
import uuid

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Deep Agents 聊天", page_icon="🤖", layout="wide")

# ============================================================
# 辅助函数
# ============================================================

def api_get(path: str, timeout: int = 10) -> dict | None:
    try:
        resp = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def api_delete(path: str) -> bool:
    try:
        resp = httpx.delete(f"{API_BASE}{path}", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def format_time(ts: float) -> str:
    if ts <= 0:
        return ""
    t = time.localtime(ts)
    now = time.localtime()
    if t.tm_yday == now.tm_yday and t.tm_year == now.tm_year:
        return time.strftime("%H:%M", t)
    elif t.tm_year == now.tm_year:
        return time.strftime("%m-%d %H:%M", t)
    else:
        return time.strftime("%Y-%m-%d %H:%M", t)


def refresh_sessions() -> None:
    data = api_get("/sessions")
    st.session_state.session_list = data["sessions"] if data else []


def load_messages(sid: str) -> list:
    data = api_get(f"/history/{sid}")
    return data["messages"] if data else []


def delete_session(sid: str) -> None:
    if api_delete(f"/history/{sid}"):
        if st.session_state.current_session == sid:
            st.session_state.current_session = None
            st.session_state.messages_loaded = False
        refresh_sessions()
        st.rerun()


# ============================================================
# Session State 初始化
# ============================================================

for key, default in [
    ("session_list", []),
    ("current_session", None),
    ("messages", []),
    ("messages_loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.markdown("## 🤖 Deep Agents")
    st.caption("本地大模型 · Web 聊天")

    if st.button("🆕 新建会话", use_container_width=True, type="primary"):
        st.session_state.current_session = None
        st.session_state.messages = []
        st.session_state.messages_loaded = False
        st.rerun()

    st.divider()
    refresh_sessions()

    if st.session_state.session_list:
        st.markdown("**历史会话**")
        for s in st.session_state.session_list:
            sid = s["session_id"]
            title = s["title"] or "新会话"
            count = s["message_count"]
            updated = format_time(s["updated_at"])
            is_active = sid == st.session_state.current_session

            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if is_active:
                    st.markdown(
                        f"<div style='background:#1a1a2e;border-radius:8px;padding:6px 10px;"
                        f"border-left:3px solid #4FC3F7;margin-bottom:4px;'>"
                        f"<b>{title}</b><br>"
                        f"<small style='color:#888;'>{count}条 · {updated}</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(title, key=f"sid_{sid}", use_container_width=True):
                        st.session_state.current_session = sid
                        st.session_state.messages_loaded = False
                        st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{sid}"):
                    delete_session(sid)
                    st.rerun()
    else:
        st.info("暂无会话记录")

    st.divider()
    st.caption("Powered by Ollama + Deep Agents")

# ============================================================
# 主区域
# ============================================================

st.title("💬 Deep Agents 本地聊天")

# 加载当前会话消息
if st.session_state.current_session and not st.session_state.messages_loaded:
    msgs = load_messages(st.session_state.current_session)
    st.session_state.messages = msgs
    st.session_state.messages_loaded = True

# 显示消息
for msg in st.session_state.messages:
    if msg["role"] == "human":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "ai":
        with st.chat_message("assistant"):
            # 如果有工具调用，先显示调用标签
            if msg.get("tool_calls"):
                tools = ", ".join(msg["tool_calls"])
                st.markdown(
                    f"<div style='font-size:0.8em;color:#FFA726;margin-bottom:4px;'>"
                    f"🔧 调用了工具: {tools}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
    elif msg["role"] == "tool":
        with st.chat_message("tool", avatar="🛠"):
            st.code(msg["content"][:500], language="text")

# ============================================================
# 输入 + 发送（同步请求，带 spinner）
# ============================================================

prompt = st.chat_input("输入你的问题…")

if prompt:
    # 新会话自动生成 ID
    if not st.session_state.current_session:
        st.session_state.current_session = str(uuid.uuid4())

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt})

    # 同步请求（带 spinner，UI 自动展示等待状态）
    with st.chat_message("assistant"):
        with st.spinner("⏳ 思考中…（本地模型较慢，请稍候）"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/chat",
                    json={
                        "session_id": st.session_state.current_session,
                        "message": prompt,
                    },
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    new_sid = data["session_id"]
                    reply = data["response"]
                    st.session_state.current_session = new_sid
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "ai", "content": reply})
                else:
                    err = f"❌ 后端返回错误: HTTP {resp.status_code}"
                    st.error(err)
                    st.session_state.messages.append({"role": "ai", "content": err})
            except httpx.TimeoutException:
                err = "⏰ 请求超时，模型响应太慢。请重试或换一个更简单的问法。"
                st.error(err)
                st.session_state.messages.append({"role": "ai", "content": err})
            except httpx.ConnectError:
                err = "❌ 无法连接到后端，请确认 `uvicorn` 是否在运行。"
                st.error(err)
                st.session_state.messages.append({"role": "ai", "content": err})
            except Exception as e:
                err = f"❌ 请求失败: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "ai", "content": err})

    # 刷新会话列表
    refresh_sessions()
    st.rerun()
