# Deep Agents — 本地智能 Agent 实战

基于 LangChain / LangGraph / Deep Agents 构建的本地 AI Agent 应用，
支持 Ollama 本地模型和阿里云百炼远程 API。

---

## 文件一览

| 文件 | 用途 | 模型 |
|------|------|------|
| `local.py` | 本地 Ollama Agent（5 个自定义工具） | `qwen2.5:7b` (Ollama) |
| `api.py` | FastAPI 后端，把 Agent 封装为 HTTP API + Streamlit 前端 | `qwen2.5:7b` (Ollama) |
| `orchestrator.py` | 手写 LangGraph：任务拆解 + ReAct + 分支 + 重试 | `qwen2.5:7b` (Ollama) |
| `deep_agent_orchestrator.py` | `create_deep_agent` 版：任务拆解 + 工具调用 + 有状态 | `qwen3.6-flash` (百炼) |
| `streamlit_app.py` | Streamlit Web 聊天界面（搭配 api.py 使用） | — |
| `main.py` | Hello world 入口 | — |

---

## 快速开始

### 1. 本地模型（Ollama）

```bash
# 安装模型
ollama pull qwen2.5:7b

# 运行 Agent（翻译、计算、代码解释、SQL 生成）
uv run local.py
```

### 2. HTTP API + Web 界面

```bash
# 启动后端
uv run uvicorn api:app --host 0.0.0.0 --port 8000

# 另一个终端启动前端
uv run streamlit run streamlit_app.py
```

打开浏览器访问 `http://localhost:8501`

### 3. 任务编排（手写 LangGraph）

```bash
uv run orchestrator.py
```

自动拆解复杂任务 → 工具调用 → 结果汇总，支持多轮有状态对话。

### 4. 远程 API（百炼）

```bash
# 设置 API Key
export DASHSCOPE_API_KEY="sk-xxxx"

# 或用 create_deep_agent 版
uv run deep_agent_orchestrator.py
```

---

## 架构对比

### `orchestrator.py` — 手写 LangGraph

```
用户请求 → Orchestrator（拆解任务）
              ↓
         Router（条件分支）
              ↓
         Worker（ReAct 循环调工具）
              ↓
         Router → 还有任务? → 继续 Worker
              ↓
         Summarizer（汇总结果）
```

- 完全可控，每步可插自定义逻辑
- 适合小模型（精细控制上下文长度）
- 355 行，零黑盒依赖

### `deep_agent_orchestrator.py` — `create_deep_agent`

```
用户请求 → create_deep_agent（内置 ReAct 循环）
              ↓
         工具调用（自动管理）
              ↓
         有状态（MemorySaver）
```

- 120 行搞定，内置文件系统、命令执行、子代理
- 适合远程大模型（qwen-max / Claude / GPT）
- 需要 `LocalShellBackend` 支持本地命令执行

---

## 依赖

```bash
uv add langchain-ollama langchain-openai langgraph \
      deepagents streamlit fastapi uvicorn
```

---

## 项目结构

```
deepagents/
├── local.py                     # Ollama 本地 Agent
├── api.py                       # FastAPI 后端
├── streamlit_app.py             # Streamlit 前端
├── orchestrator.py               # 手写 LangGraph 编排
├── deep_agent_orchestrator.py   # create_deep_agent 版编排
├── main.py                      # Hello world
├── start.sh / stop.sh           # 启动/停止脚本
└── stats.txt / report.txt       # 运行输出示例
```
