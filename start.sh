#!/usr/bin/env bash
# 启动 Deep Agents 后端 (FastAPI) + 前端 (Streamlit)
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo " Deep Agents 本地服务启动"
echo "=========================================="

# 启动后端
echo "[1/2] 启动后端 API (端口 8000)..."
uv run uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "  PID: $BACKEND_PID"

# 等待后端就绪
echo "  等待后端就绪..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "  ✅ 后端就绪"
        break
    fi
    sleep 1
done

# 启动前端
echo "[2/2] 启动 Streamlit 前端 (端口 8501)..."
STREAMLIT_SERVER_HEADLESS=true \
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
uv run streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.headless true &
FRONTEND_PID=$!
echo "  PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo " 服务已启动"
echo " 后端 API:  http://localhost:8000"
echo "  API 文档:  http://localhost:8000/docs"
echo "  Web 聊天:  http://localhost:8501"
echo "=========================================="
echo ""
echo "关闭服务请运行: ./stop.sh"
echo ""

# 记录 PID 以便 stop.sh 读取
echo "$BACKEND_PID" > /tmp/deepagents_backend.pid
echo "$FRONTEND_PID" > /tmp/deepagents_frontend.pid

# 等待子进程
wait
