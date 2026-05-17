#!/usr/bin/env bash
# 关闭 Deep Agents 前后端服务
set -e

echo "=========================================="
echo " 关闭 Deep Agents 服务"
echo "=========================================="

# 从 PID 文件关闭后端
if [ -f /tmp/deepagents_backend.pid ]; then
    PID=$(cat /tmp/deepagents_backend.pid)
    if kill "$PID" 2>/dev/null; then
        echo "  ✅ 后端 (PID $PID) 已关闭"
    else
        echo "  ⚠️  后端 (PID $PID) 不在运行"
    fi
    rm -f /tmp/deepagents_backend.pid
fi

# 从 PID 文件关闭前端
if [ -f /tmp/deepagents_frontend.pid ]; then
    PID=$(cat /tmp/deepagents_frontend.pid)
    if kill "$PID" 2>/dev/null; then
        echo "  ✅ 前端 (PID $PID) 已关闭"
    else
        echo "  ⚠️  前端 (PID $PID) 不在运行"
    fi
    rm -f /tmp/deepagents_frontend.pid
fi

# 按端口查找进程（兜底）
BACKEND_PID=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    echo "  ✅ 后端端口 8000 进程 (PID $BACKEND_PID) 已关闭"
fi

FRONTEND_PID=$(lsof -ti:8501 2>/dev/null || true)
if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    echo "  ✅ 前端端口 8501 进程 (PID $FRONTEND_PID) 已关闭"
fi

echo ""
echo "=========================================="
echo " 所有服务已关闭"
echo "=========================================="
