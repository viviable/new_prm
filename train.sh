#!/bin/bash

# 启用错误检查和命令回显
set -e

# 检查必要的环境变量
: "${MASTER_ADDR:?Error: MASTER_ADDR environment variable is not set}"
: "${NODE_RANK:?Error: NODE_RANK environment variable is not set}"

# 设置环境
source /home/i-chengjie/venv/verl/bin/activate
cd /home/i-chengjie/PURE
sudo cp nccl.conf /etc/nccl.conf

# 配置变量
ip=$MASTER_ADDR
port=6379
ip_head=$ip:$port
node_rank=$NODE_RANK

# 定义日志函数
log() {
    echo "[RANK $node_rank $(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# 清理函数
cleanup() {
    log "正在清理进程..."
    ray stop
    exit 0
}

# 注册清理函数
trap cleanup SIGTERM SIGINT

# make sure we set environment variables before Ray initialization
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=1

# 主节点启动
if [ "$node_rank" == "0" ]; then
    log "启动主节点 ray 服务..."
    ray start --head --node-ip-address=$ip --port=$port
    if [ $? -ne 0 ]; then
        log "Error: Ray 主节点启动失败"
        exit 1
    fi
fi

log "等待主节点初始化..."
sleep 120

# 工作节点启动
if [ "$node_rank" != "0" ]; then
    log "启动工作节点..."
    ray start --address "$ip_head"
    if [ $? -ne 0 ]; then
        log "Error: Ray 工作节点启动失败"
        exit 1
    fi
fi

log "等待集群就绪..."
sleep 120

# 运行训练任务
if [ "$node_rank" == "0" ]; then
    ray status
    log "开始训练任务..."
    PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo
else
    log "工作节点等待任务..."
    # 使用 wait 替代 sleep
    while true; do
        sleep 120 &
        wait $!
        # log "debug: check gpu usage"
        # nvidia-smi
    done
fi
