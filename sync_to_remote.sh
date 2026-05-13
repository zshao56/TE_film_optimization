#!/bin/bash
# 同步新文件到远程服务器的脚本

# 使用方法：
# 1. 修改下面的远程服务器信息
# 2. chmod +x sync_to_remote.sh
# 3. ./sync_to_remote.sh

# ===== 配置区域 =====
REMOTE_USER="your_username"           # 你的远程用户名
REMOTE_HOST="your_server_ip"          # 远程服务器IP或域名
REMOTE_PATH="/path/to/TE_film"        # 远程项目路径
# ===================

# 要同步的文件
FILES=(
    "analysis_best_structures.py"
    "IMPROVEMENT_PLAN.md"
)

echo "开始同步文件到远程服务器..."

for file in "${FILES[@]}"; do
    echo "正在上传: $file"
    scp "$file" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"
    if [ $? -eq 0 ]; then
        echo "✓ $file 上传成功"
    else
        echo "✗ $file 上传失败"
    fi
done

echo ""
echo "同步完成！现在可以在远程服务器上运行："
echo "  cd ${REMOTE_PATH}"
echo "  python analysis_best_structures.py"
