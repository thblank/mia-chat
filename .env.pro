# D:\th\py_memory\mia-chat\.env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 接口鉴权令牌，务必和下面 AUTH_TOKEN 一致
# 生成方式：python -c "import secrets;print(secrets.token_urlsafe(32))"
AUTH_TOKEN=改成你随机生成的长字符串

# Qdrant 集合名（多用户共享一个集合，靠 user_id 元数据隔离）
QDRANT_COLLECTION=chat_memories

# HuggingFace 镜像（国内下载向量模型用）
HF_ENDPOINT=https://hf-mirror.com