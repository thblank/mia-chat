import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from starlette.responses import HTMLResponse
from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from mem0 import Memory

# 1. 初始化 FastAPI 应用，相当于 @SpringBootApplication
app = FastAPI(title="MiaChat Memory Service")

# 2. 你的 Mem0 配置（完全复用之前的，多加了个 path 保存数据）
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "deepseek-flash",
            "api_key": "sk-e6e169a026e74ab586615a69ebf7ca18", # 你的 Key
            "openai_base_url": "https://api.deepseek.com"
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-zh-v1.5",
            "embedding_dims": 512
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 512,
            "collection_name": "xiaoming_memories",
            "path": "./my_qdrant_db"  # 数据存在当前目录下
        }
    }
}

print("正在初始化记忆系统...")
m = Memory.from_config(config)

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key="sk-e6e169a026e74ab586615a69ebf7ca18", # 🚨 和上面保持一致
    base_url="https://api.deepseek.com"
)
# 定义接收聊天请求的数据结构
class ChatRequest(BaseModel):
    user_id: str
    message: str


# ================= 核心接口：聊天 =================
@app.post("/chat")
def chat_with_ai(req: ChatRequest, background_tasks: BackgroundTasks):
    # 1. 去记忆库检索相关历史
    results = m.search(req.message, filters={"user_id": req.user_id}, top_k=3)
    memories = [item["memory"] for item in results.get("results", [])]

    # 2. 拼接 System Prompt，把记忆喂给大模型
    system_prompt = "你是一个聪明的AI助手。"
    if memories:
        system_prompt += "\n以下是你记住的关于该用户的信息：\n" + "\n".join(memories)

    # 3. 调用 DeepSeek 生成回复
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    )
    reply = response.choices[0].message.content

    # 4. 把这一轮对话作为新记忆存入（放到后台任务跑，不卡顿前端）
    messages_to_save = [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": reply}
    ]
    background_tasks.add_task(m.add, messages_to_save, user_id=req.user_id)

    return {"reply": reply, "used_memories": memories}


# ================= 前端页面 =================
@app.get("/")
def get_index():
    # 直接读取同目录下的 index.html 文件返回
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# 添加在文件最下面，注意前面不要有缩进
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)