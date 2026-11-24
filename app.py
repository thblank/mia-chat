import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

# 1. 先加载 .env（必须在导入 mem0/torch 之前设置 HF_ENDPOINT）
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from starlette.responses import HTMLResponse
from openai import OpenAI
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from mem0 import Memory

# 2. 所有敏感配置只从环境变量读，代码里不出现任何明文
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chat_memories").strip()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
# 3. 启动即校验（fail fast），缺配置直接报错，不要带着空 Key 跑起来
if not DEEPSEEK_API_KEY:
    raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在项目根目录 .env 中配置")
if not AUTH_TOKEN:
    raise RuntimeError(
        "缺少 AUTH_TOKEN，请在 .env 中配置，可用 "
        'python -c "import secrets;print(secrets.token_urlsafe(32))" 生成'
    )

app = FastAPI(title="MiaChat Memory Service")


def verify_token(authorization: str | None = Header(default=None)) -> None:
    """校验请求头 Authorization: Bearer <AUTH_TOKEN>"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="缺少或格式错误的 Authorization 头，应为: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[len("Bearer "):].strip()
    # 用 compare_digest 做常数时间比较，避免时序攻击
    if not secrets.compare_digest(token.encode("utf-8"), AUTH_TOKEN.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Token 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )


config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": DEEPSEEK_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "openai_base_url": DEEPSEEK_BASE_URL,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-zh-v1.5",
            "embedding_dims": 512,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 512,
            "collection_name": QDRANT_COLLECTION,
            "path": "./my_qdrant_db",
        },
    },
}

print("正在初始化记忆系统...")
m = Memory.from_config(config)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    # user_id 由调用方（Java 端）传入，做基本格式约束，防止脏数据污染向量库
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.\-]+$",
        description="业务侧用户ID，例如 user_1001",
    )
    message: str = Field(..., min_length=1, max_length=4000)


# ================= 核心接口：聊天（需鉴权） =================
@app.post("/chat", dependencies=[Depends(verify_token)])
def chat_with_ai(req: ChatRequest, background_tasks: BackgroundTasks):
    results = m.search(req.message, filters={"user_id": req.user_id}, top_k=3)
    memories = [item["memory"] for item in results.get("results", [])]

    system_prompt = "你是一个聪明的AI助手。"
    if memories:
        system_prompt += "\n以下是你记住的关于该用户的信息：\n" + "\n".join(memories)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message},
        ],
        max_tokens=MAX_TOKENS,
    )
    reply = response.choices[0].message.content

    messages_to_save = [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": reply},
    ]
    background_tasks.add_task(m.add, messages_to_save, user_id=req.user_id)

    return {"reply": reply, "used_memories": memories, "user_id": req.user_id}


# ================= 健康检查（不鉴权，不放任何敏感信息） =================
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


# ================= 前端页面（页面本身不鉴权，因为要能被浏览器打开） =================
@app.get("/")
def get_index():
    with open(BASE_DIR / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)