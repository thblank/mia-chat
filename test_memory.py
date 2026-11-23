import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 1. 导入库：相当于 Java 里的 import
# from 模块名 import 类名
from mem0 import Memory

# 2. 字典（Dictionary）：Python 中最核心的数据结构
# 它相当于 Java 里的 Map 或者写 JSON。
# 格式是：{"键": "值"}。Python 不像 Java 需要提前声明 Map<String, Object> 变量类型，直接用大括号就行。
config = {
    "llm": {  # 大括号嵌套，相当于 Map 里又装了一个 Map
        "provider": "openai",   # Mem0 底层默认用 OpenAI 的规范，DeepSeek 完全兼容这种规范
        "config": {
            "model": "deepseek-flash",  # 指定模型名称
            "api_key": "sk-e6e169a026e74ab586615a69ebf7ca18", # 替换成你的真实 Key
            "openai_base_url": "https://api.deepseek.com" # 告诉它把请求发给 DeepSeek 的服务器，而不是 OpenAI
        }
    },
    "embedder": {
        "provider": "huggingface", # 文本向量化（Embedding）由 HuggingFace 库提供
        "config": {
            # 这是一个极其轻量级的中文向量模型
            # 类似于给这段话提取一个“指纹”，后续检索全靠它
            "model": "BAAI/bge-small-zh-v1.5",
            "embedding_dims": 512  # 关键：明确声明维度为512
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 512 , # 关键：告诉Qdrant用512维来创建集合
            "collection_name": "xiaoming_memories", # 指定集合名称
            "path": "./my_qdrant_db"  # 指定本地存储路径
        }
    }
}

# 打印一句话，相当于 Java 的 System.out.println()
print("正在初始化记忆系统，首次运行需要下载本地向量模型，请稍候...")

# 3. 实例化对象：相当于 Java 的 new Memory(config)
# 这里用了一个设计模式：通过工厂方法 from_config 来创建对象
m = Memory.from_config(config)

# 4. 列表（List）嵌套字典：
# 这是一个 Python 列表，用中括号 [] 表示，相当于 Java 的 List<Map<String, String>>。
# 这里构造的是标准的对话格式，role 只能是 "user"（用户）或 "assistant"（AI）
messages = [
    {"role": "user", "content": "你好，我叫小明，是个Java开发。我最近在学Python，但我特别讨厌吃香菜。"},
    {"role": "assistant", "content": "你好小明，很高兴认识你！Java转Python挺容易的。我记住你不吃香菜了。"}
]

print("\n📥 正在写入记忆...")
# 5. 调用对象的方法：
# 相当于 Java 里的 m.add(messages, "xiaoming")
# 注意：Python 没有分号结尾，代码块也不用大括号，而是用**缩进（4个空格）**来表示层级！
m.add(messages, user_id="xiaoming")

# 6. 测试检索
print("\n🔍 测试检索 1：他的职业是什么？")
# 调用 search 方法，返回的结果是一个字典，里面包含了 "results" 这个列表
res1 = m.search("这个人的工作是什么？", filters={"user_id": "xiaoming"})

# 7. for 循环遍历：
# 相当于 Java 的 for (Map r : res1.get("results")) { ... }
# Python 里直接写 for 变量名 in 列表:
for r in res1["results"]:
    # 这里的缩进表示循环体（没有 Java 的大括号）
    # r 是一个字典，通过 r['memory'] 获取记忆的内容
    print(f"  -> 找到记忆：{r['memory']}")

print("\n🔍 测试检索 2：饮食有什么禁忌？")
res2 = m.search("他吃饭有什么忌口？", filters={"user_id": "xiaoming"})
for r in res2["results"]:
    print(f"  -> 找到记忆：{r['memory']}")