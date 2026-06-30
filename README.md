# 车险智能客服 MVP

基于路由决策 + LangChain 的车险智能客服系统（MVP 版本）。

## 项目状态

当前完成：
- ✅ 路由决策模块（关键词配置化）
- ✅ 三级路由策略（关键词 → LLM 兜底 → 会话状态复用）
- ✅ 配置文件外部化（`config/config.yaml`）

开发中：
- ⏳ 三个 Chain 初始化（普通链 / 报价链 / 理赔链）
- ⏳ Memory 记忆管理
- ⏳ RAG 向量检索（FAISS）
- ⏳ 工具函数（查保单、算保费、转人工）
- ⏳ Gradio 交互界面
- ⏳ /health 健康检查接口

## 技术栈

- Python 3.10+
- LangChain / LangGraph
- DeepSeek API
- FAISS (向量检索)
- Gradio (交互界面)

## 本地运行（待完整实现后）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python app.py
