# 车险智能客服 MVP

基于 LangChain \+ DeepSeek \+ FastAPI 的车险智能客服系统，支持多轮对话、RAG 检索、工具调用与公网部署。

## 项目状态

当前完成：

- ✅ 路由决策模块（关键词配置化）

- ✅ 三级路由策略（关键词 → 轻量级小模型意图识别 → 会话状态复用）

- ✅ 三个 Chain（普通链 / 报价链 / 理赔链）

- ✅ 跨轮 Memory 记忆管理 \+ 上下文摘要压缩

- ✅ RAG 向量检索（FAISS \+ FastEmbed）

- ✅ 工具函数（查保单、算保费、转人工）

- ✅ Gradio 交互界面

- ✅ /health 健康检查接口

- ✅ JWT Token 访问控制（7天有效期）

- ✅ 每日 Token 限额管理

- ✅ 公网部署（Render）

- ✅ LangGraph StateGraph 多 Agent 编排（route → insurance → general 三级图编排）

- ✅ RAGAS 质量评估（15 条测试集 + 4 大指标量化 RAG 链路质量）

- ✅ 全链路 trace_id 日志追踪（contextvars + logging Filter 自动注入）

- ✅ JWT 有效期前端展示（页面顶部显示 Token 到期时间）

- ✅ Git 版本控制（功能分支开发 + 结构化提交）

## 技术栈

- Python 3\.12\+

- FastAPI \+ Uvicorn

- LangChain / LangGraph

- DeepSeek API

- FAISS \+ FastEmbed（向量检索）

- Gradio（交互界面）

- Render（公网部署）

## 核心功能

|功能|说明|
|---|---|
|路由决策|关键词优先 \+ 会话状态复用 \+ 轻量级小模型意图识别|
|跨轮记忆|支持多轮对话上下文记忆，自动摘要压缩|
|RAG 检索|FAISS 向量检索 \+ FastEmbed 轻量级 Embedding|
|工具调用|保费计算、保单查询、转人工|
|访问控制|JWT Token 认证，7天有效期|
|Agent 编排|LangGraph StateGraph 构建多 Agent 协作图，条件路由动态调度|
|质量评估|RAGAS 4 大指标量化评估，faithfulness 0.98 / context_precision 0.97|
|成本控制|每日 Token 限额，JSON 日志记录|

## 快速开始

### 1\. 克隆项目

```bash
git clone https://github.com/rooneyron/car_insurance_mvp.git
cd car_insurance_mvp
```

### 2\. 安装依赖

```bash
pip install -r requirements.txt
```

### 3\. 配置环境变量

项目根目录创建 `.env` 环境变量文件，配置如下参数：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
ACCESS_TOKEN_SECRET=你的JWT密钥（用于Token生成）
USE_LOCAL_RERANK=true
```

### 4\. 启动服务

```bash
python app.py
```

本地访问地址：`http://127.0.0.1:8000/gradio?token=你的Token`

### 公网访问

公网部署地址：[https://car\-insurance\-mvp\.onrender\.com/gradio](https://car-insurance-mvp.onrender.com/gradio)

**Token 生成方式**：

```bash
python generate_token.py
```

执行脚本生成 Token 后，拼接至上述公网 URL 末尾即可正常访问。

## 项目结构

```text
car_insurance_mvp/
├── app.py                 # FastAPI 主入口
├── generate_token.py      # JWT Token 生成脚本
├── test_ragas.py          # RAGAS 评估脚本
├── requirements.txt       # 本地开发依赖
├── requirements-prod.txt  # 生产环境轻量依赖
├── src/
│   ├── core/
│   │   └── routing.py     # 路由决策模块
│   ├── chains/
│   │   └── chains.py      # LangGraph StateGraph 编排
│   ├── memory/
│   │   └── __init__.py    # 跨轮 Memory 管理
│   ├── tools/
│   │   └── __init__.py    # 工具函数（查保单/算保费/转人工）
│   ├── api.py             # REST API 路由
│   ├── chat.py            # 对话调度（LangGraph 入口）
│   ├── gradio_ui.py       # Gradio 界面
│   ├── rag.py             # RAG 检索（FAISS + BGE-Reranker）
│   ├── state.py           # LangGraph 状态定义
│   ├── logger.py          # 统一日志配置
│   ├── context.py         # trace_id contextvars 存储
│   ├── logging_filters.py # 日志 Filter（注入 trace_id）
│   ├── token_usage.py     # Token 统计与限额
│   ├── constants.py       # 跨模块常量
│   ├── error_types.py     # 错误码与用户提示
│   └── route_types.py     # 路由枚举
├── data/
│   ├── policies.json          # 模拟保单数据
│   ├── insurance_terms.txt    # 保险条款知识库
│   ├── faiss_index.bin        # FAISS 向量索引
│   ├── chunks.pkl             # 文本切块缓存
│   └── ragas_report.json      # RAGAS 评估基线报告
├── doc/
│   ├── architecture.mmd       # 系统架构图（Mermaid 源文件）
│   ├── architecture.png       # 系统架构图（PNG）
│   ├── EXPERIENCE_LOG.md      # 开发决策日志
│   └── TEST_LOG.md            # 测试报告
└── config/
    └── config.yaml        # 路由关键词配置
```

## 关键决策记录

详细开发决策日志可查阅：`docs/EXPERIENCE_LOG.md`

项目核心优化决策：

- 工具与 Agent 解耦，采用 MCP 就绪架构，便于后续功能拓展

- 生产环境使用 FastEmbed 替代 sentence\-transformers，内存占用从 800MB 降至 300MB，大幅优化部署性能

- 新增 JWT Token 访问控制 \+ 每日 Token 限额机制，保障服务安全、控制调用成本

- 接入 Memory 上下文摘要压缩模块，优化多轮对话上下文冗余问题

- 用 LangGraph StateGraph 替代手动路由，构建 route → insurance → general 多 Agent 编排图，实现条件分支动态调度

- 接入 RAGAS 评估框架，基于 15 条人工测试集量化 RAG 链路质量，faithfulness 0.98 / context_precision 0.97

- 实现全链路 trace_id 日志追踪，基于 contextvars + logging Filter，业务代码零侵入

- 采用 Git 功能分支工作流（feature/\*），结构化提交信息，便于代码审查与版本回溯

## 演示注意事项

- Render 免费实例 15 分钟无访问后自动休眠，再次访问唤醒耗时约 30\-60 秒

- 访问 Token 有效期为 7 天，到期需重新生成

- 可通过 Render 后台仪表盘手动暂停、重启服务

## 致谢

感谢 DeepSeek 在整个开发过程中提供的技术支持和代码审查。从路由设计到 RAG 实现，从 Memory 摘要到生产环境部署，每一步都有鼎力支持。

项目命名、核心技术难题的解决，均离不开相关技术助力。

2026\.07\.18