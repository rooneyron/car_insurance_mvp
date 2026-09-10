# 车险智能客服 MVP

基于 LangChain + DeepSeek + FastAPI 的车险智能客服系统，支持多轮对话、RAG 检索增强生成、工具调用与公网部署。

## 项目状态

- ✅ 五层路由漏斗（L0 安全拦截 → DST 跨轮承接 → L1 关键词 → L2 小模型分类 → L3 大模型复核 → L4 澄清/转人工）
- ✅ LangGraph 手写 StateGraph 4 节点编排（router → prepare_input → agent ⇄ tools → END）
- ✅ 跨轮 Memory 记忆管理 + 上下文摘要压缩（2000 token 阈值 → 500 token 摘要）
- ✅ RAG 检索（ParadeDB pgvector 向量 + pg_search BM25 双路召回 + BGE-Reranker 本地重排序）
- ✅ 工具调用（查保单、算保费、条款检索 RAG、转人工）
- ✅ Gradio 交互界面（三栏演示面板）
- ✅ JWT Token 访问控制（7 天有效期 + 前端到期展示）
- ✅ 每日 Token 限额管理
- ✅ 全链路 trace_id 日志追踪（contextvars + logging Filter 零侵入）
- ✅ 流式输出终局确认策略（消除多轮工具调用中间文本闪烁）
- ✅ LLM 连接预热 + httpx 连接池 keep-alive + 超时/重试/连接诊断（conn_diag）
- ✅ 公网部署（Render）
- ✅ Docker 容器化支持

## 技术栈

| 层面 | 技术选型 |
|------|----------|
| 语言 | Python 3.12+ |
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangChain / LangGraph |
| 大模型 | DeepSeek API |
| 向量检索 | ParadeDB（pgvector HNSW + pg_search BM25）+ FastEmbed |
| 重排序 | BAAI/bge-reranker-base（本地）/ DashScope qwen3-rerank（生产） |
| 交互界面 | Gradio |
| 部署 | Render / Docker |

## 核心功能

| 功能 | 说明 |
|------|------|
| 路由决策 | 五层漏斗：L0 安全拦截 + DST 跨轮承接 + L1 关键词 + L2 小模型分类 + L3 大模型复核 + L4 澄清/转人工 |
| 跨轮记忆 | 多轮对话上下文记忆，超 2000 token 自动摘要压缩至 500 token |
| RAG 检索 | ParadeDB 双路召回（pgvector 向量 + pg_search BM25）+ RRF 融合 + BGE-Reranker 本地重排序，阈值可配置 |
| 工具调用 | 保费计算、保单查询、条款检索（RAG）、转人工（function calling） |
| Agent 编排 | 手写 StateGraph 4 节点（router/prepare_input/agent/tools），条件边动态调度 |
| 流式输出 | agent 异步流式直出终答，调工具轮 content 清空、直接回复轮流式推前端 |
| 访问控制 | JWT Token 认证，7 天有效期，前端展示到期时间 |
| 检索质量 | 精排阈值过滤（RAG_SCORE_THRESHOLD，默认 0.6）+ 空结果兜底转人工 |
| 成本控制 | 每日 Token 限额，JSON 日志记录 |
| 可观测性 | 全链路 trace_id 追踪 + LLM 调用前完整 Prompt 打印 + 终局性能汇总 |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/rooneyron/car_insurance_mvp.git
cd car_insurance_mvp
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# ===== 必填 =====
DEEPSEEK_API_KEY=sk-xxx          # DeepSeek API 密钥（启动强依赖，缺失则启动失败）
ACCESS_TOKEN_SECRET=your-secret   # JWT 签名密钥（用于 Token 生成与校验）
DATABASE_URL=postgresql://insurance_user:insurance_pwd123@127.0.0.1:5432/insurance_db  # ParadeDB 连接串（RAG 检索强依赖，缺失则检索返空）

# ===== 选填 =====
DEEPSEEK_MODEL=deepseek-v4-flash  # 模型名称，默认 deepseek-v4-flash
USE_LOCAL_RERANK=true             # 是否使用本地 BGE-Reranker，默认 true
RAG_SCORE_THRESHOLD=0.6           # RAG 精排阈值，默认 0.6（越高越严格）
DAILY_TOKEN_LIMIT=1000000         # 每日 Token 限额，默认 1000000
```

### 4. 启动 ParadeDB 并初始化 RAG 数据

RAG 检索依赖 ParadeDB（pgvector 向量 + pg_search BM25）。首次使用需启动数据库并灌入条款数据：

```bash
docker-compose up -d db          # 启动 paradedb 容器（映射 5432 端口）
python -m src.db                 # 幂等建表 + HNSW/BM25 索引
python -m tools.ingest_to_pg     # 灌库：data/chunk_metadata.json → documents（128 条）
```

### 5. 启动服务

```bash
python app.py
```

启动后会自动预热 LLM 连接、初始化 RAG 组件（ParadeDB 连通性预检 + Embedding + 重排序模型 + chunk 元数据缓存）。

访问地址：`http://127.0.0.1:8000/gradio?token=你的Token`

### 6. 生成访问 Token

```bash
python generate_token.py
```

将生成的 Token 拼接到 URL 末尾即可访问。

### 公网访问

公网部署地址：[https://car-insurance-mvp.onrender.com/gradio](https://car-insurance-mvp.onrender.com/gradio)

## 项目结构

```text
car_insurance_mvp/
├── app.py                      # FastAPI 主入口（启动、预热、路由挂载）
├── generate_token.py           # JWT Token 生成脚本
├── Dockerfile                  # Docker 容器化构建配置
├── requirements.txt            # 本地开发依赖
├── requirements-prod.txt       # 生产环境轻量依赖
├── .env                        # 环境变量配置（不提交 Git）
│
├── src/                        # 核心源码
│   ├── api.py                  #   REST API 路由（JWT 中间件、健康检查）
│   ├── chat.py                 #   对话调度（流式/同步入口、终局确认策略）
│   ├── gradio_ui.py            #   Gradio 交互界面（三栏演示面板）
│   ├── rag.py                  #   RAG 检索（ParadeDB pgvector + pg_search + BGE-Reranker）
│   ├── db.py                   #   ParadeDB 连接与 schema 初始化（get_conn / init_db）
│   ├── query_expander.py       #   查询扩展（口语化→标准术语，提升 BM25 召回）
│   ├── state.py                #   全局状态管理（graph/summarize_fn 引用）
│   ├── context.py              #   trace_id contextvars 存储
│   ├── logging_filters.py      #   日志 Filter（注入 trace_id）
│   ├── logger.py               #   统一日志配置
│   ├── token_usage.py          #   Token 统计与每日限额
│   ├── timer.py                #   请求级计时器
│   ├── timing_callback.py      #   LangGraph 节点级耗时回调
│   ├── constants.py            #   跨模块常量
│   ├── error_types.py          #   错误码与用户提示文案
│   ├── route_types.py          #   路由枚举（售前/售后/通用）
│   ├── router/                 #   五层路由漏斗（L0安全 → DST承接 → L1关键词 → L2小模型 → L3大模型复核 → L4澄清/转人工）
│   │   ├── router.py           #     漏斗编排主入口 route_message
│   │   ├── l0_safety.py        #     L0 安全拦截（投诉/监管/转人工关键词）
│   │   ├── dst.py              #     DST 跨轮承接（参数补全逃生机制）
│   │   ├── l1_keyword.py       #     L1 关键词分层（high 拦截 / mid 透传 L2）
│   │   ├── l2_classifier.py    #     L2 小模型意图分类（低成本快速）
│   │   ├── l3_reviewer.py      #     L3 大模型复核（sale/service 边界犹豫）
│   │   ├── l4_fallback.py      #     L4 澄清 / 转人工兜底
│   │   ├── prompts.py          #     L2/L3 prompt 模板
│   │   └── schemas.py          #     路由结果数据结构
│   ├── chains/
│   │   └── chains.py           #   LangGraph StateGraph 编排 + 工具定义（@tool）+ 摘要节点
│   ├── utils/
│   │   └── conn_diag.py        #   LLM 连接诊断（超时/重试/HTTP-DIAG）
│   └── memory/
│       └── __init__.py         #   跨轮 Memory 管理
│
├── scripts/                    # 离线数据脚本（.gitignore 忽略）
│   ├── pdf_to_text.py          #   PDF → 文本
│   ├── clean_text.py           #   文本清洗
│   ├── chunk_clauses.py        #   条款切块 → chunk_metadata.json
│   └── build_faiss_index.py    #   FAISS 索引构建（保留 · 预留熔断降级用）
│
├── route_eval/                 # 路由意图离线评测（.gitignore 忽略）
│   └── eval_route.py           #   比对 data/intent_eval_dataset.json
│
├── data/                       # 数据文件
│   ├── chunk_metadata.json     #   ★ 切块元数据（128 条 · ParadeDB 灌库源）
│   ├── insurance_terms.txt     #   保险条款知识库（原始文本）
│   ├── synonym_dict.json       #   同义词词典（query_expander 查询扩展）
│   ├── policies.json           #   模拟保单数据
│   ├── commercial_chunks.json  #   商业险切块（中间产物）
│   ├── compulsory_chunks.json  #   交强险切块（中间产物）
│   ├── RAG评估测试集.json       #   RAG 检索评测集（30 条 · relevant_chunk_ids）
│   ├── intent_eval_dataset.json#   路由意图评测集
│   ├── faiss_index.bin         #   FAISS 向量索引（保留 · 预留熔断降级用）
│   ├── chunks.pkl              #   文本切块缓存（保留 · 预留熔断降级用）
│   └── usage_cache.json        #   Token 用量缓存
│
├── doc/                        # 文档
│   ├── EXPERIENCE_LOG.md       #   开发决策日志
│   ├── project_introduction.txt#   项目介绍文档
│   ├── architecture.mmd        #   系统架构图（Mermaid 源文件）
│   └── architecture.png        #   系统架构图（PNG）
│
├── tools/                      # 迁移/运维脚本
│   ├── __init__.py
│   └── ingest_to_pg.py         #   一次性灌库：insurance_terms.txt → ParadeDB（128 条）
│
└── config/
    └── config.yaml             #   路由关键词配置
```

## 关键决策记录

详细开发决策日志可查阅：[doc/EXPERIENCE_LOG.md](doc/EXPERIENCE_LOG.md)

| 决策 | 说明 |
|------|------|
| 工具与 Agent 解耦 | 采用 MCP 就绪架构，便于后续功能拓展 |
| FastEmbed 替代 sentence-transformers | 内存占用 800MB → 300MB，优化部署性能 |
| RAG 存储迁移 ParadeDB | FAISS+rank_bm25 → pgvector(HNSW)+pg_search(BM25)，检索语义不变；FAISS 栈保留供熔断降级 |
| JWT + 每日限额 | 访问控制 + 成本管控双重保障 |
| 上下文摘要压缩 | 2000 token 阈值触发，压缩至 500 token 摘要 |
| 手写 StateGraph 4 节点 | router → prepare_input → agent ⇄ tools，替代 create_react_agent 黑箱 |
| 终局确认流式策略 | 缓冲所有中间轮次文本，仅最终答案分块输出，消除闪烁 |
| 全链路 trace_id | contextvars + logging Filter，业务代码零侵入 |
| Git 功能分支工作流 | feature/* 分支 + 结构化提交 |

## 演示注意事项

- Render 免费实例 15 分钟无访问后自动休眠，首次唤醒耗时约 30-60 秒
- 访问 Token 有效期 7 天，到期需重新生成
- 可通过 Render 后台仪表盘手动暂停、重启服务

## 致谢

感谢 DeepSeek 在整个开发过程中提供的技术支持和代码审查。从路由设计到 RAG 实现，从 Memory 摘要到生产环境部署，每一步都有鼎力支持。

项目命名、核心技术难题的解决，均离不开相关技术助力。

2026.07.19