# 车险智能客服 MVP

基于 LangChain + DeepSeek + FastAPI 的车险智能客服系统，支持多轮对话、RAG 检索增强生成、工具调用与公网部署。

## 项目状态

- ✅ 五层路由漏斗（L0 安全拦截 → DST 跨轮承接 → L1 关键词 → L2 小模型分类 → L3 大模型复核 → L4 澄清/转人工）
- ✅ LangGraph 手写 StateGraph 4 节点编排（router → prepare_input → agent ⇄ tools → END）
- ✅ 跨轮 Memory 记忆管理 + 上下文摘要压缩（2000 token 阈值 → 500 token 摘要）
- ✅ 长期记忆双存储（sqlite 结构化画像 5 字段 + PG 语义向量事实短句，对话后异步提取、去重、数量控制）
- ✅ RAG 检索（ParadeDB pgvector 向量 + pg_search BM25 双路召回 + BGE-Reranker 本地重排序）
- ✅ 工具调用（查保单、算保费、条款检索 RAG、转人工）
- ✅ Gradio 交互界面（三栏演示面板）
- ✅ 飞书机器人 in-process 直连（长连接事件 → chat_api → interactive 卡片回复，异步 handler 秒 ack 防重投）
- ✅ 飞书语音消息处理（语音下载 → 火山引擎 ASR 转文字 → chat_api → 卡片回复，支持 opus/wav 格式）
- ✅ JWT Token 访问控制（7 天有效期 + 前端到期展示）
- ✅ 每日 Token 限额管理
- ✅ 全链路 trace_id 日志追踪（contextvars + logging Filter 零侵入）
- ✅ 流式输出终局确认策略（消除多轮工具调用中间文本闪烁）
- ✅ LLM 连接预热 + httpx 连接池 keep-alive + 超时/重试/连接诊断（conn_diag）
- ✅ 公网部署（Render）
- ✅ Docker 容器化支持（双场景：本地 GPU + 阿里云无 GPU）
- ✅ 阿里云生产部署（Docker + ParadeDB + DashScope API Rerank）

## 技术栈

| 层面 | 技术选型 |
|------|----------|
| 语言 | Python 3.12+ |
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangChain / LangGraph |
| 大模型 | DeepSeek API |
| 向量检索 | ParadeDB（pgvector HNSW + pg_search BM25）+ FastEmbed |
| 重排序 | BAAI/bge-reranker-base（本地）/ DashScope qwen3-rerank（生产） |
| 长期记忆 | sqlite（结构化画像）+ ParadeDB（语义向量）+ LLM 异步提取 |
| 交互界面 | Gradio + 飞书机器人（lark-oapi） |
| 语音识别 | 火山引擎豆包语音 ASR（录音文件识别极速版，ogg_opus/wav） |
| 部署 | Render / Docker / 阿里云 ECS |

## 核心功能

| 功能 | 说明 |
|------|------|
| 路由决策 | 五层漏斗：L0 安全拦截 + DST 跨轮承接 + L1 关键词 + L2 小模型分类 + L3 大模型复核 + L4 澄清/转人工 |
| 跨轮记忆 | 多轮对话上下文记忆，超 2000 token 自动摘要压缩至 500 token |
| 长期记忆 | 对话后异步提取用户画像（姓名/年龄/证件/手机/车牌）+ 语义事实短句，双存储（sqlite 结构化 + PG 向量），敏感信息掩码注入 LLM |
| RAG 检索 | ParadeDB 双路召回（pgvector 向量 + pg_search BM25）+ RRF 融合 + BGE-Reranker 本地重排序，阈值可配置 |
| 工具调用 | 保费计算、保单查询、条款检索（RAG）、转人工（function calling） |
| Agent 编排 | 手写 StateGraph 4 节点（router/prepare_input/agent/tools），条件边动态调度 |
| 流式输出 | agent 异步流式直出终答，调工具轮 content 清空、直接回复轮流式推前端 |
| 飞书机器人 | in-process 直连 chat_api，长连接事件 → interactive 卡片回复，异步 handler 秒 ack 防重投，reply 网络异常退避重试 |
| 飞书语音消息 | 语音下载 → 火山引擎 ASR 转文字 → chat_api → 卡片回复（顶部显示识别结果），opus 自动 fallback 转 wav（ffmpeg），专用事件循环避免 asyncio.run 连接池冲突 |
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
FEISHU_APP_ID=cli_xxx             # 飞书应用 App ID（缺失则跳过飞书机器人）
FEISHU_APP_SECRET=xxx             # 飞书应用 App Secret（缺失则跳过飞书机器人）
VOLC_ASR_API_KEY=xxx              # 火山引擎 ASR API Key（飞书语音转文字，缺失则语音功能不可用）
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
├── Dockerfile                  # Docker 容器化构建配置（含 ffmpeg + 阿里云 apt 镜像）
├── docker-compose.yml          # 本地开发 Docker 配置（GPU + 本地 Rerank）
├── docker-compose.prod.yml     # 阿里云生产 Docker 配置（无 GPU + API Rerank）
├── requirements.txt            # 本地开发依赖
├── requirements-prod.txt       # 生产环境轻量依赖
├── .env                        # 环境变量配置（不提交 Git）
│
├── src/                        # 核心源码
│   ├── api.py                  #   REST API 路由（JWT 中间件、健康检查）
│   ├── chat.py                 #   对话调度（流式/同步入口、终局确认策略）
│   ├── gradio_ui.py            #   Gradio 交互界面（三栏演示面板）
│   ├── asr.py                  #   火山引擎 ASR 语音识别（录音文件识别极速版，speech_to_text）
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
│   ├── feishu/                 #   飞书机器人模块
│   │   ├── feishu_bot.py       #     飞书机器人（in-process 直连 chat_api，长连接事件 → interactive 卡片回复，语音消息 ASR）
│   │   └── feishu_ws_echo.py   #     飞书 WebSocket 调试工具
│   ├── utils/
│   │   └── conn_diag.py        #   LLM 连接诊断（超时/重试/HTTP-DIAG）
│   └── memory/                 #   长期记忆模块（结构化 + 语义双存储 + 对话后异步提取）
│       ├── __init__.py         #     模块入口（单例 + spawn_memory_extraction 异步提取触发）
│       ├── structured_store.py #     sqlite 结构化画像（姓名/年龄/证件/手机/车牌 5 字段）
│       ├── semantic_store.py   #     PG 语义向量（用户持久事实短句 + embedding 去重）
│       ├── extractor.py        #     LLM 提取器（从对话提取结构化 + 语义记忆）
│       ├── reader.py           #     记忆读取（load_user_memory + 掩码 prompt 注入）
│       ├── mask.py             #     敏感信息掩码（身份证/手机/车牌遮罩）
│       └── dedup_store.py      #     飞书消息去重（只读判重 + reply 成功后登记）
│
├── tests/                      # 测试脚本
│   ├── test_demo_panel.py      #   自动化演示测试（三栏 10 按钮端到端验证）
│   ├── test_mask.py            #   敏感信息掩码单元测试
│   └── test_reader.py          #   记忆读取单元测试
│
├── eval/                       # 评估脚本
│   └── eval_rag_retrieval.py   #   RAG 检索质量评估
│
├── route_eval/                 # 路由意图离线评测
│   └── eval_route.py           #   比对 data/intent_eval_dataset.json
│
├── scripts/                    # 离线数据脚本
│   ├── pdf_to_text.py          #   PDF → 文本
│   ├── clean_text.py           #   文本清洗
│   ├── chunk_clauses.py        #   条款切块 → chunk_metadata.json
│   └── build_faiss_index.py    #   FAISS 索引构建（保留 · 预留熔断降级用）
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
│   ├── user_memory.db          #   长期记忆 + 飞书去重 sqlite（结构化画像 + processed_messages）
│   └── usage_cache.json        #   Token 用量缓存
│
├── doc/                        # 文档
│   ├── EXPERIENCE_LOG.md       #   开发决策日志
│   ├── project_introduction.txt#   项目介绍文档
│   ├── architecture.mmd        #   系统架构图（Mermaid 源文件）
│   └── architecture.png        #   系统架构图（PNG）
│
└── tools/                      # 迁移/运维脚本
    ├── __init__.py
    └── ingest_to_pg.py         #   一次性灌库：data/chunk_metadata.json → ParadeDB（128 条）
```

## 关键决策记录

详细开发决策日志可查阅：[doc/EXPERIENCE_LOG.md](doc/EXPERIENCE_LOG.md)

| 决策 | 说明 |
|------|------|
| 工具与 Agent 解耦 | 采用 MCP 就绪架构，便于后续功能拓展 |
| FastEmbed 替代 sentence-transformers | 内存占用 800MB → 300MB，优化部署性能 |
| RAG 存储迁移 ParadeDB | FAISS+rank_bm25 → pgvector(HNSW)+pg_search(BM25)，检索语义不变；FAISS 栈保留供熔断降级 |
| 长期记忆双存储 | sqlite 结构化画像（5 字段，本地快查）+ PG 语义向量（事实短句，embedding 去重，相似度 0.85 阈值）；对话后 LLM 异步提取，绝不阻塞主流程 |
| 飞书 in-process 直连 | feishu_bot.py 与 Gradio 同源 from src.chat import chat_api，不走 HTTP/JWT，复用已加载 graph/RAG/记忆；一条 python app.py 同起 API+Gradio+飞书 |
| 飞书长期事件循环 | 飞书专用 asyncio 事件循环（start_feishu_loop），所有消息通过 run_coroutine_threadsafe 提交，避免每次 asyncio.run 创建/销毁循环导致 httpx 连接池绑定错误 |
| 飞书异步 handler | lark SDK 同步调 handler 且 handler 返回后才回写 ack；handler 拆成快速提取+只读判重+submit 线程池秒返回（实测 0ms），chat_api+reply+登记异步跑，避免 ack 超时致飞书重投 |
| 去重判重/登记分离 | is_duplicate_message 只读 SELECT 不登记；mark_message_processed 仅在 reply 成功后 INSERT OR IGNORE，避免回复失败的消息被误标记、重投跳过致用户永久收不到回复 |
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

2026.09.20