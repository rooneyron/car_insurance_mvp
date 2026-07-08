# 车险智能客服 MVP - 经验与决策日志

> 本文档记录开发过程中遇到的关键问题、权衡过程及最终决策，用于复盘和知识沉淀。

---

## 记录一：RAG 部署策略决策

### 背景

在 MVP 开发过程中，我们实现了一套完整的本地 RAG 系统，包含：
- 文本切割（RecursiveCharacterTextSplitter）
- 本地 Embedding（BAAI/bge-small-zh-v1.5，33MB）
- FAISS 向量索引持久化
- 本地 Cross-Encoder 精排（BAAI/bge-reranker-base，1.1GB）

本地开发测试运行良好，检索和精排效果均达到预期。但在规划公网部署时，发现 1.1GB 的 Rerank 模型会成为瓶颈。

### 遇到的问题

1. **资源限制**：Render、Railway 等平台的免费层内存通常在 512MB ~ 1GB 之间，无法加载 1.1GB 的 Rerank 模型。
2. **成本考量**：若继续使用本地 Rerank，必须升级到高内存实例（约 $25-50/月）或 GPU 实例（约 $100+/月），与 MVP“轻量验证”的定位不符。
3. **效果与成本的两难**：Rerank 确实提升了检索质量，但其带来的价值在当前阶段难以量化，是否值得为 MVP 付出额外成本？

### 决策过程

**问题重构**：将“如何部署 1.1GB 的 Rerank 模型？”重新定义为“如何在 MVP 阶段以最低成本验证 RAG+重排逻辑的价值？”

**评估维度**：
- 目标对齐：MVP 核心是验证产品逻辑，而非验证某个模型的线上性能。
- 资源约束：线上部署应尽可能利用免费资源。
- 风险控制：本地开发质量不能下降。
- 可扩展性：未来切换回高性能方案应零成本。

**备选方案**：
1. 放弃 Rerank，直接使用 FAISS 检索结果。
2. 使用商业 Rerank API（如 Cohere），但需额外注册和预存费用。
3. 使用 LLM（DeepSeek Chat）进行重排，复用已有 API。

### 最终决策

**采用策略模式，根据环境切换重排实现：**

| 环境 | 重排策略 | 原因 |
|------|---------|------|
| 本地开发 | Cross-Encoder (1.1GB 模型) | 保证调试精度，无成本压力 |
| 线上部署 | LLM (DeepSeek Chat) 重排 | 复用现有 API，零额外配置，轻量部署 |

**实现方式**：通过环境变量 `USE_LOCAL_RERANK` 控制。

**决策理由**：
1. 保留了本地开发质量。
2. 线上部署完全不受模型大小限制，可继续使用免费层。
3. 未来若要切回本地 Rerank 或接入商业 API，只需修改配置，无需重构。
4. MVP 阶段的核心目标是验证逻辑，而非比拼精度。

### 经验教训

1. **做 MVP 的关键能力是“取舍”**：不是所有好功能都要在这个阶段上线。把预算和精力花在验证核心假设上。
2. **环境隔离要提前考虑**：开发环境和生产环境的需求不同，从第一天起就为它们设计不同的实现路径。
3. **大模型不是唯一答案**：1.1GB 的模型效果虽好，但用 200 字的 Prompt + LLM 也能达到 80% 的效果。MVP 要的是“够用”，不是“完美”。

---

## 记录二：Python 模块导入问题

### 遇到的问题

在 `src/chains/chains.py` 中尝试 `from src.rag import search_terms` 时，始终报错 `ModuleNotFoundError: No module named 'src'`。

### 原因分析

执行 `python src/chains/chains.py` 时，Python 会将 `src/chains/` 作为工作目录的起点，而不是项目根目录。因此 `src` 不在 `sys.path` 中，无法被识别为模块。

### 解决方案

采用 `python -m src.chains.chains` 方式运行，Python 会自动将项目根目录加入 `sys.path`。

同时，在 `chains.py` 顶部加入路径修正：

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

记录三：DeepSeek Embedding API 不可用
遇到的问题
在实现 RAG 时，最初计划使用 DeepSeek 提供的 Embedding API（text-embedding-ada-002），但调用时返回 404 错误。

原因分析
DeepSeek 目前尚未开放公开的 Embedding API 服务。虽然其 Chat API 兼容 OpenAI 格式，但 Embedding 接口并不支持。

解决方案
转向纯本地 Embedding 方案，使用 BAAI/bge-small-zh-v1.5 模型（33MB），完全离线运行，不再依赖任何外部 API。

经验教训
在技术选型时，不能仅凭“兼容 OpenAI 格式”就假设所有 API 功能都可用。要仔细核对文档，确认所需的具体功能（特别是 Embedding 这类非核心接口）是否真正支持。

text

---

## 保存位置说明

请将上述文件命名为 `EXPERIENCE_LOG.md`，并保存到项目的 `docs/` 目录下，即路径为：
D:\car_insurance_mvp\docs\EXPERIENCE_LOG.md

text

如果你还没有 `docs/` 文件夹，可以先用以下命令创建：

```bash
mkdir docs

## 记录四：公网部署安全与成本控制策略

### 背景

MVP 部署到公网后，面临两个现实风险：
1. **API Key 被刷爆**：API 地址暴露后，任何人都可以调用，可能产生不可控的费用。
2. **无访问控制**：任何人都能访问服务，与“定向演示”的需求不符。

### 问题分析

- DeepSeek API 按 Token 计费，价格便宜但不等于免费。如果被恶意刷量，单日费用仍可能达到数十元。
- MVP 的预期使用量极低（仅用于演示和测试），正常情况下日均费用不超过 0.1 元。但缺乏防护等于把风险敞口打开。

### 解决方案

采用 **Token 认证 + 用量限额** 双层防护：

---

#### 第一层：访问 Token（7天有效期）

**目标**：只有持有有效 Token 的用户才能访问服务。

**实现方式**（JWT + 环境变量）：

1. **生成 Token 的接口**（仅管理员可用）：
```javascript
// 部署在服务端，如 /admin/generate-token
app.get('/admin/generate-token', (req, res) => {
  const adminKey = req.query.adminKey;
  
  // 简单校验，防止他人恶意生成
  if (adminKey !== process.env.ADMIN_SECRET) {
    return res.status(403).json({ error: '无权限' });
  }
  
  const token = jwt.sign(
    { purpose: 'car-insurance-demo' },
    process.env.ACCESS_TOKEN_SECRET,
    { expiresIn: '7d' }
  );
  
  res.json({ 
    link: `https://your-demo-domain.com?token=${token}` 
  });
});


## 优化记录：响应耗时分析与显示

### 1. 前端增加耗时显示
在回复前展示总耗时，如 `💰 报价链 ⏱️ 1.4s`。

### 2. 后台增加计时日志
新增 Timer 工具，记录各阶段耗时，输出示例：
总耗时: 1395ms
  ├── 初始化 Chain: 5ms
  ├── 初始化 RAG: 0ms
  ├── 路由决策: 0ms
  ├── Agent 调用: 1395ms

### 3. 耗时原因分析
- 纯 LLM 调用基准耗时：约 0.9s（DeepSeek API）
- Agent 完整调用耗时：约 1.4s
- 本地 Agent 框架 + 工具执行：约 0.5s

### 4. 预加载
应用启动时加载模型，首条消息无需等待模型加载。

### 5. 结论
一次对话约 2 次 LLM 调用，单次约 0.7s，合计 1.4s，耗时合理，无紧急优化必要。



## 用户体验优化：进度感知与状态提示

### 背景
当前系统响应时间约 6.5 秒（含 LLM 调用 + RAG 检索），用户提交消息后界面无任何反馈，容易产生“系统卡死”的错觉，影响演示体验。

### 优化目标
在不改变实际耗时的情况下，通过进度提示让用户感知到系统正在工作，降低等待焦虑。

### 实现方式
- 在 Gradio 界面中，用户提交消息后立即显示状态提示：
  - `"正在理解问题..."`（路由 + 第一次 LLM 调用）
  - `"🔍 正在查询保险条款..."`（RAG 工具执行）
  - `"正在生成回答..."`（第二次 LLM 调用生成回复）
- 最终回复内容采用流式输出，逐字展示，增强实时感。

### 实现方式（技术选型）
- Gradio 的 `gr.ChatInterface` 配合生成器函数，通过 `yield` 分步更新界面。
- 在 `chat_api` 中拆分执行阶段，每进入一个阶段就 `yield` 一条状态消息。

### 预期效果
用户看到状态提示后，能清晰感知系统正在按步骤处理，等待焦虑显著降低。演示时能直观展示系统的“工作过程”，而非面对一个黑盒。



## 2026-07-09 功能模块优化

### 1. Memory 摘要（上下文压缩）

**背景**：多轮对话后上下文持续膨胀，超过模型窗口限制会导致请求失败或成本上升。

**实现方式**：集成 `langmem.short_term.SummarizationNode`，在每次调用模型前通过 `pre_model_hook` 执行摘要检查。

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `max_tokens` | 2000 | 触发摘要的阈值，上下文超过此值生成摘要 |
| `max_summary_tokens` | 500 | 摘要内容的 token 上限 |
| `input_messages_key` | "messages" | 读取消息的状态键 |
| `output_messages_key` | "messages" | 写入摘要的状态键 |

**验证方式**：设置 `max_tokens=100` 测试，发送多轮对话后观察消息数从 5 降到 3，摘要功能生效。

**涉及文件**：`src/chains/chains.py`


### 2. Token 访问控制（JWT 链接认证）

**背景**：部署公网后，需要控制谁能访问演示服务，避免被恶意调用。

**实现方式**：
- 使用 JWT 生成访问 Token，有效期 7 天
- 在 FastAPI 中间件中验证所有请求（除 `/health`、`/`、静态资源外）
- 通过 `generate_token.py` 脚本在本地生成 Token

**验证方式**：
- 不带 Token 访问 `/gradio` → 返回 401 `Missing token`
- 带有效 Token → 正常加载 Gradio 界面

**涉及文件**：
- `app.py`（中间件）
- `generate_token.py`（新建）


### 3. Token 消耗统计（每日限额 + JSON 日志）

**背景**：需要控制每日 API 调用成本，避免超出预算；同时记录每次请求的 Token 消耗用于分析。

**实现方式**：
- 在 `src/token_usage.py` 中管理每日累计 Token 消耗
- 默认限额：`1,000,000 token/天`，0 点自动重置
- 数据持久化：`data/usage_cache.json`
- 每次请求从 `usage_metadata` 提取 Token 数并累加
- 打印 JSON 格式日志到终端
- 新增 `/queryToken` 接口查询当日累计用量

| 字段 | 说明 |
|------|------|
| `input_tokens` | 本次输入 Token 数 |
| `output_tokens` | 本次输出 Token 数 |
| `cached_tokens` | 缓存命中 Token 数 |
| `total_tokens` | 本次总 Token 数 |
| `daily_usage` | 当日累计用量 |

**验证方式**：
- 设置 `DAILY_TOKEN_LIMIT=10` 测试，第二条消息返回“今日 Token 配额已用完”
- 终端输出 `[Token日志]` JSON
- `/queryToken` 返回累计数据

**涉及文件**：
- `src/token_usage.py`（新建）
- `app.py`（集成统计和限额检查）
- `data/usage_cache.json`（自动生成）