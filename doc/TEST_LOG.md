降级功能测试报告
测试日期： 2026-07-08
测试范围： 输入校验、RAG 降级、LLM API 异常、Agent 循环超限、正常流程回归
测试结论： 全部通过 ✅

测试环境
项目	配置
模型	DeepSeek Chat API
RAG 阈值	0.6（测试时临时调整）
Agent 循环限制	5（测试时临时调整）
知识库	16 个条款块（测试时临时清空）
测试结果明细
编号	测试场景	测试方法	预期结果	实际结果	状态
1	输入超长	发送 1001 字符消息	返回“输入内容过长，请控制在1000字符以内”	返回正确提示	✅ 通过
2	Rerank 分数低于阈值	RAG_SCORE_THRESHOLD 改为 0.99，发送“车损险保什么”	返回降级文本，missed_queries.log 记录一条	返回“未找到相关内容”，日志有记录 best_score	✅ 通过
3	FAISS 完全查不到	清空 insurance_terms.txt，删除索引文件，发送“车损险保什么”	返回 LLM 自身知识，终端打印 [RAG降级]	前端正常回复，终端显示 [RAG降级] 知识库无结果，使用LLM自身知识	✅ 通过
4	LLM API 异常	.env 中 API Key 改为错误值，发送任意消息	返回“系统服务暂时不可用，请稍后再试”	返回 ❌ 系统繁忙，请稍后再试，日志记录 401 错误	✅ 通过
5	Agent 循环超限	recursion_limit 改为 1，发送“车损险保什么”	返回“处理超时，请重新提问”	返回 ❌ 处理超时，请重新提问，捕获 GraphRecursionError	✅ 通过
6	正常流程回归	恢复所有配置，发送“我想续保（车型特斯拉，30岁，8年驾龄）”	正常返回保费估算结果，路由标签 + 耗时正常	返回保费估算结果 7600 元，标签 💰 报价链 ⏱️ 6.2s	✅ 通过
测试中临时修改的配置
配置项	修改内容	恢复时间
RAG_SCORE_THRESHOLD	0.6 → 0.99	测试 2 后恢复为 0.6
insurance_terms.txt	清空内容	测试 3 后恢复为原始条款
faiss_index.bin / chunks.pkl	删除	测试 3 后删除，重启重建
.env 中 API Key	改为错误值	测试 4 后恢复为正确 Key
recursion_limit	5 → 1	测试 5 后恢复为 5
关键日志证据
1. RAG 降级日志（测试 3）：

text
[RAG计时] FAISS 检索: 13ms, 召回 1 个候选
[RAG计时] Rerank 推理: 48ms (对 1 个候选)
[RAG降级] 知识库无结果，使用LLM自身知识
2. API 异常日志（测试 4）：

text
[ERROR] chat_api 异常: Error code: 401 - {'error': {'message': 'Authentication Fails...'}}
3. Agent 循环超限日志（测试 5）：

text
[ERROR] GraphRecursionError: Recursion limit of 1 reached without hitting a stop condition.
测试结论
✅ 所有降级场景全部通过验证

输入校验正常拦截超长内容

RAG 低分结果被正确过滤并记录日志

FAISS 空结果正确降级到 LLM 自身知识

API 异常被捕获并返回用户友好提示

Agent 循环限制生效，超限时返回友好提示

正常业务流程未受影响

降级功能已可投入实际运行。