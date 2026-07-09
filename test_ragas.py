"""
RAGAS 评估脚本
基于 insurance_terms.txt 条款数据，构建 15 条问答测试集，
调用现有 RAG 链路 + DeepSeek LLM 生成回答，
使用 RAGAS 4 大指标量化评估 RAG 链路质量。
"""

import os
import sys
import json
import time

# 强制离线模式，避免连接 huggingface.co 超时
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 兼容补丁：修复 langchain_community 与 ragas 的导入兼容问题
# ============================================================
from langchain_google_vertexai import ChatVertexAI
import langchain_community.chat_models
_vertexai_mod = type(sys)('vertexai')
_vertexai_mod.ChatVertexAI = ChatVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = _vertexai_mod

from src.logger import setup_logging, get_logger
from src.rag import init_rag, search_terms
from langchain_openai import ChatOpenAI

setup_logging()
logger = get_logger(__name__)

# ============================================================
# 1. 测试集（15 条，基于 insurance_terms.txt 条款内容）
# ============================================================

TEST_DATASET = [
    {
        "question": "车损险保障哪些情况造成的损失？",
        "ground_truth": "车损险保障碰撞、倾覆、坠落、火灾、爆炸、自燃、外界物体坠落倒塌、暴风龙卷风雷击雹灾暴雨洪水海啸、地陷冰陷崖崩雪崩泥石流滑坡，以及载运车辆的渡船遭受自然灾害造成的损失。"
    },
    {
        "question": "第三方责任险的免赔率是多少？",
        "ground_truth": "第三方责任险免赔率：负全部责任免赔20%，负主要责任免赔15%，负同等责任免赔10%，负次要责任免赔5%。"
    },
    {
        "question": "交强险的死亡伤残赔偿限额是多少？",
        "ground_truth": "交强险死亡伤残赔偿限额为18万元，医疗费用赔偿限额1.8万元，财产损失赔偿限额2000元。无责情况下为上述限额的10%。"
    },
    {
        "question": "玻璃险保不保天窗玻璃？",
        "ground_truth": "玻璃险不保天窗玻璃。玻璃险只保障挡风玻璃和车窗玻璃单独破碎，不包含天窗玻璃、车灯玻璃、后视镜玻璃。"
    },
    {
        "question": "自燃险的免赔额是多少？",
        "ground_truth": "自燃险赔偿金额按实际损失计算，但需扣除免赔额，通常为损失金额的20%。"
    },
    {
        "question": "盗抢险赔偿金额怎么算？",
        "ground_truth": "盗抢险赔偿金额按保险金额扣除折旧后的实际价值计算，免赔率通常为20%。需经公安机关立案侦查满60天未查明下落才赔付。"
    },
    {
        "question": "无法找到第三方特约险有什么用？",
        "ground_truth": "无法找到第三方特约险保障车辆因第三方原因受损且无法找到第三方时，本应扣除的30%免赔率由保险公司承担，投保后车损险免赔率由30%降为0。"
    },
    {
        "question": "驾乘险每座赔偿限额有哪些选择？",
        "ground_truth": "驾乘险每座赔偿限额由投保人选择：1万元、2万元、5万元、10万元。免赔率与三者险相同。"
    },
    {
        "question": "交强险是国家强制要求买的吗？",
        "ground_truth": "是的，交强险（机动车交通事故责任强制保险）是国家规定必须购买的保险，保障第三方受害人的人身伤亡和财产损失。"
    },
    {
        "question": "车损险的保险金额怎么确定？",
        "ground_truth": "车损险保险金额由投保人和保险人在投保时协商确定，但不得超过保险价值。"
    },
    {
        "question": "第三方责任险保额可以选择多少？",
        "ground_truth": "第三方责任险保险金额由投保人在5万元至100万元之间选择。"
    },
    {
        "question": "玻璃险是独立险种还是附加险？",
        "ground_truth": "玻璃险（玻璃单独破碎险）是车损险的附加险种，不是独立险种。"
    },
    {
        "question": "自燃险保障哪些原因引起的自燃？",
        "ground_truth": "自燃险保障被保险车辆因电路、线路、油路、供油系统、供气系统问题或货物自身原因起火燃烧造成的损失。"
    },
    {
        "question": "盗抢险需要等多久才能赔付？",
        "ground_truth": "盗抢险需经公安机关立案侦查后满60天未查明下落，才能进行赔付。"
    },
    {
        "question": "交强险无责情况下赔偿限额是多少？",
        "ground_truth": "交强险无责情况下的赔偿限额为有责限额的10%，即死亡伤残1.8万元、医疗费用1800元、财产损失200元。"
    },
]


# ============================================================
# 2. 生成回答（调用 RAG 检索 + LLM 生成）
# ============================================================

def generate_answer(question: str, contexts: list) -> str:
    """使用 DeepSeek LLM 基于检索到的上下文生成回答"""
    llm = ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )

    context_text = "\n\n".join(contexts) if contexts else "未检索到相关条款。"

    prompt = f"""你是一个车险客服助手。请根据以下保险条款内容，简洁准确地回答用户的问题。

## 参考条款
{context_text}

## 用户问题
{question}

请直接回答，不要说"根据条款"等前缀。"""

    response = llm.invoke(prompt)
    return response.content


def run_rag_pipeline(test_data: list) -> list:
    """对每条测试数据执行 RAG 检索 + LLM 生成，收集评估所需数据"""
    results = []
    total = len(test_data)

    for i, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("[%d/%d] 问题: %s", i, total, question)

        # Step 1: RAG 检索
        contexts = search_terms(question, top_k=3)
        logger.info("  检索到 %d 条上下文", len(contexts))

        # Step 2: LLM 生成回答
        answer = generate_answer(question, contexts)
        logger.info("  回答: %s...", answer[:60])

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        # 避免 API 限流
        time.sleep(1)

    return results


# ============================================================
# 3. RAGAS 评估
# ============================================================

def run_evaluation(results: list):
    """使用 RAGAS 计算 4 大指标"""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # 构建 HuggingFace Dataset
    dataset = Dataset.from_list(results)

    # 配置 LLM（RAGAS 用它做指标评估）
    # 增加 request_timeout 避免 DeepSeek API 超时
    llm = ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        request_timeout=120,  # 120秒超时（默认60s太短）
    )

    # 配置本地 Embeddings（DeepSeek 无 /embeddings 接口，用本地模型替代）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    # 执行评估
    logger.info("=" * 50)
    logger.info("开始 RAGAS 评估（4 大指标）...")
    logger.info("=" * 50)

    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,    # 本地 embeddings，避免调 DeepSeek /embeddings
        batch_size=1,             # 串行执行，避免 DeepSeek API 限流超时
        raise_exceptions=False,   # 单条失败不中断，继续评估其余数据
    )

    return score


# ============================================================
# 4. 输出报告
# ============================================================

def print_report(score):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("           RAGAS 评估报告")
    print("=" * 60)

    # ragas 0.2.x 返回 EvaluationResult 对象
    # scores 是 List[Dict]（每个样本的分数），_repr_dict 是 averaged 指标
    if hasattr(score, '_repr_dict'):
        score_dict = score._repr_dict
    elif hasattr(score, 'scores') and isinstance(score.scores, list):
        # 手动计算平均值
        from collections import defaultdict
        import math
        metric_sums = defaultdict(float)
        metric_counts = defaultdict(int)
        for sample in score.scores:
            for k, v in sample.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    metric_sums[k] += v
                    metric_counts[k] += 1
        score_dict = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
    elif isinstance(score, dict):
        score_dict = score
    else:
        score_dict = {}

    for metric_name, metric_value in score_dict.items():
        if metric_name != "dataset":
            print(f"  {metric_name:30s}: {metric_value:.4f}")

    print("=" * 60)

    # 保存结果
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {k: v for k, v in score_dict.items() if k != "dataset"},
    }

    report_path = os.path.join("data", "ragas_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("评估报告已保存至: %s", report_path)


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    logger.info(">>> 开始 RAGAS 评估流程...")

    # 1. 初始化 RAG
    logger.info("初始化 RAG 系统...")
    init_rag()

    # 2. 执行 RAG 流水线，收集数据
    logger.info("执行 RAG 流水线（检索 + 生成）...")
    results = run_rag_pipeline(TEST_DATASET)

    # 3. RAGAS 评估
    logger.info("执行 RAGAS 评估...")
    score = run_evaluation(results)

    # 4. 输出报告
    print_report(score)

    logger.info(">>> RAGAS 评估完成！")
