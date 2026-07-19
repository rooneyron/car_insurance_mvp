"""
RAGAS 评估脚本 - Prompt 对比版
对比 sale 和 service 两种 Prompt 对 RAG 回答风格的影响
"""

import os
import sys
import json
import time
import math

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 兼容补丁
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
# 1. 测试集（15 条）
# ============================================================

TEST_DATASET = [
    {"question": "车损险保障哪些情况造成的损失？", "ground_truth": "车损险保障碰撞、倾覆、坠落、火灾、爆炸、自燃、外界物体坠落倒塌、暴风龙卷风雷击雹灾暴雨洪水海啸、地陷冰陷崖崩雪崩泥石流滑坡，以及载运车辆的渡船遭受自然灾害造成的损失。"},
    {"question": "第三方责任险的免赔率是多少？", "ground_truth": "第三方责任险免赔率：负全部责任免赔20%，负主要责任免赔15%，负同等责任免赔10%，负次要责任免赔5%。"},
    {"question": "交强险的死亡伤残赔偿限额是多少？", "ground_truth": "交强险死亡伤残赔偿限额为18万元，医疗费用赔偿限额1.8万元，财产损失赔偿限额2000元。无责情况下为上述限额的10%。"},
    {"question": "玻璃险保不保天窗玻璃？", "ground_truth": "玻璃险不保天窗玻璃。玻璃险只保障挡风玻璃和车窗玻璃单独破碎，不包含天窗玻璃、车灯玻璃、后视镜玻璃。"},
    {"question": "自燃险的免赔额是多少？", "ground_truth": "自燃险赔偿金额按实际损失计算，但需扣除免赔额，通常为损失金额的20%。"},
    {"question": "盗抢险赔偿金额怎么算？", "ground_truth": "盗抢险赔偿金额按保险金额扣除折旧后的实际价值计算，免赔率通常为20%。需经公安机关立案侦查满60天未查明下落才赔付。"},
    {"question": "无法找到第三方特约险有什么用？", "ground_truth": "无法找到第三方特约险保障车辆因第三方原因受损且无法找到第三方时，本应扣除的30%免赔率由保险公司承担，投保后车损险免赔率由30%降为0。"},
    {"question": "驾乘险每座赔偿限额有哪些选择？", "ground_truth": "驾乘险每座赔偿限额由投保人选择：1万元、2万元、5万元、10万元。免赔率与三者险相同。"},
    {"question": "交强险是国家强制要求买的吗？", "ground_truth": "是的，交强险（机动车交通事故责任强制保险）是国家规定必须购买的保险，保障第三方受害人的人身伤亡和财产损失。"},
    {"question": "车损险的保险金额怎么确定？", "ground_truth": "车损险保险金额由投保人和保险人在投保时协商确定，但不得超过保险价值。"},
    {"question": "第三方责任险保额可以选择多少？", "ground_truth": "第三方责任险保险金额由投保人在5万元至100万元之间选择。"},
    {"question": "玻璃险是独立险种还是附加险？", "ground_truth": "玻璃险（玻璃单独破碎险）是车损险的附加险种，不是独立险种。"},
    {"question": "自燃险保障哪些原因引起的自燃？", "ground_truth": "自燃险保障被保险车辆因电路、线路、油路、供油系统、供气系统问题或货物自身原因起火燃烧造成的损失。"},
    {"question": "盗抢险需要等多久才能赔付？", "ground_truth": "盗抢险需经公安机关立案侦查后满60天未查明下落，才能进行赔付。"},
    {"question": "交强险无责情况下赔偿限额是多少？", "ground_truth": "交强险无责情况下的赔偿限额为有责限额的10%，即死亡伤残1.8万元、医疗费用1800元、财产损失200元。"},
]

# ============================================================
# 2. 两种 Prompt 模板（直接提取自 chains.py）
# ============================================================

PROMPT_SALE = """你是一个车险售前助手，帮助用户计算保费、推荐投保方案。

重要规则：
1. 如果用户没有提供车型、年龄、驾龄，请检查对话历史中是否曾经提供过这些信息，如果有则直接使用。
2. 只有对话历史中也没有这些信息时，才向用户询问。
3. 请友好、专业地回答。"""

PROMPT_SERVICE = """你是一个车险售后助手，帮助用户查询保单、解释理赔条款、处理投诉。

重要规则：
1. 当用户查询保单时，如果用户没有提供身份证号，请检查对话历史中是否曾经提供过，如果有则直接使用。
2. 如果对话历史中也没有身份证号，再向用户询问。
3. 不要重复索要用户已经提供过的信息。
4. 必要时可转人工。"""

# ============================================================
# 3. 全局 LLM 实例（复用，避免重复创建）
# ============================================================

_llm_instance = None
_ragas_llm_instance = None


def _get_llm(temperature=0.3):
    """获取复用的 LLM 实例（用于生成回答）"""
    global _llm_instance
    if _llm_instance is None or _llm_instance.temperature != temperature:
        _llm_instance = ChatOpenAI(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
        )
    return _llm_instance


def _patch_openai_n1(llm):
    """Monkey-patch OpenAI sync+async client，强制 n=1（DeepSeek 不支持 n>1）"""
    if hasattr(llm, 'client') and llm.client:
        original_create = llm.client.create
        def patched_create(**kwargs):
            kwargs["n"] = 1
            return original_create(**kwargs)
        llm.client.create = patched_create
    if hasattr(llm, 'async_client') and llm.async_client:
        original_acreate = llm.async_client.create
        async def apatched_create(**kwargs):
            kwargs["n"] = 1
            return await original_acreate(**kwargs)
        llm.async_client.create = apatched_create
    return llm


def _get_ragas_llm():
    """获取复用的 RAGAS 评估 LLM 实例"""
    global _ragas_llm_instance
    if _ragas_llm_instance is None:
        _ragas_llm_instance = _patch_openai_n1(ChatOpenAI(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0,
            request_timeout=120,
        ))
    return _ragas_llm_instance


# ============================================================
# 4. 生成回答（复用 LLM 实例）
# ============================================================

def generate_answer(question: str, contexts: list, system_prompt: str) -> str:
    llm = _get_llm(temperature=0.3)

    context_text = "\n\n".join(contexts) if contexts else "未检索到相关条款。"

    full_prompt = f"""{system_prompt}

请根据以下保险条款内容回答用户的问题。

## 参考条款
{context_text}

## 用户问题
{question}

请直接回答，不要说"根据条款"等前缀。"""

    response = llm.invoke(full_prompt)
    return response.content


def run_rag_pipeline(test_data: list, system_prompt: str, existing_results: list = None) -> list:
    """
    运行 RAG 流水线，支持断点续跑。
    existing_results: 已完成的结果列表，会跳过已完成的 question。
    """
    results = list(existing_results) if existing_results else []
    completed_questions = {r["question"] for r in results}
    total = len(test_data)

    for i, item in enumerate(test_data, 1):
        question = item["question"]

        # 断点续跑：跳过已完成的
        if question in completed_questions:
            logger.info("[%d/%d] 跳过已完成: %s", i, total, question)
            continue

        ground_truth = item["ground_truth"]
        logger.info("[%d/%d] 问题: %s", i, total, question)

        try:
            contexts = search_terms(question, top_k=3)
            answer = generate_answer(question, contexts, system_prompt)

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            })
        except Exception as e:
            logger.error("[%d/%d] 处理失败，跳过: %s | 错误: %s", i, total, question, str(e))
            continue

        time.sleep(1)

    return results


def run_evaluation(results: list) -> dict:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy
    # from ragas.metrics import faithfulness, context_precision, context_recall
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings

    dataset = Dataset.from_list(results)

    llm = _get_ragas_llm()

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("开始 RAGAS 评估（仅 answer_relevancy）...")

    score = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,
            # faithfulness,      # 需要时取消注释
            # context_precision, # 需要时取消注释
            # context_recall,    # 需要时取消注释
        ],
        llm=llm,
        embeddings=embeddings,
        batch_size=1,
        raise_exceptions=False,
    )

    if hasattr(score, '_repr_dict'):
        return score._repr_dict
    elif isinstance(score, dict):
        return score
    else:
        return {}


def save_results(results: list, score: dict, label: str):
    # 保存样本
    samples_path = os.path.join("data", f"ragas_samples_{label}.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存评分
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": label,
        "metrics": score,
    }
    report_path = os.path.join("data", f"ragas_report_{label}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("✅ %s 完成 | 指标: %s", label,
                ", ".join(f"{k}={v:.4f}" for k, v in score.items() if isinstance(v, (int, float)) and not math.isnan(v)))
    logger.info("   样本: %s", samples_path)
    logger.info("   报告: %s", report_path)


def _load_existing_results(label: str) -> list:
    """加载已有的样本结果，用于断点续跑"""
    samples_path = os.path.join("data", f"ragas_samples_{label}.json")
    if os.path.exists(samples_path):
        try:
            with open(samples_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


# ============================================================
# 5. 主流程
# ============================================================

if __name__ == "__main__":
    logger.info(">>> 开始 RAGAS Prompt 对比测试（sale vs service）...")
    init_rag()

    # ========== 冒烟测试：先跑 1 条验证全链路 ==========
    logger.info("🔥 冒烟测试：跑 1 条验证全链路（RAG + LLM + RAGAS 评估）...")
    smoke_data = TEST_DATASET[:1]

    smoke_sale = run_rag_pipeline(smoke_data, PROMPT_SALE)
    if not smoke_sale:
        logger.error("❌ 冒烟测试失败：未能生成任何结果，请检查 LLM 和 RAG 配置。中止执行。")
        sys.exit(1)

    smoke_score = run_evaluation(smoke_sale)
    smoke_ar = smoke_score.get("answer_relevancy")
    logger.info("冒烟测试结果: answer_relevancy = %s", smoke_ar)

    if smoke_ar is None or (isinstance(smoke_ar, float) and math.isnan(smoke_ar)):
        logger.error("❌ 冒烟测试失败！answer_relevancy 为 NaN，请检查 LLM 和 RAGAS 配置。中止执行。")
        sys.exit(1)

    logger.info("✅ 冒烟测试通过！开始正式跑 15 条...\n")

    # ========== 正式测试：跑全量 15 条（支持断点续跑）==========
    # 跑 sale Prompt
    logger.info("=" * 50)
    logger.info("🔵 测试 Prompt: SALE")
    existing_sale = _load_existing_results("sale")
    results_sale = run_rag_pipeline(TEST_DATASET, PROMPT_SALE, existing_results=existing_sale)
    if results_sale:
        score_sale = run_evaluation(results_sale)
        save_results(results_sale, score_sale, "sale")
    else:
        logger.error("SALE 流水线无有效结果，跳过评估")
        score_sale = {}

    # 跑 service Prompt
    logger.info("=" * 50)
    logger.info("🟢 测试 Prompt: SERVICE")
    existing_service = _load_existing_results("service")
    results_service = run_rag_pipeline(TEST_DATASET, PROMPT_SERVICE, existing_results=existing_service)
    if results_service:
        score_service = run_evaluation(results_service)
        save_results(results_service, score_service, "service")
    else:
        logger.error("SERVICE 流水线无有效结果，跳过评估")
        score_service = {}

    # 打印对比结果
    print("\n" + "=" * 60)
    print("           📊 Prompt 对比结果")
    print("=" * 60)
    for metric in ["answer_relevancy"]:
        sale_val = score_sale.get(metric)
        service_val = score_service.get(metric)
        sale_str = f"{sale_val:.4f}" if isinstance(sale_val, (int, float)) and not math.isnan(sale_val) else "N/A"
        service_str = f"{service_val:.4f}" if isinstance(service_val, (int, float)) and not math.isnan(service_val) else "N/A"
        print(f"  {metric:25s} | SALE: {sale_str:>8s} | SERVICE: {service_str:>8s}")
    print("=" * 60)

    logger.info(">>> 全部完成！")
