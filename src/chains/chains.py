"""
三个 Chain 的初始化 + Memory 挂载

- General Chain：不带工具，纯对话（成本最低）
- Sale Chain：报价/售前（带工具：算保费、查条款）
- Service Chain：理赔/售后（带工具：查保单、查条款、转人工）

所有 Chain 共用同一个 MemorySaver（基于 session_id）
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional
import os

# ---------- 全局 Memory（所有 Chain 共用） ----------
shared_memory = MemorySaver()


# ---------- 工具函数（占位 Mock，后续替换） ----------
def mock_tool(tool_name: str, **kwargs) -> str:
    """占位工具，用于测试 Chain 是否能正常初始化"""
    return f"[Mock] {tool_name} 被调用，参数: {kwargs}"


# ---------- 初始化三个 Chain ----------
def init_chains(api_key: Optional[str] = None, model_name: str = "deepseek-chat"):
    """
    初始化三个 Chain，并返回它们和共享 Memory

    参数:
        api_key: DeepSeek API Key（如果不传，从环境变量 DEEPSEEK_API_KEY 读取）
        model_name: 模型名称，默认 deepseek-chat

    返回:
        (general_chain, agent_sale, agent_service, memory)
    """
    # 1. 获取 API Key
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请提供 DeepSeek API Key 或设置环境变量 DEEPSEEK_API_KEY")

    # 2. 创建 LLM 实例（DeepSeek 兼容 OpenAI 接口）
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )

    # 3. 普通链（General Chain）：不带工具，纯对话
    general_chain = llm

    # 4. 报价链（Sale Chain）：带工具（目前是 Mock）
    from langchain_core.tools import tool

    @tool
    def calculate_premium(car_model: str, driver_age: int, years_driving: int) -> str:
        """
        估算车险保费。
        在用户询问保费、报价、投保费用时调用。
        参数：
            car_model: 车型（如：特斯拉 Model 3）
            driver_age: 驾驶员年龄（整数）
            years_driving: 驾龄（整数）
        """
        return mock_tool("calculate_premium", car_model=car_model, driver_age=driver_age, years_driving=years_driving)

    @tool
    def search_insurance_terms(query: str) -> str:
        """
        搜索保险条款内容（RAG）。
        当用户询问具体条款、保障范围、免责条款时调用。
        参数：
            query: 搜索关键词（如：车损险、第三方责任险）
        """
        return mock_tool("search_insurance_terms", query=query)

    # 报价链的工具列表
    sale_tools = [calculate_premium, search_insurance_terms]

    # 5. 理赔链（Service Chain）：带工具
    @tool
    def query_policy(policy_id: str, id_card: str) -> str:
        """
        查询保单详情（保额、到期日、险种等）。
        当用户询问保单信息、保单状态时调用。
        参数：
            policy_id: 保单号（字符串）
            id_card: 身份证号（字符串）
        """
        return mock_tool("query_policy", policy_id=policy_id, id_card=id_card)

    @tool
    def transfer_to_human(reason: str) -> str:
        """
        转人工客服。
        仅在用户明确说出"转人工"、"投诉"、"我要人工"时调用。
        参数：
            reason: 转人工的原因（用户原话摘要）
        """
        return "__TRANSFER__"

    service_tools = [query_policy, search_insurance_terms, transfer_to_human]

    # 6. 使用 create_react_agent 创建 Agent（带工具）
    agent_sale = create_react_agent(
        model=llm,
        tools=sale_tools,
        checkpointer=shared_memory,
        prompt="你是一个车险售前助手，帮助用户计算保费、推荐投保方案。请友好、专业地回答。"
    )

    agent_service = create_react_agent(
        model=llm,
        tools=service_tools,
        checkpointer=shared_memory,
        prompt="你是一个车险售后助手，帮助用户查询保单、解释理赔条款、处理投诉。必要时可转人工。"
    )

    return general_chain, agent_sale, agent_service, shared_memory


# ---------- 测试代码 ----------
if __name__ == "__main__":
    print(">>> 开始测试 Chain 初始化...")

    try:
        general, sale_agent, service_agent, memory = init_chains()

        print("✅ 三个 Chain 初始化成功！")
        print(f"  - General Chain 类型: {type(general).__name__}")
        print(f"  - Sale Agent 类型: {type(sale_agent).__name__}")
        print(f"  - Service Agent 类型: {type(service_agent).__name__}")
        print(f"  - Memory 类型: {type(memory).__name__}")

        print("\n✅ 初始化测试通过。")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n请确保:")
        print("  1. 已设置环境变量 DEEPSEEK_API_KEY")
        print("  2. 或者修改 init_chains() 调用，传入 api_key 参数")