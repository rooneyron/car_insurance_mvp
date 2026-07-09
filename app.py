"""
车险智能客服 MVP - 主入口
职责：环境初始化、模型预加载、组装各模块并启动服务。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.logger import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

import uvicorn
import gradio as gr
from src.api import create_app
from src.gradio_ui import create_gradio_interface
from src.state import set_chains


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    # ---------- 预加载：启动时加载所有模型 ----------
    logger.info("正在预加载模型...")
    from src.chains.chains import init_chains
    from src.rag import init_rag

    try:
        # 初始化 Chain，获取三个 Agent 和 Memory
        general_chain, agent_sale, agent_service, _ = init_chains()
        set_chains(general_chain, agent_sale, agent_service)
        # 初始化 RAG（加载 FAISS 索引和模型）
        init_rag()
        logger.info("预加载完成")
    except Exception as e:
        logger.warning("预加载失败: %s，服务仍会启动，但第一条消息可能较慢", e)

    # ---------- 组装应用 ----------
    app = create_app()
    demo = create_gradio_interface()
    app = gr.mount_gradio_app(app, demo, path="/gradio")

    logger.info("=" * 50)
    logger.info("车险智能客服 MVP 已启动")
    logger.info("=" * 50)
    logger.info("API 文档: http://127.0.0.1:8000/docs")
    logger.info("健康检查: http://127.0.0.1:8000/health")
    logger.info("Gradio 界面: http://127.0.0.1:8000/gradio")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
