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
from src.state import set_graph


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    # ---------- 预加载：启动时加载所有模型 ----------
    logger.info("正在预加载模型...")
    from src.chains.chains import init_graph
    from src.rag import init_rag_components

    try:
        # 初始化 StateGraph 编排图
        graph, llm, llm_classifier = init_graph()
        set_graph(graph)
        # 初始化 RAG（启动预加载全部组件：FAISS/Embedding/CrossEncoder/jieba/BM25/词典 + 预热推理）
        init_rag_components()
        logger.info("预加载完成")
    except Exception as e:
        logger.warning("预加载失败: %s，服务仍会启动，但第一条消息可能较慢", e)

    # ---------- 连接层诊断：主动测 classifier(dashscope)+deepseek 的 DNS/TCP/TLS 分段耗时 ----------
    # 判断偶发 TLS 握手卡死是本机网络还是代码问题；默认关闭，LLM_CONN_DIAG=1 开启。
    if os.environ.get("LLM_CONN_DIAG", "0") == "1":
        try:
            from src.utils.conn_diag import startup_diagnose
            startup_diagnose()
        except Exception as e:
            logger.warning("连接层诊断失败（不影响服务）: %s", e)

    # ---------- 连接预热：提前建立到 LLM API 的 TCP/TLS 连接 ----------
    try:
        from src.chains.chains import warmup_llm
        warmup_llm(llm, llm_classifier)
    except Exception as e:
        logger.warning("LLM API 预热失败（不影响服务）: %s", e)

    # ---------- 长期记忆：建 PG 语义表 + 实例化 sqlite 结构化存储 ----------
    try:
        from src.db import init_db
        init_db()                       # 幂等：建 documents/semantic_memory 表 + 索引
    except Exception as e:
        logger.warning("init_db 失败（不影响服务启动）: %s", e)
    try:
        from src.memory import structured_store  # noqa: F401  触发 sqlite 建表
        logger.info("长期记忆存储就绪")
    except Exception as e:
        logger.warning("记忆存储初始化失败（不影响服务）: %s", e)

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
