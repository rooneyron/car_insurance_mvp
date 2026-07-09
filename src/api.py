"""
FastAPI 应用工厂
创建 FastAPI 实例、注册中间件和路由。
"""

import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import jwt
from pydantic import BaseModel
from src.constants import APP_VERSION, SERVICE_NAME, JWT_ALGORITHM, PUBLIC_PATHS
from src.token_usage import get_today_usage, DAILY_TOKEN_LIMIT
from src.chat import chat_api
from src.logger import get_logger

logger = get_logger(__name__)


def _verify_jwt_token(token: str, secret: str) -> tuple:
    """
    验证 JWT Token。
    返回 (is_valid: bool, error_response: JSONResponse | None)
    """
    if not token:
        return False, JSONResponse(status_code=401, content={"detail": "Missing token"})
    try:
        jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return True, None
    except jwt.ExpiredSignatureError:
        return False, JSONResponse(status_code=401, content={"detail": "Token expired"})
    except jwt.InvalidTokenError:
        return False, JSONResponse(status_code=401, content={"detail": "Invalid token"})


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    application = FastAPI(title="车险智能客服 MVP", version=APP_VERSION)

    # ---------- Token 验证配置 ----------
    access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")
    if not access_token_secret:
        raise RuntimeError("环境变量 ACCESS_TOKEN_SECRET 未配置，请在 .env 文件中设置")

    # ---------- Token 验证中间件 ----------
    @application.middleware("http")
    async def verify_token(request: Request, call_next):
        path = request.url.path

        # 放行白名单路径
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # 放行 /gradio/ 子资源（静态文件、API 等）
        if path.startswith("/gradio/"):
            return await call_next(request)

        # 其余路径需要 token（/gradio 页面通过 query param，其他路径同理）
        token = request.query_params.get("token")
        is_valid, error_resp = _verify_jwt_token(token, access_token_secret)
        if not is_valid:
            return error_resp

        return await call_next(request)

    # ---------- 注册路由 ----------
    register_routes(application)

    return application


def register_routes(application: FastAPI):
    """注册所有 API 路由"""

    @application.get("/health")
    async def health_check():
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "service": SERVICE_NAME,
                "version": APP_VERSION,
                "timestamp": int(time.time())
            }
        )

    @application.get("/")
    async def root():
        return {
            "message": "车险智能客服 MVP",
            "docs": "/docs",
            "health": "/health",
            "gradio": "/gradio"
        }

    @application.get("/queryToken")
    async def query_token():
        """查询今日 Token 使用量和费用"""
        usage = get_today_usage()
        return {
            "date": usage["date"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "daily_token_limit": DAILY_TOKEN_LIMIT,
        }

    class ChatRequest(BaseModel):
        session_id: str
        message: str

    @application.post("/chat")
    async def chat(req: ChatRequest):
        """对话接口"""
        result = chat_api(req.session_id, req.message)
        return result
"""
FastAPI 应用工厂
创建 FastAPI 实例、注册中间件和路由。
"""

import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import jwt
from pydantic import BaseModel
from src.token_usage import get_today_usage, DAILY_TOKEN_LIMIT
from src.chat import chat_api
from src.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    application = FastAPI(title="车险智能客服 MVP", version="0.1.0")

    # ---------- Token 验证配置 ----------
    access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")
    if not access_token_secret:
        raise RuntimeError("环境变量 ACCESS_TOKEN_SECRET 未配置，请在 .env 文件中设置")

    # ---------- Token 验证中间件 ----------
    @application.middleware("http")
    async def verify_token(request: Request, call_next):
        path = request.url.path

        # 放行健康检查、根路径、清单、favicon、Token 查询接口
        if path in ("/health", "/", "/manifest.json", "/favicon.ico", "/queryToken", "/docs"):
            return await call_next(request)

        # 放行 OpenAPI JSON（FastAPI 文档依赖）
        if path == "/openapi.json":
            return await call_next(request)

        # 放行所有以 /gradio/ 开头的子资源（静态文件、API 等）
        if path.startswith("/gradio/"):
            return await call_next(request)

        # /gradio 页面本身需要 token
        if path == "/gradio":
            token = request.query_params.get("token")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing token"})
            try:
                jwt.decode(token, access_token_secret, algorithms=["HS256"])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            return await call_next(request)

        # 其他路径也需要 token
        token = request.query_params.get("token")
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Missing token"})
        try:
            jwt.decode(token, access_token_secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return JSONResponse(status_code=401, content={"detail": "Token expired"})
        except jwt.InvalidTokenError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})

        return await call_next(request)

    # ---------- 注册路由 ----------
    register_routes(application)

    return application


def register_routes(application: FastAPI):
    """注册所有 API 路由"""

    @application.get("/health")
    async def health_check():
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "service": "car_insurance_mvp",
                "version": "0.1.0",
                "timestamp": int(time.time())
            }
        )

    @application.get("/")
    async def root():
        return {
            "message": "车险智能客服 MVP",
            "docs": "/docs",
            "health": "/health",
            "gradio": "/gradio"
        }

    @application.get("/queryToken")
    async def query_token():
        """查询今日 Token 使用量和费用"""
        usage = get_today_usage()
        return {
            "date": usage["date"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "daily_token_limit": DAILY_TOKEN_LIMIT,
        }

    class ChatRequest(BaseModel):
        session_id: str
        message: str

    @application.post("/chat")
    async def chat(req: ChatRequest):
        """对话接口"""
        result = chat_api(req.session_id, req.message)
        return result
