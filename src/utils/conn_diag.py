"""
连接层诊断工具：主动测量到目标主机的 DNS 解析 / TCP 连接 / TLS 握手分段耗时。

用途：排查偶发的 TLS 握手卡死（实测 dashscope qwen-turbo 曾卡 22s 才 ConnectionReset），
      判断到底是「本机网络问题」还是「代码问题」——
      健康时各阶段都应 <1s（日志里 DeepSeek 秒连）；若某阶段单独卡住即定位到症结。

调用点：
  - 启动时 startup_diagnose()：对 classifier(dashscope) + deepseek 各测一次（app.py，默认关，LLM_CONN_DIAG=1 开启）；
  - 连接失败后 diagnose_url()/diagnose_connection()：复测当时的连接质量（chains.py warmup 失败处已接入）。

只读诊断，不改任何业务逻辑。
"""
import os
import time
import socket
import ssl
from urllib.parse import urlsplit

from src.logger import get_logger

logger = get_logger(__name__)


def diagnose_connection(host: str, port: int = 443, timeout: float = 10.0) -> dict:
    """
    诊断到 host:port 的连接质量，分段记录 DNS 解析 / TCP 连接 / TLS 握手耗时。

    返回 dict（含 dns_time_ms / resolved_ip / tcp_time_ms / tls_time_ms / total_time_ms / error），
    并打一条 [ConnDiag] 日志。timeout 为每个阶段的上限（秒）。
    """
    result = {
        "host": host,
        "port": port,
        "dns_time_ms": None,
        "resolved_ip": None,
        "tcp_time_ms": None,
        "tls_time_ms": None,
        "total_time_ms": None,
        "error": None,
    }
    t0 = time.time()
    sock = None
    ssock = None
    try:
        # ① DNS 解析
        t = time.time()
        addr_info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        ip = addr_info[0][4][0]
        result["dns_time_ms"] = int((time.time() - t) * 1000)
        result["resolved_ip"] = ip

        # ② TCP 连接
        t = time.time()
        sock = socket.create_connection((ip, port), timeout=timeout)
        result["tcp_time_ms"] = int((time.time() - t) * 1000)

        # ③ TLS 握手（create_default_context 会校验证书，等价真实 HTTPS 握手）
        t = time.time()
        ctx = ssl.create_default_context()
        ssock = ctx.wrap_socket(sock, server_hostname=host)
        result["tls_time_ms"] = int((time.time() - t) * 1000)

        result["total_time_ms"] = int((time.time() - t0) * 1000)
        logger.info(
            "[ConnDiag] %s:%d | DNS=%sms IP=%s TCP=%sms TLS=%sms 总=%sms",
            host, port, result["dns_time_ms"], ip,
            result["tcp_time_ms"], result["tls_time_ms"], result["total_time_ms"],
        )
    except Exception as e:
        result["error"] = str(e)
        result["total_time_ms"] = int((time.time() - t0) * 1000)
        logger.warning(
            "[ConnDiag] %s:%d 连接失败 | 耗时=%sms 错误=%s | 已完成 DNS=%sms IP=%s TCP=%sms TLS=%sms",
            host, port, result["total_time_ms"], e,
            result["dns_time_ms"], result["resolved_ip"],
            result["tcp_time_ms"], result["tls_time_ms"],
        )
    finally:
        try:
            if ssock is not None:
                ssock.close()          # 关闭 ssock 会一并关闭底层 sock
            elif sock is not None:
                sock.close()
        except Exception:
            pass
    return result


def diagnose_url(url: str, timeout: float = 10.0) -> dict:
    """从 base_url 解析出 host/port 后调用 diagnose_connection。"""
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        logger.warning("[ConnDiag] 无法从 URL 解析 host: %s", url)
        return {"host": None, "error": "bad_url"}
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return diagnose_connection(host, port, timeout)


def startup_diagnose(timeout: float = 10.0) -> None:
    """
    启动时对已配置的 LLM 端点各测一次连接质量（按 host 去重）。

    端点来源：
      - classifier：ROUTER_CLASSIFIER_BASE_URL（.env 现为 dashscope 兼容端点）
      - 主 LLM/reviewer：DeepSeek https://api.deepseek.com/v1
    """
    deepseek_url = "https://api.deepseek.com/v1"
    classifier_url = os.environ.get("ROUTER_CLASSIFIER_BASE_URL", deepseek_url)
    logger.info("[ConnDiag] 启动连接诊断（各阶段应 <1s；某段单独卡住即症结）...")
    seen = set()
    for url in (classifier_url, deepseek_url):
        host = urlsplit(url).hostname
        if not host or host in seen:
            continue
        seen.add(host)
        diagnose_url(url, timeout)
