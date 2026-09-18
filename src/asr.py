# -*- coding: utf-8 -*-
"""火山引擎豆包语音 ASR 模块 - 录音文件识别极速版

提供 speech_to_text() 函数，将音频二进制转文字。
- 同步返回，无需轮询
- 支持 wav/mp3/ogg_opus 格式
- 最大 100MB / 2小时

环境变量:
    VOLC_ASR_API_KEY: 新版 API Key (X-Api-Key)
    备选: VOLC_ASR_APP_KEY + VOLC_ASR_ACCESS_KEY (旧版双 key，暂未实现)
"""

import os
import json
import uuid
import base64
import subprocess
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)

# 加载 .env（被其他模块 import 时确保环境变量可用）
load_dotenv()

# ============================================================
# 配置
# ============================================================

# API 端点 - 录音文件识别极速版 (同步返回)
ASR_API_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"

# 默认音频参数
DEFAULT_AUDIO_RATE = 16000  # 采样率
DEFAULT_AUDIO_BITS = 16     # 位深
DEFAULT_AUDIO_CHANNEL = 1   # 声道数

# 超时设置
ASR_TIMEOUT = 30  # ASR 请求超时秒数


# ============================================================
# 内部函数
# ============================================================

def _get_api_key() -> str:
    """获取 API Key，优先从环境变量读取"""
    api_key = os.environ.get("VOLC_ASR_API_KEY")
    if api_key:
        return api_key
    
    # 旧版双 key 暂未实现
    app_key = os.environ.get("VOLC_ASR_APP_KEY")
    access_key = os.environ.get("VOLC_ASR_ACCESS_KEY")
    if app_key and access_key:
        logger.warning("旧版双 key 鉴权暂未实现，请使用新版 VOLC_ASR_API_KEY")
    
    return ""


def _build_headers(api_key: str, request_id: str) -> dict:
    """构建请求头"""
    return {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "X-Api-Resource-Id": "volc.seedasr.auc",
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }


def _build_request_body(audio_data: bytes, audio_format: str, audio_codec: str = None) -> dict:
    """
    构建请求体
    
    audio_format: wav / mp3 / ogg_opus
    audio_codec: 默认与 format 相同
    """
    if audio_codec is None:
        audio_codec = audio_format
    
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    
    return {
        "user": {"uid": "feishu_bot"},
        "audio": {
            "data": audio_base64,
            "format": audio_format,
            "codec": audio_codec,
            "rate": DEFAULT_AUDIO_RATE,
            "bits": DEFAULT_AUDIO_BITS,
            "channel": DEFAULT_AUDIO_CHANNEL,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,      # 逆文本正则化 (数字转中文)
            "enable_punc": False,    # 标点恢复
            "enable_ddc": False,     # 语气词过滤
            "enable_speaker_info": False,
            "enable_channel_split": False,
            "show_utterances": False,
            "vad_segment": False,
            "sensitive_words_filter": "",
        },
    }


def _parse_response(response_json: dict) -> str:
    """解析响应，提取识别文字"""
    # 检查错误码 (如果存在)
    code = response_json.get("code")
    if code is not None and code != 0:
        error_msg = response_json.get("message", "未知错误")
        logger.error("[ASR] API 返回错误: code=%s, message=%s", code, error_msg)
        return ""
    
    # 从 result.text 获取识别结果
    result = response_json.get("result", {})
    text = result.get("text")
    if text:
        return text.strip()
    
    # 尝试从 utterances 字段获取
    utterances = result.get("utterances", [])
    if utterances:
        return "".join(u.get("text", "") for u in utterances).strip()
    
    logger.warning("[ASR] 响应中未找到识别结果: %s", json.dumps(response_json, ensure_ascii=False)[:500])
    return ""


def _convert_opus_to_wav(opus_data: bytes) -> bytes:
    """用 ffmpeg 将 opus 转为 16kHz 单声道 wav"""
    try:
        # 检查 ffmpeg 是否可用
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("[ASR] ffmpeg 不可用，无法转换 opus 格式")
        return b""
    
    temp_input = None
    temp_output = None
    try:
        # 写入临时 opus 文件
        with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f:
            f.write(opus_data)
            temp_input = f.name
        
        temp_output = temp_input.replace(".opus", ".wav")
        
        # ffmpeg 转换: 16kHz 单声道 wav
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_input,
            "-ac", "1",       # 单声道
            "-ar", "16000",   # 16kHz 采样率
            "-sample_fmt", "s16",
            temp_output
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode != 0:
            logger.error("[ASR] ffmpeg 转换失败: %s", result.stderr.decode()[:500])
            return b""
        
        # 读取转换后的 wav
        with open(temp_output, "rb") as f:
            return f.read()
    
    except subprocess.TimeoutExpired:
        logger.error("[ASR] ffmpeg 转换超时")
        return b""
    except Exception as e:
        logger.error("[ASR] ffmpeg 转换异常: %s", e)
        return b""
    finally:
        # 清理临时文件
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)


# ============================================================
# 公开接口
# ============================================================

def speech_to_text(audio_bytes: bytes, file_ext: str = "wav") -> str:
    """
    语音转文字
    
    Args:
        audio_bytes: 音频二进制数据
        file_ext: 文件扩展名 (wav/mp3/opus/ogg_opus)
    
    Returns:
        识别出的文字，失败返回空字符串
    """
    if not audio_bytes:
        logger.warning("[ASR] 音频数据为空")
        return ""
    
    api_key = _get_api_key()
    if not api_key:
        logger.error("[ASR] 未配置 VOLC_ASR_API_KEY")
        return ""
    
    # 确定音频格式
    format_map = {
        "wav": "wav",
        "mp3": "mp3",
        "opus": "ogg_opus",  # 飞书 opus 尝试用 ogg_opus 格式
        "ogg_opus": "ogg_opus",
        "ogg": "ogg_opus",
    }
    audio_format = format_map.get(file_ext.lower(), "wav")
    
    # 如果是 opus 格式，先尝试直接传，失败再转 wav
    audio_data = audio_bytes
    actual_format = audio_format
    
    # 第一轮尝试：直接用原始格式
    request_id = str(uuid.uuid4())
    headers = _build_headers(api_key, request_id)
    body = _build_request_body(audio_data, actual_format)
    
    logger.info("[ASR] 请求: format=%s, size=%d bytes", actual_format, len(audio_data))
    
    try:
        response = requests.post(ASR_API_URL, headers=headers, json=body, timeout=ASR_TIMEOUT)
        
        if response.status_code != 200:
            logger.error("[ASR] HTTP %d: %s", response.status_code, response.text[:200])
            # 如果是 opus 格式失败，尝试转 wav
            if file_ext.lower() in ("opus", "ogg", "ogg_opus"):
                logger.info("[ASR] opus 直接识别失败，尝试转 wav...")
                wav_data = _convert_opus_to_wav(audio_bytes)
                if wav_data:
                    return speech_to_text(wav_data, "wav")
            return ""
        
        response_json = response.json()
        text = _parse_response(response_json)
        
        if text:
            logger.info("[ASR] 识别成功: %s", text[:100])
            return text
        
        # 识别结果为空，如果是 opus 格式，尝试转 wav
        if file_ext.lower() in ("opus", "ogg", "ogg_opus") and not text:
            logger.info("[ASR] opus 识别结果为空，尝试转 wav...")
            wav_data = _convert_opus_to_wav(audio_bytes)
            if wav_data:
                return speech_to_text(wav_data, "wav")
        
        return text
    
    except requests.exceptions.Timeout:
        logger.error("[ASR] 请求超时 (%ds)", ASR_TIMEOUT)
        return ""
    except requests.exceptions.RequestException as e:
        logger.error("[ASR] 请求异常: %s", e)
        return ""
    except Exception as e:
        logger.error("[ASR] 未知异常: %s", e)
        return ""


# ============================================================
# 命令行测试
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python -m src.asr <音频文件路径>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"文件不存在: {audio_file}")
        sys.exit(1)
    
    ext = Path(audio_file).suffix.lstrip(".")
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    
    print(f"文件: {audio_file}")
    print(f"大小: {len(audio_data)} 字节")
    print(f"格式: {ext}")
    print("-" * 40)
    
    result = speech_to_text(audio_data, ext)
    
    print("-" * 40)
    if result:
        print(f"识别结果: {result}")
    else:
        print("识别失败")
