"""
火山引擎豆包语音 - 录音文件识别极速版 API 测试
官方文档: https://docs.volcengine.com/docs/DoubaoVoice/recording-file-recognition-lite-http

特点:
- 一次请求同步返回识别结果，无需轮询
- 支持 wav/mp3/ogg opus 格式
- 最大 100MB / 2小时

运行方式:
    python test_volc_asr.py

环境变量:
    VOLC_ASR_API_KEY: 新版 API Key (X-Api-Key)
    备选: VOLC_ASR_APP_KEY + VOLC_ASR_ACCESS_KEY (旧版双 key)
"""

import os
import sys
import json
import uuid
import base64
import requests
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# ============================================================
# 配置
# ============================================================

# API 端点 - 录音文件识别极速版 (同步返回)
API_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"

# 测试音频文件路径
AUDIO_FILE = r"C:\Users\HH\Desktop\火山语音对接\车损险保什么.wav"

# 音频参数 (根据实际文件调整)
AUDIO_FORMAT = "wav"      # wav / mp3 / ogg_opus
AUDIO_CODEC = "raw"       # raw / mp3 / ogg_opus
AUDIO_RATE = 16000        # 采样率
AUDIO_BITS = 16           # 位深
AUDIO_CHANNEL = 1         # 声道数

# ============================================================
# 鉴权配置
# ============================================================

def get_api_key():
    """获取 API Key，优先从环境变量读取"""
    # 新版: 单一 API Key
    api_key = os.environ.get("VOLC_ASR_API_KEY")
    if api_key:
        return api_key, "new"
    
    # 旧版: 双 key 组合 (这里只返回提示，实际旧版需要两个 key)
    app_key = os.environ.get("VOLC_ASR_APP_KEY")
    access_key = os.environ.get("VOLC_ASR_ACCESS_KEY")
    if app_key and access_key:
        # 旧版鉴权需要特殊处理，这里先返回提示
        print("[警告] 旧版双 key 鉴权暂未实现，请使用新版 VOLC_ASR_API_KEY")
        return None, None
    
    return None, None


def build_headers(api_key: str, request_id: str) -> dict:
    """
    构建请求头
    
    新版鉴权 (X-Api-Key):
        - x-api-key: API Key
        - X-Api-Resource-Id: 资源 ID (固定 volc.seedasr.auc)
        - X-Api-Request-Id: 请求 ID (UUID)
        - X-Api-Sequence: 序列号 (-1 表示非流式)
    """
    return {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "X-Api-Resource-Id": "volc.seedasr.auc",
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }


def build_request_body(audio_data: bytes) -> dict:
    """
    构建请求体
    
    根据官方文档，请求体结构:
    {
        "user": {"uid": "用户标识"},
        "audio": {
            "data": "base64编码的音频数据",
            "format": "音频格式",
            "codec": "编码格式",
            "rate": 采样率,
            "bits": 位深,
            "channel": 声道数
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,           # 逆文本正则化 (数字转中文等)
            "enable_punc": False,         # 标点恢复
            "enable_ddc": False,          # 语气词过滤
            "enable_speaker_info": False, # 说话人信息
            "enable_channel_split": False,# 声道分离
            "show_utterances": False,     # 显示语句列表
            "vad_segment": False,         # VAD 分段
            "sensitive_words_filter": ""  # 敏感词过滤
        }
    }
    """
    # 音频转 base64
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    
    return {
        "user": {
            "uid": "豆包语音测试"
        },
        "audio": {
            "data": audio_base64,
            "format": AUDIO_FORMAT,
            "codec": AUDIO_CODEC,
            "rate": AUDIO_RATE,
            "bits": AUDIO_BITS,
            "channel": AUDIO_CHANNEL,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": False,
            "enable_ddc": False,
            "enable_speaker_info": False,
            "enable_channel_split": False,
            "show_utterances": False,
            "vad_segment": False,
            "sensitive_words_filter": "",
        },
    }


def parse_response(response_json: dict) -> str:
    """
    解析响应，提取识别文字
    
    实际响应结构:
    {
        "audio_info": {"duration": 2999},
        "result": {
            "additions": {"duration": "2999"},
            "text": "识别的文字"
        }
    }
    """
    # 检查错误码 (如果存在)
    code = response_json.get("code")
    if code is not None and code != 0:
        error_msg = response_json.get("message", "未知错误")
        return f"[错误] code={code}, message={error_msg}"
    
    # 从 result.text 获取识别结果
    result = response_json.get("result", {})
    text = result.get("text")
    if text:
        return text
    
    # 尝试从 utterances 字段获取
    utterances = result.get("utterances", [])
    if utterances:
        return "".join(u.get("text", "") for u in utterances)
    
    # 尝试从 data 字段获取 (旧版结构)
    data = response_json.get("data", {})
    if data:
        text = data.get("text")
        if text:
            return text
    
    return "[警告] 未找到识别结果，请检查响应结构"


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("火山引擎豆包语音 - 录音文件识别极速版 API 测试")
    print("=" * 60)
    
    # 1. 检查音频文件
    if not os.path.exists(AUDIO_FILE):
        print(f"[错误] 音频文件不存在: {AUDIO_FILE}")
        return
    
    file_size = os.path.getsize(AUDIO_FILE)
    print(f"\n[1] 音频文件: {AUDIO_FILE}")
    print(f"    文件大小: {file_size} 字节 ({file_size/1024:.1f} KB)")
    
    # 2. 读取音频文件
    try:
        with open(AUDIO_FILE, "rb") as f:
            audio_data = f.read()
    except Exception as e:
        print(f"[错误] 读取音频文件失败: {e}")
        return
    
    # 3. 获取 API Key
    api_key, auth_type = get_api_key()
    if not api_key:
        print("\n[错误] 未配置 API Key")
        print("请设置环境变量: set VOLC_ASR_API_KEY=your-api-key")
        print("\n当前环境变量:")
        print(f"  VOLC_ASR_API_KEY: {os.environ.get('VOLC_ASR_API_KEY', '(未设置)')}")
        print(f"  VOLC_ASR_APP_KEY: {os.environ.get('VOLC_ASR_APP_KEY', '(未设置)')}")
        return
    
    print(f"\n[2] 鉴权方式: 新版 X-Api-Key")
    print(f"    API Key: {api_key[:8]}...{api_key[-8:]}")
    
    # 4. 构建请求
    request_id = str(uuid.uuid4())
    headers = build_headers(api_key, request_id)
    body = build_request_body(audio_data)
    
    print(f"\n[3] 请求配置:")
    print(f"    URL: {API_URL}")
    print(f"    Request-ID: {request_id}")
    print(f"    请求体大小: {len(json.dumps(body))} 字节")
    print(f"    音频 base64 大小: {len(body['audio']['data'])} 字节")
    
    # 5. 发送请求
    print(f"\n[4] 发送请求...")
    try:
        response = requests.post(API_URL, headers=headers, json=body, timeout=60)
    except requests.exceptions.Timeout:
        print("[错误] 请求超时 (60s)")
        return
    except requests.exceptions.ConnectionError as e:
        print(f"[错误] 连接失败: {e}")
        return
    except Exception as e:
        print(f"[错误] 请求异常: {e}")
        return
    
    # 6. 处理响应
    print(f"\n[5] 响应状态码: {response.status_code}")
    
    if response.status_code != 200:
        print(f"[错误] HTTP {response.status_code}")
        print(f"响应内容: {response.text[:500]}")
        return
    
    try:
        response_json = response.json()
    except json.JSONDecodeError as e:
        print(f"[错误] 解析 JSON 失败: {e}")
        print(f"响应内容: {response.text[:500]}")
        return
    
    # 打印完整响应 (隐藏敏感 key)
    print(f"\n[6] 完整响应:")
    safe_response = {k: v for k, v in response_json.items() if "key" not in k.lower()}
    print(json.dumps(safe_response, ensure_ascii=False, indent=2))
    
    # 7. 提取识别结果
    print(f"\n[7] 识别结果:")
    result_text = parse_response(response_json)
    print(f"    {result_text}")
    
    # 8. 完成
    print(f"\n{'=' * 60}")
    if result_text.startswith("["):
        print("识别失败")
    else:
        print("识别成功")
    print("=" * 60)


if __name__ == "__main__":
    main()
