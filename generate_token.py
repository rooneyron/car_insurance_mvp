#!/usr/bin/env python
"""
生成带有效期的访问 Token（用于部署后限制访问）
"""

import jwt
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


# 从环境变量读取密钥，如果没有则使用默认值（生产环境务必修改）
SECRET = os.environ.get("ACCESS_TOKEN_SECRET", "your-secret-key-change-me-in-production")

# Token 有效期（7 天）
EXPIRY_DAYS = 7

def generate_token():
    """生成 JWT token，有效期 7 天"""
    payload = {
        "exp": int(time.time()) + EXPIRY_DAYS * 24 * 3600,
        "iat": int(time.time()),
        "purpose": "car_insurance_demo",
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    return token

if __name__ == "__main__":
    token = generate_token()
    expiry_date = (datetime.now() + timedelta(days=EXPIRY_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔗 访问链接（替换 your-domain.com 为实际部署地址）:")
    print(f"https://your-domain.com/?token={token}")
    print(f"\n⏱️ 有效期至: {expiry_date} (UTC+8)")
    print(f"\n📌 请将链接发给演示对象，7 天后自动失效。")