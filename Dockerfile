FROM python:3.11-slim

WORKDIR /app

# 安装依赖（生产环境不含 sentence-transformers）
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p data

EXPOSE 8000

# 生产环境默认配置
ENV USE_LOCAL_RERANK=false

CMD ["python", "app.py"]
