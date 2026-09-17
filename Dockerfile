FROM python:3.11-slim
WORKDIR /app

# libgomp1：CUDA 版 torch 的 OpenMP 运行库依赖，python:slim 默认不含，缺失会导致 import torch 失败
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 清华pip加速，国内必加，否则下载很慢
# torch==2.11.0+cu128 由 requirements.txt 内 --extra-index-url（上海交大 pytorch 镜像）提供
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir  --progress=on --timeout 1200 -r requirements.txt


COPY . .
RUN mkdir -p data

EXPOSE 8000
# 镜像默认 false：无 GPU 的部署走 DashScope API rerank，省运行时开销。
# 本地 Docker Desktop 用 GPU 本地 rerank 时，由 docker-compose.yml 的 environment 覆盖为 true。
ENV USE_LOCAL_RERANK=false
CMD ["python", "app.py"]
