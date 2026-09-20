FROM python:3.11-slim
WORKDIR /app

# 构建参数：默认 requirements.txt（本地/阿里云有 GPU），生产传 requirements-prod.txt（无 GPU，镜像更小）
ARG REQ_FILE=requirements.txt

# 阿里云服务器访问 Debian 官方源慢/不通，换阿里云镜像
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

# libgomp1：CUDA 版 torch 的 OpenMP 运行库依赖，python:slim 默认不含，缺失会导致 import torch 失败
# requirements-prod.txt 不含 torch，但装 libgomp1 无害（~2MB），保持 Dockerfile 统一
# ffmpeg：飞书语音 opus 转 wav（ASR 模块需要）
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 清华 pip 加速，国内必加，否则下载很慢
# torch==2.11.0+cu128 由 requirements.txt 内 --extra-index-url（上海交大 pytorch 镜像）提供
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY ${REQ_FILE} .
RUN pip install --no-cache-dir --progress=on --timeout 1200 -r ${REQ_FILE}


COPY . .
RUN mkdir -p data

EXPOSE 8000
# 镜像默认 false：无 GPU 的部署走 DashScope API rerank，省运行时开销。
# 本地 Docker Desktop 用 GPU 本地 rerank 时，由 docker-compose.yml 的 environment 覆盖为 true。
ENV USE_LOCAL_RERANK=false
CMD ["python", "app.py"]
