FROM python:3.11-slim
WORKDIR /app

# 清华pip加速，国内必加，否则下载很慢
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --progress=on --timeout 1200 -r requirements.txt


COPY . .
RUN mkdir -p data

EXPOSE 8000
ENV USE_LOCAL_RERANK=false
CMD ["python", "app.py"]
