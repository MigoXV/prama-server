FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/node:24-alpine-pnpm-builder AS frontend-build

WORKDIR /app/src/web

COPY src/web/package.json src/web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY src/web/ ./
RUN pnpm build

# 基于已经包含 Poetry 的基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/pytorch:2.8.0-amd64 AS runtime

# 设置工作目录
WORKDIR /app

# 安装 Linux 系统级依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 安装依赖
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    /app/.venv/bin/pip install --no-cache-dir datasets torchcodec==0.7.0

# 拷贝必要的文件以安装依赖
COPY pyproject.toml poetry.lock README.md ./
RUN mkdir -p src/prama_server && \
    touch src/prama_server/__init__.py && \
    poetry install --no-root

# 拷贝源代码文件
COPY . .

# 拷贝前端构建产物，运行时不再依赖 Node.js 或 Nginx
COPY --from=frontend-build /app/src/web/dist /app/web-dist
ENV PRAMA_SERVER_WEB_DIST=/app/web-dist

# 安装当前包
RUN poetry install

# 暴露 HTTP 服务端口
EXPOSE 8000

# 默认入口
CMD ["poetry", "run", "prama-server", "serve-http"]
