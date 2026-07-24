# Prama Server

Prama Server 是面向语音服务的评估工具，支持 ASR、关键词、VAD、LID 和 SE 任务，提供 HTTP/Web 界面与离线 CLI。

## 安装与运行

项目使用 Poetry 管理依赖：

```bash
poetry install
poetry run prama-server serve-http --host 0.0.0.0 --port 8000 --reload
```

浏览器访问 `http://localhost:8000`。Web 前端的“帮助”页包含各任务的数据集格式、指标公式和边界定义。

本地开发前端时，可另开终端运行：

```bash
cd src/web
pnpm install --frozen-lockfile
pnpm dev
```

Vite 开发服务器会把 `/api` 请求代理到本机的 `8000` 端口。

## Docker 部署

项目使用多阶段镜像构建：前端由
`registry.cn-hangzhou.aliyuncs.com/migo-dl/node:24-alpine-pnpm-builder`
编译，产物复制到 Python 运行时镜像，并由 FastAPI 与后端接口统一托管。最终镜像不包含 Node.js 或 Nginx。

构建统一镜像：

```bash
docker build \
  -t registry.cn-hangzhou.aliyuncs.com/migo-dl/prama-server:0.8.0a2 \
  .
```

启动容器：

```bash
docker run --rm \
  -p 8000:8000 \
  -e PRAMA_WORKDIR=/data-bin \
  -v "$PWD/data-bin:/data-bin" \
  registry.cn-hangzhou.aliyuncs.com/migo-dl/prama-server:0.8.0a2
```

浏览器访问 `http://localhost:8000`，健康检查接口为
`GET http://localhost:8000/api/health`。数据目录默认挂载到
`./data-bin`，可通过 `PRAMA_WORKDIR` 调整容器内的工作目录。

## 离线评估

CLI 会显示逐样本进度，并把逐样本结果写入 TSV：

```bash
poetry run prama-server eval asr --dataset-path data-bin/audiofolder/asr-demo
poetry run prama-server eval vad --dataset-path data-bin/audiofolder/vad-demo
poetry run prama-server eval --help
```

## VAD 数据工具

把 JSONL 数据或扁平 WAV/CSV 目录切分并转换为 audiofolder：

```bash
poetry run python -m prama_server.utils.trim.app \
  --dataset-path data-bin/raw-vad \
  --chunk-seconds 30 \
  --output data-bin/audiofolder/vad
```

根据评估 JSON 中的逐样本标准指标筛选新数据集：

```bash
poetry run python -m prama_server.utils.vad_select.app \
  --result-json outputs/vad-result.json \
  --dataset-path data-bin/audiofolder/vad \
  --output data-bin/audiofolder/vad-selected \
  --min-frame-recall 0.8 \
  --max-segment-false-alarm-rate 0.2
```

两个工具都先写入同级临时目录，成功后再替换目标目录；覆盖已有目标必须显式传入 `--overwrite`。
