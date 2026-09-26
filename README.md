# Prama Server

Prama Server 是面向语音服务的评估工具，支持 ASR、关键词、VAD、LID 和 SE 任务，提供 HTTP/Web 界面与离线 CLI。

## 安装与运行

项目的开发依赖为 Node.js、pnpm 和 Poetry。首次安装依赖：

```bash
poetry install
pnpm --dir src/web install
```

可选的服务与前端开发配置见 [`.env.example`](.env.example)。复制为 `.env` 后，可在启动
Python 服务前执行 `set -a; . ./.env; set +a` 加载变量；Vite 会自动读取其中的
`VITE_API_PROXY_TARGET`。

标准运行前先构建前端，随后由 FastAPI 统一托管 `src/web/dist` 和 API：

```bash
pnpm --dir src/web run build
poetry run prama-server serve-http --host 0.0.0.0 --port 8000 --reload
```

浏览器访问 `http://localhost:8000`。Web 前端的“帮助”页包含各任务的数据集格式、指标公式和边界定义。

本地开发前端时，可另开终端运行：

```bash
cd src/web
pnpm install --frozen-lockfile
pnpm dev
```

Vite 开发服务器会把 `/api` 请求代理到本机的 `8000` 端口。VS Code 启动 HTTP 后端调试前会自动执行一次前端构建。

前端类型检查、构建与浏览器回归测试：

```bash
pnpm --dir src/web run lint
pnpm --dir src/web run build
pnpm --dir src/web run test:e2e
```

浏览器测试优先使用 `PLAYWRIGHT_CHROMIUM_PATH` 指定的 Chromium，未指定时会探测系统 Chromium，再回退到 Playwright 自带浏览器。

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

## 本地 prama wheel 与逐条指标

当前服务代码使用 prama 0.2.0a1 的 `iter_wer` / `iter_cer`。本地联调通过 pip
覆盖 Poetry 环境中的包，不修改本项目 `pyproject.toml` 或 `poetry.lock`：

```bash
(cd /workspace/libs/prama && poetry build)
poetry run python -m pip install --no-deps --force-reinstall \
  /workspace/libs/prama/dist/prama-0.2.0a1-cp310-cp310-linux_x86_64.whl
poetry run python -c 'from importlib.metadata import version; print(version("prama"))'
```

该 wheel 适用于当前 Linux x86_64 / Python 3.10 环境。重新执行 `poetry install`
或同步锁定依赖可能恢复旧版本，此时需要重新安装 wheel。已经启动的服务需重启才能加载新包。

ASR 每条最终识别返回后立即计算 WER/CER，通过
`GET /api/evaluations/{job_id}/events` 发送两个 `metric_result` SSE 事件：

- `metric`：`wer` 或 `cer`。
- `utterance`：当前样本的标识、逐 token 对齐和本条计数。
- `summary`：到当前样本的累计计数；`wer` / `accuracy` 为百分数。

前端在任务运行期间更新概览和对齐报告，不必等全部样本完成。重连时用
`metric_snapshot` 恢复已有报告；`done` 保留完整最终结果格式。ASR 临时转写仍走
`partial_inference_result`，不计入最终指标。并发任务使用线程；流式 native 对象通过
上下文管理器释放。最后仍执行完整批量汇总，逐条计数和对齐以实际 sclite 结果为准。

验证命令：

```bash
poetry run python -m unittest tests.test_asr_metric_stream tests.test_http_recalculate tests.test_http_frontend tests.test_session tests.test_prama_cli tests.test_vad_evaluator -q
pnpm --dir src/web run lint
pnpm --dir src/web run build
pnpm --dir src/web exec playwright test --grep 'ASR 运行中'
```

流式测试使用真实 wheel 的 C 实现，覆盖 8 线程、空文本、中英文、插入/删除/替换和
与批量结果的严格比较。SSE 测试仅替代外部 ASR 识别服务，验证第二条识别开始前即可
读取第一条指标；浏览器测试验证运行中展示对齐结果及重复事件去重。
