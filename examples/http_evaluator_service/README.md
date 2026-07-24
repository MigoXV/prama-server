# HTTP 评估服务前端说明

HTTP 评估服务已经作为主程序的一部分放在 `prama_server.servicer.http`。服务自带 Web 前端，用于提交 ASR、关键词、VAD、LID 和 SE 评估任务。

## 运行

在项目根目录执行：

```bash
poetry install
poetry run prama-server serve-http --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：

```text
http://localhost:8000
```

生产镜像会先使用专用 Node.js 构建镜像编译前端，再把静态文件复制到
Python 运行时中，由 FastAPI 统一提供页面和 `/api` 接口。容器只需暴露
`8000` 端口：

```bash
docker run --rm \
  -p 8000:8000 \
  -e PRAMA_WORKDIR=/data-bin \
  -v "$PWD/data-bin:/data-bin" \
  registry.cn-hangzhou.aliyuncs.com/migo-dl/prama-server:0.8.0a2
```

离线评估和 TSV 导出可以使用：

```bash
poetry run prama-server eval --help
```

## 接口

- `GET /api/health`：检查 HTTP 服务是否可用。
- `POST /api/evaluations`：创建评估任务。
- `GET /api/evaluations/{job_id}`：查看任务快照。
- `GET /api/evaluations/{job_id}/events`：通过 SSE 订阅任务进度。

默认参数沿用 `tests/demo01.py`：

- ASR gRPC 地址：`192.168.1.24:50008`
- 数据集路径：`data-bin/jacktol/ATC-ASR-Dataset`
- split：`test`
- 采样率：`16000`
- 热词：`HOTEL`

## 注意事项

这个示例使用进程内内存保存任务状态，适合本地开发和演示。请使用单 worker 运行，不要把它当作生产队列或持久化任务系统。
