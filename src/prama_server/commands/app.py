from __future__ import annotations

import logging

import typer
import uvicorn

from prama_server.prama_cli import app as eval_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer()
app.add_typer(eval_app, name="eval")


@app.command()
def serve_http(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="HTTP 服务监听地址",
        envvar="PRAMA_SERVER_HTTP_HOST",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="HTTP 服务监听端口",
        envvar="PRAMA_SERVER_HTTP_PORT",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="启用 Uvicorn 自动重载",
        envvar="PRAMA_SERVER_HTTP_RELOAD",
    ),
) -> None:
    logger.info("启动 HTTP 评估服务: host=%s port=%s reload=%s", host, port, reload)
    uvicorn.run(
        "prama_server.servicer.http:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
