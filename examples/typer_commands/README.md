# Typer 命令示例

这个目录演示项目里的 Typer 命令入口，便于查看参数、复制命令，或在 Web 界面不可用时运行离线评估。

- `show_help.sh`：只打印命令帮助，不连接评估引擎，也不写输出文件。
- `run_examples.sh`：默认只打印命令；设置 `RUN_LIVE=1` 后才实际执行。

在项目根目录运行：

```bash
bash examples/typer_commands/show_help.sh
bash examples/typer_commands/run_examples.sh
RUN_LIVE=1 bash examples/typer_commands/run_examples.sh
```

常用环境变量：

```bash
ASR_TARGET=192.168.0.222:50011
VAD_TARGET=192.168.0.222:50021
LID_TARGET=192.168.0.222:50026
DENOISE_TARGET=192.168.0.222:50031
MOS_TARGET=192.168.0.213:50111
SNR_TARGET=192.168.0.213:50112
OUTPUT_DIR=outputs/typer-command-examples
```

SE 评估必须至少配置 `MOS_TARGET` 或 `SNR_TARGET`。默认输出目录是 `outputs/typer-command-examples/`。
