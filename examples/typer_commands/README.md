# Typer 命令示例

这个目录演示当前项目里的 Typer 命令入口，适合现场同事快速查看参数、复制命令或在 UI 不可用时走命令行评估。

## 文件

- `show_help.sh`：只打印各个 Typer 命令的 `--help`，不连接任何评估引擎，也不写输出文件。
- `run_examples.sh`：展示 VAD 切片和 ASR/VAD/LID/关键词/SE 评估命令。默认 dry-run 只打印命令；设置 `RUN_LIVE=1` 后才实际执行。

## 查看命令帮助

在项目根目录执行：

```bash
bash examples/typer_commands/show_help.sh
```

也可以直接看某个命令：

```bash
poetry run python -m prama_server.commands.app eval lid --help
poetry run python -m prama_server.utils.trim_vad_data.app --help
```

## 运行示例命令

默认只打印将要执行的命令：

```bash
bash examples/typer_commands/run_examples.sh
```

确认目标服务地址和数据集路径后，再实际执行：

```bash
RUN_LIVE=1 bash examples/typer_commands/run_examples.sh
```

常用环境变量：

```bash
ASR_TARGET=192.168.0.222:50011
VAD_TARGET=192.168.0.222:50021
LID_TARGET=192.168.0.222:50026
DENOISE_TARGET=192.168.0.222:50027
MOS_TARGET=192.168.0.213:50111
SNR_TARGET=192.168.0.213:50112
OUTPUT_DIR=outputs/typer-command-examples
```

SE 评估必须至少提供 `MOS_TARGET` 或 `SNR_TARGET`。如果两者都为空，`run_examples.sh` 会跳过 SE 示例。

## 输出位置

`run_examples.sh` 默认输出到：

```text
outputs/typer-command-examples/
```

其中评估结果是 TSV 文件，VAD 切片示例输出到 `vad-trim-demo/`。
