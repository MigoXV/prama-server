# VAD 结果筛选数据集工具

这个模块用于把前端导出的 VAD 评估结果 JSON，重新映射回原始
VAD audiofolder 数据集，并按指标范围筛选样本，输出一个自包含、
可直接加载的新 audiofolder 数据集。

## 输入

- `--result-json`：前端“下载结果 JSON”得到的 VAD 结果文件，或包含
  `result.vad_report.samples` 的后端快照。
- `--dataset-path`：原始 VAD audiofolder 根目录。工具会读取
  `<dataset>/<split>/metadata.jsonl`，找不到时再读取
  `<dataset>/metadata.jsonl`。
- 原始 metadata 中的 `id` 必须唯一，并且要能和结果 JSON 里的样本
  `id` 一一对应。

## 输出

输出目录结构如下：

```text
<output>/<split>/
  metadata.jsonl
  selection_summary.json
  audio/
    *.wav
```

`metadata.jsonl` 会保留原始样本的 `id` 和 `seconds`，并把
`file_name` 改成输出目录内的 `audio/<name>.wav`。音频会复制到输出目录，
因此新数据集不依赖原始数据集继续存在。

## 使用示例

```bash
poetry run python -m prama_server.utils.vad_select.app \
  --result-json tmp-workspace/99b1845d80c74c25a66509cb490f530c-result.json \
  --dataset-path data-bin/db-vad-test-01 \
  --split test \
  --output data-bin/db-vad-test-01-selected \
  --overwrite
```

输出后可以直接加载：

```python
from datasets import load_dataset

dataset = load_dataset(
    "audiofolder",
    data_dir="data-bin/db-vad-test-01-selected",
    split="test",
)
```

## 指标筛选

筛选范围有两种设置方式：

1. 修改 `core.py` 顶部的 `VAD_METRIC_RANGES`，作为文件默认值。
2. 运行命令时传入对应的 `--min-*` / `--max-*` 参数覆盖默认值。

默认所有范围都是 `None`，等效于不做筛选。支持的指标包括：

- `frame_recall`
- `frame_precision`
- `frame_f1`
- `segment_recall`
- `segment_precision`
- `segment_f1`
- `frame_false_alarm_rate`
- `segment_false_alarm_rate`

示例：

```bash
poetry run python -m prama_server.utils.vad_select.app \
  --result-json tmp-workspace/99b1845d80c74c25a66509cb490f530c-result.json \
  --dataset-path data-bin/db-vad-test-01 \
  --output data-bin/db-vad-test-01-selected \
  --min-frame-recall 0.9 \
  --min-segment-f1 0.8 \
  --max-frame-false-alarm-rate 0.2 \
  --overwrite
```

所有启用的指标范围都会同时生效，样本必须全部满足才会被保留。
