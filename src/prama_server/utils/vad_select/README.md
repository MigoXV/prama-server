# VAD 结果筛选数据集工具

该工具把 VAD 评估结果中的逐样本指标映射回原始 audiofolder 数据集，
按指标闭区间筛选样本，并生成一个可独立加载的新数据集。

## 输入与输出

- `--result-json`：前端下载的 VAD 结果，或后端任务快照。
- `--dataset-path`：原始 VAD audiofolder 根目录。
- `--split`：默认读取 `<dataset>/test/metadata.jsonl`，找不到时读取
  `<dataset>/metadata.jsonl`。
- 原始 metadata 的 `id` 必须唯一，并与结果中的样本 `id` 一一对应。

输出结构：

```text
<output>/<split>/
  metadata.jsonl
  selection_summary.json
  audio/
    *.wav
```

音频会复制到输出目录。启用 `--overwrite` 时，工具先在临时目录完成全部
校验和复制，成功后才替换旧目录。

## 使用示例

```bash
poetry run python -m prama_server.utils.vad_select.app \
  --result-json outputs/vad-result.json \
  --dataset-path data-bin/audiofolder/vad-demo \
  --split test \
  --output data-bin/audiofolder/vad-selected \
  --min-frame-recall 0.9 \
  --min-segment-f1 0.8 \
  --max-frame-false-alarm-rate 0.2 \
  --overwrite
```

筛选的是每条音频的指标，所有指标都使用 `0..1` 比例而非百分数字符串。
同时启用多个范围时，样本必须满足全部条件。默认所有范围均为 `None`，
等价于复制结果中包含的全部样本。
