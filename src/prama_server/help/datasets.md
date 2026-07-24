# 数据集与评估指标说明

Prama 在线评估当前支持 ASR、关键词、VAD、LID、SE 五类测试。每类任务都必须提供包含 `audio` 字段的数据集，音频会按评估表单里的采样率读取。

## 目录

- [通用数据集要求](#通用数据集要求)
- [ASR 数据集](#asr-数据集)
- [关键词数据集](#关键词数据集)
- [VAD 数据集](#vad-数据集)
- [LID 数据集](#lid-数据集)
- [SE 评估数据集](#se-评估数据集)
- [评估指标定义](#评估指标定义)

## 通用数据集要求

- 数据集路径可以是 Hugging Face dataset 路径、`load_from_disk` 目录，或 `audiofolder` 目录。
- `audio` 字段必须能被 `datasets.Audio` 解码。
- 建议所有测试音频使用单声道 16 kHz WAV。
- `id` 或 `utt_id` 可选；未提供时系统会使用样本序号作为 ID。
- 上传目录中不要混入无关音频，避免被 `audiofolder` 自动纳入测试集。

```text
dataset/
  test/
    sample_001.wav
    sample_002.wav
    metadata.jsonl
```

## ASR 数据集

ASR 评估每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `text`: 参考转写文本。
- `id` 或 `utt_id`: 可选样本 ID。

`audiofolder` 的 `metadata.jsonl` 示例：

```jsonl
{"file_name":"sample_001.wav","id":"asr_001","text":"hello world"}
{"file_name":"sample_002.wav","id":"asr_002","text":"this is a test"}
```

## 关键词数据集

关键词评估复用 ASR 引擎：系统先对音频做一次 ASR，再从 ASR 输出文本里查找该音频对应的全部关键词。推荐一条音频记录使用 `keywords` 数组。

每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `keywords`: 非空数组；每项包含 `keyword` 和布尔字段 `expected_hit`。
- `id` 或 `utt_id`: 可选样本 ID。

`audiofolder` 的 `metadata.jsonl` 示例：

```jsonl
{"file_name":"keyword_demo_01.wav","id":"keyword_001","keywords":[{"keyword":"AUSTRIAN","expected_hit":true},{"keyword":"SPEED","expected_hit":false}]}
{"file_name":"keyword_demo_02.wav","id":"keyword_002","keywords":[{"keyword":"LEVEL","expected_hit":true}]}
```

旧格式的顶层 `keyword`、`expected_hit` 仍受支持。同一 `file_name` 的旧格式多行记录会合并为一条音频，只推理一次。`limit` 限制音频条数，每条入选音频的所有关键词都会参与统计。

匹配时会把 ASR 文本和关键词统一转小写、去标点并压缩空白。英文和数字关键词按词边界匹配，避免 `CAT` 误命中 `CATCH`；中文等非空白语言文本按规范化后的子串匹配。

## VAD 数据集

VAD 评估每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `seconds`: 参考语音段列表，单位为秒。
- `id` 或 `utt_id`: 可选样本 ID。

`seconds` 包含等长的 `starts` 和 `durations` 数组，单位均为秒。

```jsonl
{"file_name":"vad_001.wav","id":"vad_001","seconds":{"starts":[0.32,2.10],"durations":[1.14,1.70]}}
{"file_name":"vad_002.wav","id":"vad_002","seconds":{"starts":[],"durations":[]}}
```

## LID 数据集

LID 评估每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `language_id`: 真实语种标签。
- `id` 或 `utt_id`: 可选样本 ID。

LID 是开放集语种识别任务。真实标签为 `<others>` 的样本表示未知语种集合；其他真实标签都视为已知语种。系统按模型输出的原始字符串严格比较，不会把 `cn`、`zh`、`zh-CN` 等标签自动规整成同一类。

```jsonl
{"file_name":"lid_en_001.wav","id":"lid_en_001","language_id":"en"}
{"file_name":"lid_cn_001.wav","id":"lid_cn_001","language_id":"cn"}
{"file_name":"lid_unknown_001.wav","id":"lid_unknown_001","language_id":"<others>"}
```

## SE 评估数据集

SE 评估用于比较原始音频和 SE 后音频的 SNR、MOS 分数变化。每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `id` 或 `utt_id`: 可选样本 ID。

`audiofolder` 的 `metadata.jsonl` 只需要 `file_name` 和 `id`：

```jsonl
{"file_name":"denoise_001.wav","id":"denoise_001"}
{"file_name":"denoise_002.wav","id":"denoise_002"}
```

SE 评估必须至少启用 MOS 或 SNR 中的一个质量评估引擎；允许只配置其中一个。`target` 填写 SE 引擎地址，例如：

```json
{
  "task": "denoise",
  "target": "192.168.0.222:50027",
  "dataset_path": "data-bin/audiofolder/denoise-demo",
  "enable_mos": true,
  "mos_target": "192.168.0.213:50111"
}
```

MOS 和 SNR 也可以作为 ASR、VAD、LID 的可选附加评估，不改变这些任务的数据集格式。启用后，系统会对每条测试音频调用对应的质量评估引擎，并在样本行和汇总区显示分数。

## 评估指标定义

### ASR 指标

ASR 使用参考文本 `Reference` 与识别文本 `Hypothesis` 对齐后计算 WER 和 CER。WER 以词为单位，CER 以字符为单位。

| 指标 | 含义 | 分母 |
| --- | --- | --- |
| WER | 词错误率，百分制 | 参考文本词数 |
| CER | 字符错误率，百分制 | 参考文本字符数 |
| 词正确率 | `(C-I)/N = 1-WER`，百分制 | 参考文本词数 |
| 字正确率 | `(C-I)/N = 1-CER`，百分制 | 参考文本字符数 |

$$
\mathrm{WER} = \frac{S_{\mathrm{word}} + D_{\mathrm{word}} + I_{\mathrm{word}}}{N_{\mathrm{word}}}
$$

$$
\mathrm{CER} = \frac{S_{\mathrm{char}} + D_{\mathrm{char}} + I_{\mathrm{char}}}{N_{\mathrm{char}}}
$$

其中 `S` 为替换数，`D` 为删除数，`I` 为插入数，`N` 为参考文本单元数。插入错误计入分子，但分母仍只使用参考文本单元数。

正确率沿用 SCLITE 定义：

$$
\mathrm{Accuracy} = \frac{C-I}{N} = 1-\mathrm{ErrorRate}
$$

API 和界面中的 WER、CER、词正确率、字正确率均使用百分制，例如 `12.5` 表示 `12.5%`。插入错误数大于正确数时，正确率可以为负值；这是指标定义的正常结果，不会截断为 0。顶层 `accuracy` 是 `word_accuracy` 的兼容别名。

### 关键词指标

关键词评估按二分类计算。`expected_hit=true` 表示正样本，`expected_hit=false` 表示负样本；`predicted_hit` 来自 ASR 文本中的关键词查找结果。

| 指标 | 定义 |
| --- | --- |
| Accuracy | 命中样本和正确拒识样本占总样本的比例 |
| Precision | 预测命中样本中真实命中的比例 |
| Recall | 真实命中样本中被预测命中的比例 |
| F1 | Precision 和 Recall 的调和平均 |
| Miss | 真实应命中但未命中的样本数 |
| False Alarm | 真实不应命中但被预测命中的样本数 |

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
$$

$$
\mathrm{Recall} = \frac{TP}{TP + FN}
$$

$$
\mathrm{F1} = \frac{2 \times \mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

### VAD 指标

VAD 会把参考语音段和预测语音段转换为固定帧长的布尔 mask，再计算帧级指标；段级指标按语音段命中情况计算。顶层字段以及 `frame`、`segment` 是把所有样本的计数先相加再计算的 Micro 指标；`frame_macro`、`segment_macro` 是各样本比率的算术平均。

| 指标 | 定义 |
| --- | --- |
| Frame Accuracy | 所有帧中预测正确的比例 |
| Frame Recall | 真实语音帧中被预测为语音的比例 |
| Frame Precision | 预测语音帧中真实为语音的比例 |
| Segment Recall | 真实语音段中被预测段命中的比例 |
| Segment Precision | 预测语音段中命中真实段的比例 |

$$
\mathrm{FrameAccuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\mathrm{FrameRecall} = \frac{TP}{TP + FN}
$$

$$
\mathrm{FramePrecision} = \frac{TP}{TP + FP}
$$

$$
\mathrm{FrameF1} = \frac{2 \times \mathrm{FramePrecision} \times \mathrm{FrameRecall}}{\mathrm{FramePrecision} + \mathrm{FrameRecall}}
$$

段级指标有意沿用现有评估器的非对称定义：

- 段召回率以真实语音段为分母。对每个真实段，预测覆盖占该真实段的比例达到 `hit_threshold` 才算命中。
- 段精确率以预测语音段为分母。预测段只要与任一真实段存在正长度重叠就算命中，否则计为误报段；这里不应用 `hit_threshold`。

聚合时 Micro F1 由 Micro Precision 和 Micro Recall 重新计算；Macro F1 是逐样本 F1 的算术平均。任何分母为 0 的比率都定义为 0。

### LID 指标

LID 按开放集识别计算。真实标签为 `<others>` 的样本是未知语种干扰集；其他真实标签都是有效已知语种。预测置信度低于 `lid_confidence_threshold` 时，不管模型原始输出是什么，有效预测都记为 `<others>`。

| 指标 | 定义 | 分母 |
| --- | --- | --- |
| 已知语种准确率 | 真实标签不是 `<others>` 且预测正确的样本比例 | 真实已知语种样本数 |
| 类别精确率 | 预测为某个已知语种的样本中预测正确的比例 | 预测为该类别的样本数 |
| 类别召回率 | 某个有效已知语种真实标签下预测正确的样本比例；`<others>` 不计算召回率 | 该有效类别真实样本数 |
| 宏平均精确率 | 对真实标签中出现的已知类别精确率取算术平均 | 真实已知语种类别数 |
| 宏平均召回率 | 仅对真实标签不是 `<others>` 的有效类别召回率取算术平均 | 真实已知语种类别数 |
| 未知误接收 | 真实 `<others>` 干扰样本被预测为已知语种的样本数 | 不做比例化 |
| 已知被拒识 | 真实已知语种被预测为 `<others>` 的样本数 | 不做比例化 |

$$
\mathrm{Accuracy}_{\mathrm{known}} = \frac{\mathrm{Correct}_{\mathrm{known}}}{\mathrm{Total}_{\mathrm{known}}}
$$

$$
\mathrm{Recall}_{\mathrm{label}} = \frac{\mathrm{Correct}_{\mathrm{label}}}{\mathrm{Total}_{\mathrm{label}}}
$$

$$
\mathrm{Precision}_{\mathrm{label}} = \frac{\mathrm{Correct}_{\mathrm{label}}}{\mathrm{Predicted}_{\mathrm{label}}}
$$

$$
\mathrm{Precision}_{\mathrm{macro}} = \frac{\sum_{\ell \in \mathcal{K}} \mathrm{Precision}_{\ell}}{|\mathcal{K}|}
$$

$$
\mathrm{Recall}_{\mathrm{macro}} = \frac{\sum_{\ell \in \mathcal{K}} \mathrm{Recall}_{\ell}}{|\mathcal{K}|}, \quad \mathcal{K}=\{\ell \mid \ell \ne \langle others \rangle\}
$$

已知类别集合 $\mathcal{K}$ 由真实标签中出现且不等于 `<others>` 的类别构成。`<others>` 不计算类别精确率或召回率，也不进入宏平均。它仍保留在混淆矩阵中：`<others>` 行表示干扰样本是否被正确拒识；`<others>` 列表示已知语种是否被错误拒识。类别没有预测样本时，其精确率按 0 计算。

### SE 指标

SE 评估会对原始音频和增强后音频分别计算可用的质量分数，并展示增强前、增强后和差值。

| 指标 | 含义 |
| --- | --- |
| MOS Before | 原始音频 MOS 分数 |
| MOS After | SE 后音频 MOS 分数 |
| MOS Δ | `MOS After - MOS Before` |
| SNR Before | 原始音频 SNR 分数 |
| SNR After | SE 后音频 SNR 分数 |
| SNR Δ | `SNR After - SNR Before` |

$$
\Delta_{\mathrm{MOS}} = \mathrm{MOS}_{\mathrm{after}} - \mathrm{MOS}_{\mathrm{before}}
$$

$$
\Delta_{\mathrm{SNR}} = \mathrm{SNR}_{\mathrm{after}} - \mathrm{SNR}_{\mathrm{before}}
$$

如果某条样本的 MOS 或 SNR 引擎没有返回有效数值，该样本不会进入对应分数的有效样本统计；另一个已启用且成功返回的指标仍会保留。
