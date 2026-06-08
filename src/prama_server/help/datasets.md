# 数据集格式说明

Prama 在线评估当前支持 ASR、VAD、LID 三类测试。每类任务都必须提供包含 `audio` 字段的数据集，音频会按评估表单里的采样率读取。

## 通用要求

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

ASR 用于语音识别评估。每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `text`: 参考转写文本。
- `id` 或 `utt_id`: 可选样本 ID。

`audiofolder` 的 `metadata.jsonl` 示例：

```jsonl
{"file_name":"sample_001.wav","id":"asr_001","text":"hello world"}
{"file_name":"sample_002.wav","id":"asr_002","text":"this is a test"}
```

## VAD 数据集

VAD 用于语音活动检测评估。每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `seconds`: 参考语音段列表，单位为秒。
- `id` 或 `utt_id`: 可选样本 ID。

`seconds` 是二维数组，每个元素为 `[start, end]`。

```jsonl
{"file_name":"vad_001.wav","id":"vad_001","seconds":[[0.32,1.46],[2.10,3.80]]}
{"file_name":"vad_002.wav","id":"vad_002","seconds":[]}
```

## LID 数据集

LID 用于语种识别评估。每条样本必须包含：

- `audio`: 音频文件或可解码音频对象。
- `language_id`: 真实语种标签。
- `id` 或 `utt_id`: 可选样本 ID。

LID 按模型输出的原始字符串严格比较，不会隐藏规整化标签。

```jsonl
{"file_name":"lid_en_001.wav","id":"lid_en_001","language_id":"en"}
{"file_name":"lid_zh_001.wav","id":"lid_zh_001","language_id":"zh"}
```

## MOS/SNR 语音质量评估

MOS 和 SNR 是可选附加评估，不改变 ASR、VAD、LID 的数据集格式。启用后，系统会对每条测试音频调用对应的质量评估引擎，并在样本行和汇总区显示分数。

```json
{
  "enable_mos": true,
  "mos_target": "192.168.0.213:50111",
  "enable_snr": true,
  "snr_target": "192.168.0.213:50112"
}
```
