# Prama 语音评估工作空间

前端采用 React、TypeScript 和 Vite。界面沿用 Ninna 的暖白与深绿色配色，按 MANAS 类纸工作空间组织创建、运行、报告和样本诊断。

## 开发与验证

在仓库根目录运行：

```bash
corepack pnpm@10.33.0 --dir src/web install --frozen-lockfile
corepack pnpm@10.33.0 --dir src/web run dev
corepack pnpm@10.33.0 --dir src/web run build
corepack pnpm@10.33.0 --dir src/web run test:e2e
```

端到端测试使用 Chromium，并通过 `poetry run` 启动 Python HTTP 服务；评估任务由测试数据模拟，不连接真实推理引擎。测试浏览器默认路径见 `playwright.config.ts`。生产构建输出到 `src/web/dist`，由现有 HTTP 服务托管。

## 页面与状态

| 地址 | 用途 |
| --- | --- |
| `/evaluations/new?task=asr` | 统一创建入口，支持 ASR、VAD、LID、Keyword、SE |
| `/evaluations` | 本浏览器记录的每类最近任务 |
| `/evaluations/:jobId?view=run` | 运行状态与进度 |
| `/evaluations/:jobId?view=report` | 汇总报告 |
| `/evaluations/:jobId?view=diagnosis` | 样本搜索、筛选与诊断 |
| `/evaluations/:jobId?view=configuration` | 本次实际请求参数 |
| `/settings` | 保存以后新建或主动重置草稿使用的默认值 |
| `/help` | 服务端帮助文档 |

VAD 的 `filter`、`q` 和 `sample` 参数保存在 URL 中；其他任务的搜索、筛选、排序或视图参数也通过 URL 恢复。浏览器前进、后退和刷新保留当前页面。

- 各任务草稿独立保存。保存默认值不会覆盖已经保存的草稿。
- 重新评估沿用原任务的引擎、数据集、split 和 limit，并使用该任务当前草稿中的高级参数；提交前展示与原请求的差异。
- SSE 订阅按任务 ID 隔离；后台任务完成不会强行切换当前页面。断线保留最后收到的数据并提示重连。
- 当前任务列表只记录当前浏览器每类最近一次任务，最多五类。后端没有历史列表、持久化恢复、取消任务或基线比较接口，界面不提供这些能力。
- 报告筛选只改变样本列表，汇总指标保持完整评估口径。未返回的数值显示缺失标记。

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `src/App.tsx` | 页面编排、创建、导入、默认设置、重新评估 |
| `src/components/AppShell.tsx` | 统一导航、桌面收起、移动端导航与焦点返回 |
| `src/hooks/useRoute.ts` | History API 路由和查询参数 |
| `src/features/evaluation/model.ts` | 表单默认值、任务差异、迁移、请求转换 |
| `src/features/evaluation/useEvaluationJobs.ts` | 任务快照、SSE 与浏览器最近任务记录 |
| `src/features/evaluation/Configuration.tsx` | 高级参数、连接测试和请求摘要 |
| `src/features/evaluation/ReportWorkspace.tsx` | 五类报告的统一入口、搜索和筛选 |
| `src/components/Reports.tsx` | 原有领域报告、对齐、混淆矩阵和音频控件 |
| `src/components/VadEvaluationReport.tsx` | VAD 指标、样本与时间轴诊断 |
| `src/styles/workspace.scss` | Ninna 色彩变量、共享布局与响应式样式 |

`services/evaluations.ts` 和后端 HTTP/SSE 契约保持兼容。字体由本地构建资源提供：Inter Variable、Noto Sans SC Variable、IBM Plex Mono。

## 设计稿与验收材料

设计稿同步到[原 Figma 文件](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=37-2)，原有 VAD 画板 ID 保留。

| 代码界面 | Figma 入口 |
| --- | --- |
| 颜色、字体、间距 | [基础规范](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=2-69) |
| 共享控件、导航、输入框 | [组件](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=4-2) |
| 创建、任务状态、移动导航 | [共享流程](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=37-2) |
| VAD 报告与样本诊断 | [VAD](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=4-77) |
| ASR | [ASR](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=39-1200) |
| LID | [LID](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=39-4869) |
| Keyword | [Keyword](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=39-6091) |
| SE | [SE](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=39-7286) |
| 设置、帮助、目录弹窗、重新评估 | [辅助页面](https://www.figma.com/design/xVHy1aog1ucKJ9If5eA95N/Prama?node-id=39-8241) |

Figma 使用可编辑文字、矢量、组件实例和布局分组。断点分别交付画板；这些画板不是可执行的浏览器原型。浏览器原生音频条以可编辑控件表达。当前没有发布 Code Connect，代码与设计的对应关系以上表为准。

本地验收材料位于 `data-bin/tmp/2026-09-26`：

- `index.html`、`REVIEW.md`：旧版全界面截图与问题分析。
- `redesign/index.html`：新版截图画廊，逐张标明真实页面或模拟评估数据。
- `redesign/README.md`：本次改造、测试结果与范围说明。
- `redesign/figma-index.json`：画板 ID、页面和组件索引。

截图和模拟报告只验证界面结构与交互，不代表真实模型效果、推理性能或音频服务可用性。
