import { expect, test } from "@playwright/test";

const completedSnapshot = {
  job_id: "job-manas-001",
  status: "completed",
  request: {
    task: "asr",
    target: "127.0.0.1:50011",
    dataset_path: "data-bin/audiofolder/asr-demo",
    split: "test",
    limit: null,
    language_code: "en-US",
    sample_rate: 16000,
    min_reference_words: 5,
    hotwords: [],
    hotword_bias: 0,
    connect_timeout_seconds: 10,
    request_timeout_seconds: 60,
    interim_results: true,
    inference_concurrency: 0,
    asr_inference_concurrency: 0,
    vad_inference_concurrency: 0,
    lid_inference_concurrency: 0,
    enable_mos: false,
    mos_target: "",
    enable_snr: false,
    snr_target: "",
    sqa_inference_concurrency: 0,
    lid_confidence_threshold: 0,
    remove_punctuation: false,
    mask_frame_seconds: 0.01,
    chunk_duration_seconds: 0.1,
    speech_padding_seconds: 0,
    hit_threshold: 0.9,
    streaming: false,
  },
  progress: { status: "completed", total: 1, processed: 1, evaluated: 1 },
  result: {
    wer: 0.25,
    cer: 0.12,
    word_accuracy: 0.75,
    character_accuracy: 0.88,
    sample_count: 1,
    wer_report: {
      summary: { wer: 0.25, hits: 3, substitutions: 1, deletions: 0, insertions: 0 },
      utterances: [
        {
          id: "sample-001",
          index: 1,
          summary: { wer: 0.25, hits: 3, substitutions: 1, deletions: 0, insertions: 0 },
          tokens: [
            { ref: "hello", hyp: "hello", label: "equal" },
            { ref: "world", hyp: "word", label: "substitution" },
          ],
        },
      ],
    },
    cer_report: {
      summary: { wer: 0.12, hits: 8, substitutions: 1, deletions: 0, insertions: 0 },
      utterances: [],
    },
  },
  error: null,
};

const completedVadSnapshot = {
  ...completedSnapshot,
  job_id: "job-vad-001",
  request: {
    ...completedSnapshot.request,
    task: "vad",
    target: "192.168.0.222:50021",
    dataset_path: "data-bin/audiofolder/vad-demo",
    min_reference_words: 0,
  },
  progress: { status: "completed", total: 3, processed: 3, evaluated: 3 },
  result: {
    sample_count: 3,
    included_sample_count: 3,
    audio_duration_seconds: 30,
    processing_elapsed_seconds: 0.5,
    realtime_factor: 60,
    frame: { frame_f1: 0.9 },
    segment: {
      reference_segment_count: 3,
      prediction_segment_count: 4,
      segment_miss_count: 1,
      segment_false_alarm_count: 1,
      segment_recall: 2 / 3,
      segment_precision: 3 / 4,
      segment_f1: 0.7059,
    },
    vad_report: {
      samples: [
        {
          id: "sample-miss",
          index: 1,
          duration_seconds: 10,
          frame_seconds: 0.01,
          metrics: {
            frame: { frame_f1: 0.8 },
            segment: { segment_miss_count: 1, segment_false_alarm_count: 0 },
          },
          reference_segments: [{ start: 1, end: 4, duration: 3, start_frame: 100, end_frame: 400, status: "miss" }],
          prediction_segments: [{ start: 1, end: 2, duration: 1, start_frame: 100, end_frame: 200, status: "hit" }],
          regions: [{ start: 2, end: 4, duration: 2, start_frame: 200, end_frame: 400, label: "miss" }],
        },
        {
          id: "sample-split",
          index: 2,
          duration_seconds: 10,
          frame_seconds: 0.01,
          metrics: {
            frame: { frame_f1: 0.86 },
            segment: { segment_miss_count: 0, segment_false_alarm_count: 1 },
          },
          reference_segments: [{ start: 2, end: 7, duration: 5, start_frame: 200, end_frame: 700, status: "hit" }],
          prediction_segments: [
            { start: 2, end: 4, duration: 2, start_frame: 200, end_frame: 400, status: "hit" },
            { start: 4.2, end: 7, duration: 2.8, start_frame: 420, end_frame: 700, status: "hit" },
            { start: 8, end: 9, duration: 1, start_frame: 800, end_frame: 900, status: "false_alarm" },
          ],
          regions: [{ start: 8, end: 9, duration: 1, start_frame: 800, end_frame: 900, label: "false_alarm" }],
        },
        {
          id: "sample-good",
          index: 3,
          duration_seconds: 10,
          frame_seconds: 0.01,
          metrics: {
            frame: { frame_f1: 0.99 },
            segment: { segment_miss_count: 0, segment_false_alarm_count: 0 },
          },
          reference_segments: [{ start: 3, end: 6, duration: 3, start_frame: 300, end_frame: 600, status: "hit" }],
          prediction_segments: [{ start: 3, end: 6, duration: 3, start_frame: 300, end_frame: 600, status: "hit" }],
          regions: [{ start: 3, end: 6, duration: 3, start_frame: 300, end_frame: 600, label: "hit" }],
        },
      ],
    },
  },
};

test("任务导航、视图标签与窄屏布局保持可操作", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "ASR 评估" })).toBeVisible();
  await page.getByRole("button", { name: "VAD", exact: true }).click();
  await expect(page.getByRole("heading", { name: "VAD 评估" })).toBeVisible();

  await expect(page.getByRole("heading", { name: "评估配置" })).toBeVisible();
  await expect(page.getByRole("tab")).toHaveCount(0);

  await page.getByRole("button", { name: "ASR", exact: true }).click();
  const overviewTab = page.getByRole("tab", { name: /运行概览/ });
  await overviewTab.focus();
  await overviewTab.press("ArrowRight");
  await expect(page.getByRole("tab", { name: /对齐报告/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );

  await page.setViewportSize({ width: 320, height: 900 });
  await expect.poll(() =>
    page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBe(true);
});

test("VAD 完成态提供真实全局指标与样本下钻", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("prama.lastEvaluationJobId", JSON.stringify("job-vad-001"));
    localStorage.setItem("prama.evaluationForm", JSON.stringify({ task: "vad" }));
  });
  await page.route("**/api/evaluations/job-vad-001", (route) =>
    route.fulfill({ json: completedVadSnapshot }),
  );
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "VAD 评估报告" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "评估总览" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText("90.00%", { exact: true })).toBeVisible();
  await expect(page.getByText("66.67%", { exact: true })).toBeVisible();
  await expect(page.getByText("2 / 3 命中", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /Segment Recall/ }).click();
  await expect(page.getByRole("tab", { name: "样本诊断" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("button", { name: "漏检 1", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: /sample-miss/ })).toBeVisible();
  await page.getByRole("button", { name: /sample-miss/ }).click();
  await expect(page.getByRole("heading", { name: "sample-miss" })).toBeVisible();
  await expect(page.getByRole("button", { name: /播放错误区间/ })).toBeDisabled();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() =>
    page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBe(true);
});

test("VAD 运行信息使用任务快照并可返回预填配置", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("prama.lastEvaluationJobId", JSON.stringify("job-vad-001"));
    localStorage.setItem("prama.evaluationForm", JSON.stringify({ task: "vad" }));
  });
  await page.route("**/api/evaluations/job-vad-001", (route) =>
    route.fulfill({ json: completedVadSnapshot }),
  );
  await page.goto("/");

  await expect(page.getByRole("button", { name: "重新评估", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "运行信息" }).click();
  const drawer = page.getByRole("dialog", { name: "运行信息" });
  await expect(drawer).toContainText("job-vad-001");
  await expect(drawer).toContainText("192.168.0.222:50021");
  await drawer.getByRole("button", { name: "基于此配置重新评估" }).click();

  await expect(page.getByRole("heading", { name: "评估配置" })).toBeVisible();
  await expect(page.getByRole("button", { name: "创建重新评估任务" })).toBeVisible();
  await expect(page.locator("#evaluation-dataset-path")).toHaveValue("data-bin/audiofolder/vad-demo");
  await expect(page.getByRole("button", { name: "返回当前报告" })).toBeVisible();
});

for (const scenario of [
  { previousStreaming: false, currentStreaming: true },
  { previousStreaming: true, currentStreaming: false },
]) {
  test(`VAD 重新评估保留当前流式设置 ${scenario.previousStreaming} → ${scenario.currentStreaming}`, async ({ page }) => {
    await page.addInitScript(
      ({ currentStreaming }) => {
        localStorage.setItem("prama.lastEvaluationJobId", JSON.stringify("job-vad-001"));
        localStorage.setItem(
          "prama.evaluationForm",
          JSON.stringify({
            task: "vad",
            target: "current.example:50021",
            dataset_path: "data-bin/audiofolder/current-vad",
            split: "current-split",
            limit: "9",
            streaming: !currentStreaming,
          }),
        );
      },
      { currentStreaming: scenario.currentStreaming },
    );
    await page.route("**/api/evaluations/job-vad-001", (route) =>
      route.fulfill({
        json: {
          ...completedVadSnapshot,
          request: {
            ...completedVadSnapshot.request,
            split: "historical-split",
            limit: 3,
            streaming: scenario.previousStreaming,
          },
        },
      }),
    );
    const submittedRequest = page.waitForRequest((request) =>
      request.url().endsWith("/api/evaluations") && request.method() === "POST",
    );
    await page.route("**/api/evaluations", (route) =>
      route.fulfill({ json: { job_id: "job-vad-rerun", status: "queued" } }),
    );
    await page.route("**/api/evaluations/job-vad-rerun/events", (route) =>
      route.fulfill({ status: 200, contentType: "text/event-stream", body: "" }),
    );
    await page.goto("/");

    await page.getByRole("button", { name: "设置", exact: true }).click();
    const streamingSetting = page.getByRole("checkbox", { name: "使用 VAD 流式接口" });
    await expect(streamingSetting).toBeChecked({ checked: !scenario.currentStreaming });
    await streamingSetting.setChecked(scenario.currentStreaming);
    await page.getByRole("button", { name: "在线评估", exact: true }).click();

    await page.getByRole("button", { name: "重新评估", exact: true }).click();
    await expect(page.getByText("数据源沿用本次任务，高级参数使用当前设置。"))
      .toBeVisible();
    await expect(page.getByLabel("VAD 引擎地址")).toHaveValue("192.168.0.222:50021");
    await expect(page.locator("#evaluation-dataset-path"))
      .toHaveValue("data-bin/audiofolder/vad-demo");
    await expect(page.getByLabel("Split")).toHaveValue("historical-split");
    await expect(page.getByLabel("Limit")).toHaveValue("3");

    await page.getByRole("button", { name: "返回当前报告" }).click();
    await page.getByRole("button", { name: "重新评估", exact: true }).click();
    await page.getByRole("button", { name: "创建重新评估任务" }).click();
    const request = await submittedRequest;
    const body = request.postDataJSON();
    expect(body.streaming).toBe(scenario.currentStreaming);
    expect(body.target).toBe("192.168.0.222:50021");
    expect(body.dataset_path).toBe("data-bin/audiofolder/vad-demo");
    expect(body.split).toBe("historical-split");
    expect(body.limit).toBe(3);
  });
}

test("VAD 运行中切换为只读进度工作台", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("prama.lastEvaluationJobId", JSON.stringify("job-vad-running"));
    localStorage.setItem("prama.evaluationForm", JSON.stringify({ task: "vad" }));
  });
  await page.route("**/api/evaluations/job-vad-running", (route) =>
    route.fulfill({
      json: {
        ...completedVadSnapshot,
        job_id: "job-vad-running",
        status: "running",
        progress: {
          status: "running",
          total: 10,
          processed: 4,
          evaluated: 4,
          current_id: "sample-0005",
        },
        result: null,
      },
    }),
  );
  await page.route("**/api/evaluations/job-vad-running/events", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: "" }),
  );
  await page.goto("/");

  await expect(page.getByRole("progressbar", { name: "VAD 评估进度" })).toHaveAttribute("aria-valuenow", "40");
  const currentSample = page.getByRole("region", { name: "当前样本" });
  await expect(currentSample).toBeVisible();
  await expect(currentSample).toContainText("sample-0005");
  await expect(page.getByRole("heading", { name: "运行配置" })).toBeVisible();
  await expect(page.locator(".evaluation-form")).toHaveCount(0);
  const layout = await page.evaluate(() => {
    const workGrid = document.querySelector<HTMLElement>(".work-grid")!;
    const workspace = document.querySelector<HTMLElement>(".vad-run-workspace")!;
    const currentSampleValue = document.querySelector<HTMLElement>(".vad-definition-list dd")!;
    return {
      workspaceRatio: workspace.getBoundingClientRect().width / workGrid.getBoundingClientRect().width,
      currentSampleWidth: currentSampleValue.getBoundingClientRect().width,
      gridColumns: getComputedStyle(workGrid).gridTemplateColumns,
    };
  });
  expect(layout.workspaceRatio).toBeGreaterThan(0.9);
  expect(layout.currentSampleWidth).toBeGreaterThan(180);
  expect(layout.gridColumns.split(" ")).toHaveLength(1);
});

test("概览使用冷灰工具栏与紧凑连续工作面", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "SE", exact: true }).click();

  await expect(page.getByText("SE Evaluation", { exact: true })).toHaveCount(0);
  await expect(page.locator(".workspace-header .page-tabs")).toBeVisible();
  await expect(page.getByRole("heading", { name: "评估配置" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "运行状态" })).toBeVisible();
  await expect(page.getByText("尚未开始评估", { exact: true })).toBeVisible();

  const layout = await page.evaluate(() => {
    const sidebar = document.querySelector<HTMLElement>(".sidebar")!;
    const form = document.querySelector<HTMLElement>(".evaluation-form")!;
    const statusCard = document.querySelector<HTMLElement>(".overview-column")!;
    const activeTask = document.querySelector<HTMLElement>(".task-nav button.active")!;
    const activeModule = document.querySelector<HTMLElement>(".module-item.active")!;
    const title = document.querySelector<HTMLElement>(".workspace-title-row h1")!;
    const actionBar = document.querySelector<HTMLElement>(".form-action-bar")!;
    const datasetOptions = document.querySelector<HTMLElement>(".dataset-options-grid")!;
    const idleContent = document.querySelector<HTMLElement>(".idle-run-state-content")!;
    const actionControls = [
      document.querySelector<HTMLElement>(".connectivity-button")!,
      document.querySelector<HTMLElement>(".dataset-upload-button")!,
      document.querySelector<HTMLElement>(".dataset-browser-button")!,
    ].map((control) => ({
      height: control.getBoundingClientRect().height,
      background: getComputedStyle(control).backgroundColor,
      border: getComputedStyle(control).borderColor,
    }));
    const formBox = form.getBoundingClientRect();
    const actionBox = actionBar.getBoundingClientRect();
    const optionsBox = datasetOptions.getBoundingClientRect();
    const statusBox = statusCard.getBoundingClientRect();
    const idleBox = idleContent.getBoundingClientRect();
    return {
      bodyBackground: getComputedStyle(document.body).backgroundColor,
      sidebarWidth: sidebar.getBoundingClientRect().width,
      formWidth: form.getBoundingClientRect().width,
      formBackground: getComputedStyle(form).backgroundColor,
      statusBackground: getComputedStyle(statusCard).backgroundColor,
      formShadow: getComputedStyle(form).boxShadow,
      statusDivider: getComputedStyle(statusCard).borderLeftColor,
      activeTaskBackground: getComputedStyle(activeTask).backgroundColor,
      activeModuleBackground: getComputedStyle(activeModule).backgroundColor,
      titleFontSize: Number.parseFloat(getComputedStyle(title).fontSize),
      actionOffset: actionBox.y - (optionsBox.y + optionsBox.height),
      actionDistanceFromBottom: formBox.y + formBox.height - (actionBox.y + actionBox.height),
      idleCenterRatio:
        (idleBox.y + idleBox.height / 2 - statusBox.y) / statusBox.height,
      idleDot: getComputedStyle(document.querySelector<HTMLElement>(".status-dot")!).backgroundColor,
      actionControls,
    };
  });

  expect(layout.bodyBackground).toBe("rgb(247, 248, 250)");
  expect(layout.sidebarWidth).toBeCloseTo(204, 0);
  expect(layout.formWidth).toBeCloseTo(376, 0);
  expect(layout.formBackground).toBe("rgba(0, 0, 0, 0)");
  expect(layout.statusBackground).toBe("rgba(0, 0, 0, 0)");
  expect(layout.formShadow).toBe("none");
  expect(layout.statusDivider).toBe("rgb(228, 228, 231)");
  expect(layout.activeTaskBackground).toBe("rgb(234, 245, 245)");
  expect(layout.activeModuleBackground).toBe("rgba(0, 0, 0, 0)");
  expect(layout.titleFontSize).toBeGreaterThanOrEqual(22);
  expect(layout.titleFontSize).toBeLessThanOrEqual(26);
  expect(layout.actionOffset).toBeCloseTo(22, 0);
  expect(layout.actionDistanceFromBottom).toBeGreaterThan(200);
  expect(layout.idleCenterRatio).toBeGreaterThan(0.35);
  expect(layout.idleCenterRatio).toBeLessThan(0.47);
  expect(layout.idleDot).toBe("rgb(161, 161, 170)");
  for (const control of layout.actionControls) {
    expect(control.height).toBeCloseTo(44, 0);
    expect(control.background).toBe("rgb(255, 255, 255)");
    expect(control.border).toBe("rgb(228, 228, 231)");
  }
  await expect(page.getByRole("button", { name: "测试连接", exact: true })).toBeVisible();
});

test("目录浏览器可从不存在的数据集路径恢复", async ({ page }) => {
  await page.route("**/api/files/directories?path=missing%2Fdataset", (route) =>
    route.fulfill({ status: 404, json: { detail: "目录不存在: missing/dataset" } }),
  );
  await page.route("**/api/files/directories?path=data-bin", (route) =>
    route.fulfill({
      json: {
        currentPath: "data-bin",
        parentPath: null,
        entries: Array.from({ length: 30 }, (_, index) => ({
          name: `audio-${index}.wav`,
          path: `data-bin/audio-${index}.wav`,
          kind: "file",
        })),
      },
    }),
  );

  await page.goto("/");
  const datasetPath = page.locator("#evaluation-dataset-path");
  await datasetPath.fill("missing/dataset");
  await page.getByRole("button", { name: "浏览", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText("目录不存在");
  const dialog = page.getByRole("dialog", { name: "选择服务器目录" });
  await dialog.getByLabel("目录路径").fill("data-bin");
  await dialog.getByRole("button", { name: "打开目录", exact: true }).click();
  await expect(dialog.getByText("data-bin", { exact: true })).toBeVisible();
  const directoryList = dialog.locator(".directory-list");
  await expect
    .poll(() => directoryList.evaluate((element) => element.scrollHeight > element.clientHeight))
    .toBe(true);
  await directoryList.hover();
  await page.mouse.wheel(0, 300);
  await expect.poll(() => directoryList.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
  await dialog.getByRole("button", { name: "选择当前目录", exact: true }).click();

  await expect(datasetPath).toHaveValue("data-bin");
});

test("排队和失败状态不展示虚假进度", async ({ page }) => {
  let snapshot = {
    ...completedSnapshot,
    status: "queued",
    progress: { status: "queued" },
    result: null,
  };
  await page.addInitScript(() => {
    localStorage.setItem("prama.lastEvaluationJobId", JSON.stringify("job-manas-001"));
  });
  await page.route("**/api/evaluations/job-manas-001", (route) =>
    route.fulfill({ json: snapshot }),
  );

  await page.goto("/");
  await expect(page.getByText("任务已进入队列", { exact: true })).toBeVisible();
  await expect(page.locator(".status-dot")).toHaveCSS("background-color", "rgb(217, 119, 6)");
  await expect(page.locator(".progress-percent")).toHaveCount(0);
  await expect(page.getByRole("progressbar", { name: "评估进度" })).toHaveCount(0);

  snapshot = {
    ...snapshot,
    status: "failed",
    progress: { status: "failed" },
    error: "引擎连接失败",
  };
  await page.reload();
  await expect(page.getByText("评估未完成", { exact: true })).toBeVisible();
  await expect(page.getByText("引擎连接失败", { exact: true })).toBeVisible();
  await expect(page.locator(".status-dot")).toHaveCSS("background-color", "rgb(220, 38, 38)");
  await expect(page.locator(".progress-percent")).toHaveCount(0);
  await expect(page.getByRole("progressbar", { name: "评估进度" })).toHaveCount(0);
});

test("完成后的报告独占主工作区", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("prama.lastEvaluationJobId", JSON.stringify("job-manas-001"));
  });
  await page.route("**/api/evaluations/job-manas-001", (route) =>
    route.fulfill({ json: completedSnapshot }),
  );
  await page.goto("/");

  await expect(page.locator(".status-pill")).toContainText("已完成");
  await page.getByRole("tab", { name: /对齐报告/ }).click();
  await expect(page.locator(".evaluation-form")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "对齐报告" })).toBeVisible();
  await expect(page.locator(".work-grid")).toHaveClass(/report-full/);
});

test("设置重置需要确认并提供完成反馈", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await expect(page.getByText("更改会自动保存到当前浏览器")).toBeVisible();
  await expect.poll(async () => {
    const box = await page.locator(".settings-panel").boundingBox();
    return box?.width ?? 0;
  }).toBeGreaterThan(900);

  await page.getByRole("button", { name: "重置", exact: true }).click();
  const dialog = page.getByRole("alertdialog", { name: "恢复全部默认设置？" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "恢复默认值" }).click();
  await expect(page.getByRole("status")).toContainText("已恢复默认值");
});

test("帮助目录和服务器目录弹窗可访问", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "帮助", exact: true }).click();
  await expect(page.getByRole("navigation", { name: "帮助文档目录" })).toBeVisible();

  await page.getByRole("button", { name: "在线评估", exact: true }).click();
  await page.getByRole("button", { name: "浏览", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "选择服务器目录" })).toBeVisible();
  await page.getByRole("button", { name: "关闭目录浏览器" }).click();
  await expect(page.getByRole("button", { name: "浏览", exact: true })).toBeFocused();
});

test("ASR 运行中接收逐条指标并显示对齐报告", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("prama.lastEvaluationJobId", JSON.stringify("job-stream-001"));
  });
  await page.route("**/api/evaluations/job-stream-001", (route) =>
    route.fulfill({ json: {
      ...completedSnapshot,
      job_id: "job-stream-001",
      status: "running",
      result: null,
      progress: { status: "running", total: 100, processed: 1, evaluated: 1 },
    } }),
  );
  const summary = {
    ref_words: 2, hyp_words: 2, correct: 1, substitutions: 1,
    deletions: 0, insertions: 0, sentence_count: 1, sentence_errors: 1,
    wer: 50, accuracy: 50,
  };
  const payload = {
    metric: "wer", summary,
    utterance: { ...completedSnapshot.result.wer_report.utterances[0], summary },
  };
  await page.route("**/api/evaluations/job-stream-001/events", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream",
      // 重复事件不应追加第二条记录，且没有 done 事件。
      body: `event: metric_result\ndata: ${JSON.stringify(payload)}\n\nevent: metric_result\ndata: ${JSON.stringify(payload)}\n\n`,
    }),
  );
  await page.goto("/");
  await expect(page.locator(".status-pill")).toContainText("运行中");
  await page.getByRole("tab", { name: /对齐报告/ }).click();
  await expect(page.getByText("sample-001", { exact: true })).toHaveCount(1);
  await expect(page.getByText("sample-001", { exact: true })).toBeVisible();
});
