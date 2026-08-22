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

test("任务导航、视图标签与窄屏布局保持可操作", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "ASR 评估" })).toBeVisible();
  await page.getByRole("button", { name: "VAD", exact: true }).click();
  await expect(page.getByRole("heading", { name: "VAD 评估" })).toBeVisible();

  const overviewTab = page.getByRole("tab", { name: /运行概览/ });
  await overviewTab.focus();
  await overviewTab.press("ArrowRight");
  await expect(page.getByRole("tab", { name: /VAD 指标/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );

  await page.setViewportSize({ width: 320, height: 900 });
  await expect.poll(() =>
    page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBe(true);
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
