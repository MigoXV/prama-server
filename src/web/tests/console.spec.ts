import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { completedSnapshot, completedVadSnapshot } from "./fixtures";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

async function mockJob(page: Page, snapshot = completedSnapshot) {
  await page.route(`**/api/evaluations/${snapshot.job_id}`, (r) =>
    r.fulfill({ json: snapshot }),
  );
  await page.route(`**/api/evaluations/${snapshot.job_id}/events`, (r) =>
    r.fulfill({ contentType: "text/event-stream", body: ": test\n\n" }),
  );
}
async function noOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
}

async function controllableEvents(page: Page) {
  await page.addInitScript(() => {
    const streams = new Map<string, EventTarget>();
    class TestEventSource extends EventTarget {
      readyState = 1;
      constructor(public url: string) {
        super();
        streams.set(url, this);
      }
      close() {
        this.readyState = 2;
      }
    }
    (window as any).EventSource = TestEventSource;
    (window as any).emitEvaluation = (
      id: string,
      type: string,
      payload?: unknown,
    ) => {
      const event =
        payload === undefined
          ? new Event(type)
          : new MessageEvent(type, { data: JSON.stringify(payload) });
      streams.get(`/api/evaluations/${id}/events`)?.dispatchEvent(event);
    };
  });
}

test("SSE 完成更新当前报告，保持用户选择的页面", async ({ page }) => {
  await controllableEvents(page);
  await mockJob(page, {
    ...completedSnapshot,
    status: "running",
    result: null,
  } as any);
  await page.goto("/evaluations/job-manas-001?view=report");
  await expect(
    page.getByRole("heading", { name: "报告尚未生成" }),
  ).toBeVisible();
  await page.evaluate(
    (snapshot) =>
      (window as any).emitEvaluation(snapshot.job_id, "done", snapshot),
    completedSnapshot,
  );
  await expect(page.locator(".job-heading")).toContainText("已完成");
  await expect(page).toHaveURL(/view=report$/);
  await expect(page.getByRole("heading", { name: "报告尚未生成" })).toHaveCount(
    0,
  );
});

test("SSE 断线保留进度，业务错误进入失败状态", async ({ page }) => {
  await controllableEvents(page);
  await mockJob(page, {
    ...completedSnapshot,
    status: "running",
    result: null,
    progress: { total: 20, processed: 3, evaluated: 3, status: "running" },
  } as any);
  await page.goto("/evaluations/job-manas-001");
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "3");
  await page.evaluate(() =>
    (window as any).emitEvaluation("job-manas-001", "error"),
  );
  await expect(page.getByText(/实时连接中断/)).toBeVisible();
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "3");
  await page.evaluate(() =>
    (window as any).emitEvaluation("job-manas-001", "error", {
      message: "引擎处理失败",
    }),
  );
  await expect(page.locator(".job-heading")).toContainText("失败");
  await expect(page.getByText("引擎处理失败", { exact: true })).toBeVisible();
});

test("导入目录后回填服务器路径并保留其他任务草稿", async ({
  page,
}, testInfo) => {
  const directory = testInfo.outputPath("dataset");
  await mkdir(directory, { recursive: true });
  await writeFile(
    path.join(directory, "sample.wav"),
    Buffer.from("RIFF test fixture"),
  );
  let uploaded = false;
  await page.route("**/api/datasets/upload", async (route) => {
    uploaded =
      route.request().postDataBuffer()?.includes(Buffer.from("sample.wav")) ??
      false;
    await route.fulfill({
      json: {
        dataset_path: "data-bin/imported/test-dataset",
        imported_count: 1,
        message: "目录已导入",
      },
    });
  });
  await page.goto("/evaluations/new?task=asr");
  await page.locator('input[type="file"]').setInputFiles(directory);
  await expect(page.getByLabel("数据集路径", { exact: true })).toHaveValue(
    "data-bin/imported/test-dataset",
  );
  expect(uploaded).toBe(true);
  await page.getByRole("radio", { name: /VAD/ }).check();
  await expect(page.getByLabel("数据集路径", { exact: true })).not.toHaveValue(
    "data-bin/imported/test-dataset",
  );
  await page.getByRole("radio", { name: /ASR/ }).check();
  await expect(page.getByLabel("数据集路径", { exact: true })).toHaveValue(
    "data-bin/imported/test-dataset",
  );
});

test("统一创建入口保留分任务草稿，路径获得完整宽度", async ({ page }) => {
  await page.goto("/evaluations/new");
  await page
    .getByLabel("数据集路径", { exact: true })
    .fill("data-bin/my-asr-dataset");
  await page.getByRole("radio", { name: /VAD/ }).check();
  await page
    .getByLabel("数据集路径", { exact: true })
    .fill("data-bin/my-vad-dataset");
  await page.getByRole("radio", { name: /ASR/ }).check();
  await expect(page.getByLabel("数据集路径", { exact: true })).toHaveValue(
    "data-bin/my-asr-dataset",
  );
  await page.reload();
  await expect(page.getByLabel("数据集路径", { exact: true })).toHaveValue(
    "data-bin/my-asr-dataset",
  );
  expect(
    (await page.getByLabel("数据集路径", { exact: true }).boundingBox())!.width,
  ).toBeGreaterThan(400);
});
for (const task of ["asr", "vad", "lid", "keyword", "denoise"])
  test(`${task} 创建请求与真实接口一致，创建后进入任务`, async ({ page }) => {
    let body: any;
    await page.route("**/api/evaluations", async (r) => {
      body = r.request().postDataJSON();
      await r.fulfill({
        json: { job_id: `created-${task}`, status: "queued" },
      });
    });
    await page.route(`**/api/evaluations/created-${task}`, (r) =>
      r.fulfill({
        json: {
          ...completedSnapshot,
          job_id: `created-${task}`,
          request: { ...completedSnapshot.request, task },
          status: "queued",
          result: null,
        },
      }),
    );
    await page.route("**/api/evaluations/*/events", (r) =>
      r.fulfill({ contentType: "text/event-stream", body: ": test\n\n" }),
    );
    await page.goto("/evaluations/new?task=" + task);
    await page.getByRole("button", { name: "启动评估", exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`/evaluations/created-${task}`));
    expect(body.task).toBe(task);
    expect(body.sample_rate).toBe(16000);
    expect(body.limit).toBe(null);
    expect(Array.isArray(body.hotwords)).toBe(true);
    expect(body.inference_concurrency).toBe(0);
    await expect(
      page.getByRole("heading", { name: "任务已进入队列" }),
    ).toBeVisible();
    await expect(page.getByRole("progressbar")).toHaveCount(0);
  });
test("提交失败保留输入并允许重试", async ({ page }) => {
  let attempts = 0;
  await page.route("**/api/evaluations", (r) => {
    attempts++;
    return r.fulfill({ status: 503, json: { detail: "服务暂时不可用" } });
  });
  await page.goto("/evaluations/new");
  await page.getByRole("button", { name: "启动评估", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("服务暂时不可用");
  await expect(page.getByRole("alert")).toBeInViewport({ ratio: 1 });
  await expect(page.getByLabel("数据集路径", { exact: true })).not.toHaveValue(
    "",
  );
  await page.getByRole("button", { name: "启动评估", exact: true }).click();
  expect(attempts).toBe(2);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("alert")).toBeInViewport({ ratio: 1 });
});
test("设置刷新和浏览器导航保持位置，默认值不污染已有草稿", async ({ page }) => {
  await page.goto("/evaluations/new");
  await page
    .getByLabel("数据集路径", { exact: true })
    .fill("data-bin/preserved");
  await page.getByRole("link", { name: "默认设置", exact: true }).click();
  await page.getByLabel("采样率（Hz）", { exact: true }).fill("8000");
  await page.getByRole("button", { name: "保存默认值", exact: true }).click();
  await page.reload();
  await expect(page).toHaveURL(/\/settings$/);
  await expect(page.getByLabel("采样率（Hz）", { exact: true })).toHaveValue(
    "8000",
  );
  await page.getByRole("link", { name: "新建评估", exact: true }).click();
  await expect(page.getByLabel("数据集路径", { exact: true })).toHaveValue(
    "data-bin/preserved",
  );
  await expect(page.locator(".submission-summary")).toContainText("16000");
  await page.goBack();
  await expect(
    page.getByRole("heading", { name: "默认设置", exact: true }),
  ).toBeVisible();
});
test("旧浏览器数据迁移，最近任务不会强制覆盖首页", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "prama.evaluationForm",
      JSON.stringify({
        task: "asr",
        dataset_path: "data-bin/legacy",
        sample_rate: "8000",
      }),
    );
    localStorage.setItem(
      "prama.lastEvaluationJobId",
      JSON.stringify("job-manas-001"),
    );
  });
  await mockJob(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "新建评估" })).toBeVisible();
  await expect(page.getByLabel("数据集路径", { exact: true })).toHaveValue(
    "data-bin/legacy",
  );
  await page.getByRole("link", { name: "当前任务", exact: true }).click();
  await expect(page.locator(".task-table-row")).toContainText("job-manas-001");
});
test("任务 URL 直达报告，刷新和前进后退恢复筛选与样本", async ({ page }) => {
  await mockJob(page, completedVadSnapshot);
  await page.goto("/evaluations/job-vad-001?view=report");
  await expect(page.getByText("90.00%", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "样本诊断", exact: true }).click();
  await page.getByRole("button", { name: "漏检 1", exact: true }).click();
  await page.getByRole("button", { name: /sample-miss/ }).click();
  await expect(page).toHaveURL(/sample=sample-miss/);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "sample-miss", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "返回样本列表" }).click();
  await expect(
    page.getByRole("button", { name: "漏检 1", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  expect(
    (await page
      .getByRole("button", { name: "漏检 1", exact: true })
      .boundingBox())!.height,
  ).toBeLessThan(45);
});
for (const streaming of [false, true])
  test(`重新评估明确沿用当前高级参数 streaming=${streaming}`, async ({
    page,
  }) => {
    await page.addInitScript(
      ({ streaming }) =>
        localStorage.setItem(
          "prama.evaluationForm",
          JSON.stringify({ task: "vad", streaming }),
        ),
      { streaming },
    );
    await mockJob(page, {
      ...completedVadSnapshot,
      request: { ...completedVadSnapshot.request, streaming: !streaming },
    });
    await page.goto("/evaluations/job-vad-001?view=configuration");
    await page.getByRole("button", { name: "重新评估", exact: true }).click();
    await expect(page.getByLabel("数据集路径", { exact: true })).toHaveValue(
      completedVadSnapshot.request.dataset_path,
    );
    await page.locator(".advanced-settings>summary").click();
    expect(await page.getByLabel("使用 VAD 流式接口").isChecked()).toBe(
      streaming,
    );
    await expect(page.locator(".reevaluation-diff")).toContainText("streaming");
  });
test("不同任务并行订阅，切换后不会串进度", async ({ page }) => {
  await page.addInitScript(() =>
    localStorage.setItem(
      "prama.latestJobs",
      JSON.stringify({ asr: "job-manas-001", vad: "job-vad-001" }),
    ),
  );
  await mockJob(page, {
    ...completedSnapshot,
    status: "running",
    result: null,
    progress: { status: "running", total: 20, processed: 3, evaluated: 3 },
  } as any);
  await mockJob(page, {
    ...completedVadSnapshot,
    status: "running",
    result: null,
    progress: { status: "running", total: 10, processed: 7, evaluated: 7 },
  } as any);
  await page.goto("/evaluations");
  await expect(page.locator(".task-table-row")).toHaveCount(2);
  await page.locator(".task-table-row").filter({ hasText: "VAD" }).click();
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "7");
  await page.getByRole("button", { name: "← 当前任务", exact: true }).click();
  await page.locator(".task-table-row").filter({ hasText: "ASR" }).click();
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "3");
});
test("失效任务呈现不可用并保留恢复路径", async ({ page }) => {
  await page.route("**/api/evaluations/missing", (r) =>
    r.fulfill({ status: 404, json: { detail: "评估任务不存在" } }),
  );
  await page.goto("/evaluations/missing?view=report");
  await expect(page.getByRole("heading", { name: "任务不可用" })).toBeVisible();
  await expect(page.getByRole("button", { name: "重新读取" })).toBeVisible();
  await expect(page.getByRole("button", { name: "新建评估" })).toBeVisible();
});
test("目录错误可直接返回允许根目录", async ({ page }) => {
  await page.goto("/evaluations/new");
  await page.getByLabel("数据集路径", { exact: true }).fill("missing");
  await page.getByRole("button", { name: "浏览服务器目录" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("button", { name: "返回允许的根目录" }).click();
  await expect(
    dialog.getByRole("button", { name: "选择当前目录" }),
  ).toBeEnabled();
  await dialog.getByRole("button", { name: "选择当前目录" }).click();
  await expect(page.getByLabel("数据集路径", { exact: true })).not.toHaveValue(
    "missing",
  );
});
test("帮助加载失败可重试", async ({ page }) => {
  let count = 0;
  await page.route("**/api/help", (r) =>
    r.fulfill(
      ++count === 1
        ? { status: 503, json: { detail: "临时失败" } }
        : { json: { title: "帮助", markdown: "# 帮助\n\n## 恢复成功" } },
    ),
  );
  await page.goto("/help");
  await page.getByRole("button", { name: "重新加载" }).click();
  await expect(page.getByRole("heading", { name: "恢复成功" })).toBeVisible();
});
for (const width of [320, 390, 768, 1280, 1440])
  test(`${width}px 创建与 VAD 详情可操作且没有整页溢出`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/evaluations/new");
    await noOverflow(page);
    if (width < 641) {
      const box = await page
        .getByRole("button", { name: "启动评估", exact: true })
        .boundingBox();
      expect(box!.y + box!.height).toBeLessThanOrEqual(900);
      await page.getByRole("button", { name: "打开导航" }).click();
      await expect(page.getByRole("dialog", { name: "导航" })).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(
        page.getByRole("button", { name: "打开导航" }),
      ).toBeFocused();
    }
    await mockJob(page, completedVadSnapshot);
    await page.goto("/evaluations/job-vad-001?view=report");
    await expect(page.getByText("90.00%", { exact: true })).toBeVisible();
    await noOverflow(page);
    await page.goto(
      "/evaluations/job-vad-001?view=diagnosis&sample=sample-miss",
    );
    await noOverflow(page);
    await expect(
      page.getByRole("heading", { name: "sample-miss" }),
    ).toBeVisible();
  });
test("200% 缩放等效视口、reduced-motion 与主流程无障碍", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 640, height: 450 });
  await page.goto("/evaluations/new");
  await noOverflow(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(results.violations).toEqual([]);
  await page.goto("/settings");
  const settings = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(settings.violations).toEqual([]);
});
