import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const currentDirectory = fileURLToPath(new URL(".", import.meta.url));
const repositoryRoot = resolve(currentDirectory, "../..");
const configuredChromium = process.env.PLAYWRIGHT_CHROMIUM_PATH;
const systemChromium = "/usr/bin/chromium";
const chromiumExecutable = configuredChromium ||
  (existsSync(systemChromium) ? systemChromium : undefined);

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:8010",
    viewport: { width: 1280, height: 900 },
    launchOptions: chromiumExecutable ? { executablePath: chromiumExecutable } : {},
  },
  webServer: {
    command:
      "pnpm --dir src/web run build && PRAMA_WORKDIR=$PWD/data-bin poetry run prama-server serve-http --host 127.0.0.1 --port 8010",
    cwd: repositoryRoot,
    url: "http://127.0.0.1:8010/api/health",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
