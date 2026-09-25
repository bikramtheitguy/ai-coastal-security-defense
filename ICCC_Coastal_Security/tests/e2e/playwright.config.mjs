// E2E: starts the backend (serving the built frontend) on a fresh throw-away database with the
// simulator disabled, so movement is advanced deterministically through the API.
import { defineConfig } from "@playwright/test";
const PORT = process.env.E2E_PORT ?? "8100";
const PY = process.env.PYTHON ?? "python3";
export default defineConfig({
  testDir: ".",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://localhost:${PORT}`, viewport: { width: 1600, height: 900 }, trace: "retain-on-failure" },
  webServer: {
    command: `bash -c 'rm -rf /tmp/iccc_e2e && mkdir -p /tmp/iccc_e2e && cd ../../backend && DATA_DIR=/tmp/iccc_e2e SIM_ENABLED=false PBKDF2_ITERATIONS=20000 ${PY} -m uvicorn app.main:app --port ${PORT}'`,
    url: `http://localhost:${PORT}/api/public/info`,
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
