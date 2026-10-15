"""Real-browser QA against a loopback-only Streamlit process, using invented data."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/browser"


@pytest.fixture(scope="session")
def app_url():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    interpreter = os.environ.get("STREAMLIT_PYTHON", sys.executable)
    with (EVIDENCE / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                interpreter,
                "-m",
                "streamlit",
                "run",
                "dashboard/app.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--server.fileWatcherType",
                "none",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                assert process.poll() is None, "Streamlit exited; inspect server.log"
                try:
                    with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as response:
                        if response.read() == b"ok":
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise AssertionError("Streamlit did not start within 45 seconds")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@pytest.fixture(scope="session")
def browser():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        (EVIDENCE / "browser-version.json").write_text(
            json.dumps({"engine": "Chromium", "version": browser.version}),
            encoding="utf-8",
        )
        yield browser
        browser.close()


@pytest.fixture
def page(browser, app_url, request):
    context = browser.new_context(
        viewport={"width": 1365, "height": 900},
        reduced_motion="reduce",
        timezone_id="Asia/Kolkata",
    )
    failures, external = [], []

    def local_only(route):
        url = route.request.url
        if urlsplit(url).hostname in {"127.0.0.1", "localhost"}:
            route.continue_()
        else:
            external.append(url)
            route.abort()

    context.route("**/*", local_only)
    page = context.new_page()
    page.set_default_timeout(15_000)
    page.on("pageerror", lambda error: failures.append(str(error)))
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page.goto(app_url)
    try:
        yield page
    finally:
        name = request.node.name.replace("[", "-").replace("]", "")
        page.screenshot(path=str(EVIDENCE / f"{name}.jpg"), type="jpeg", quality=75)
        (EVIDENCE / f"{name}.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
        context.tracing.stop(path=str(EVIDENCE / f"{name}-trace.zip"))
        context.close()
    assert not failures, failures
    assert not external, f"Unexpected external requests: {external}"
