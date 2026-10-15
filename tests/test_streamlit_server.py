"""Check the actual Streamlit server entry point in a fresh process."""

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_server_starts_and_serves_health_and_html(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    log_path = tmp_path / "streamlit.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "dashboard/app.py"),
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
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text(encoding="utf-8")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/_stcore/health", timeout=2
                    ) as response:
                        if response.read() == b"ok":
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise AssertionError(f"Streamlit did not become healthy: {log_path.read_text()}")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
                assert response.status == 200
                assert b"<html" in response.read().lower()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
