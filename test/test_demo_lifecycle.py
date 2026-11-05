"""Legacy graphical entrypoints must report their port and stop on one interrupt."""

import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

pytest.importorskip("viser")
pytest.importorskip("yourdfpy")


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX SIGINT delivery")
@pytest.mark.parametrize(
    "module,arguments,ready",
    [
        ("example.demo", ["--steps", "2", "--dt", "0.01"], "Viewer ready at"),
        ("viser_viewer", [], "Viser viewer running at"),
    ],
)
def test_viewer_reports_bound_port_and_exits_on_one_interrupt(module, arguments, ready):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", module, "--port", str(port), *arguments],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = queue.Queue()
    captured = []

    def read_output():
        for line in process.stdout:
            captured.append(line)
            lines.put(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 20
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "".join(captured)
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                pytest.fail("Viewer did not become ready:\n" + "".join(captured))
            if line.startswith(ready):
                break
        match = re.search(r"http://127\.0\.0\.1:(\d+)", line)
        assert match and int(match.group(1)) > 0, line
        with urllib.request.urlopen(match.group(), timeout=3) as response:
            assert response.status == 200
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == 0, "".join(captured)
        reader.join(timeout=2)
        assert "Server stopped" in "".join(captured)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=2)
        process.stdout.close()
