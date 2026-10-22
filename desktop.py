import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

WINDOW_TITLE = "SteelEye 钢铁表面缺陷智能检测系统"
WINDOW_SIZE = (1440, 900)
STARTUP_TIMEOUT_SECONDS = 120
HEALTH_POLL_INTERVAL_SECONDS = 0.5
BUNDLED_WEIGHT_NAME = "yolov8n.pt"


def resolve_runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def silence_missing_streams(runtime_dir: Path) -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = runtime_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = open(log_dir / "desktop.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def seed_default_weights(runtime_dir: Path) -> None:
    bundle_dir = Path(getattr(sys, "_MEIPASS", runtime_dir))
    source = bundle_dir / BUNDLED_WEIGHT_NAME
    target = runtime_dir / "models" / "weights" / BUNDLED_WEIGHT_NAME
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(port: int):
    import uvicorn
    from server import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    return server


def wait_until_ready(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except OSError:
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
    return False


def open_window(url: str, server) -> None:
    try:
        import webview

        webview.create_window(WINDOW_TITLE, url, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
        webview.start()
    except Exception:
        webbrowser.open(url)
        while not server.should_exit:
            time.sleep(1)


def main() -> None:
    runtime_dir = resolve_runtime_dir()
    os.chdir(runtime_dir)
    silence_missing_streams(runtime_dir)
    seed_default_weights(runtime_dir)

    port = find_free_port()
    server = start_server(port)
    if not wait_until_ready(port):
        raise SystemExit("后端服务启动超时")

    open_window(f"http://127.0.0.1:{port}", server)
    server.should_exit = True
    os._exit(0)


if __name__ == "__main__":
    main()
