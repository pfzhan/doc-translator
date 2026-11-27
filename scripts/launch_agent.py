"""Run the server in the background on macOS as a launchd agent: started at login, restarted when it exits, and
independent of the terminal or editor it was set up from.

    python3 scripts/launch_agent.py install [--host 127.0.0.1] [--port 8765]
    python3 scripts/launch_agent.py uninstall

install writes ~/Library/LaunchAgents/com.doc-translator.server.plist for this checkout, loads it and waits for the
server to answer. The agent runs scripts/doc-translator, so System Settings > Login Items lists it under that name.
Output goes to server.log in the project, as with the nohup command in the README.

launchd does not read the shell's environment: ALLOWED_HOSTS, ACCESS_TOKEN, DOC_TRANSLATOR_DATA and CC_SWITCH_DB, when
set while installing, are written into the agent (the file is readable by the current user only). Install again after
changing them, or after moving the project. uninstall stops the server and removes the agent.
"""

import argparse
import os
import plistlib
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "com.doc-translator.server"
AGENT = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
# Settings the server reads from its environment, carried over from the shell that installs the agent
PASSED_ENV = ("ALLOWED_HOSTS", "ACCESS_TOKEN", "DOC_TRANSLATOR_DATA", "CC_SWITCH_DB")
# launchd starts agents with a bare PATH; Homebrew's tools (tesseract and others) live in these
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
# A server that keeps failing at start (port taken, broken install) is restarted at most this often, in seconds
THROTTLE = 30
START_TIMEOUT = 60


def agent(root: Path, host: str, port: int, environ: Mapping[str, str]) -> dict[str, object]:
    """The launchd job of the server in the checkout at root."""
    env = {"PATH": PATH, "HOST": host, "PORT": str(port)}
    env.update({name: environ[name] for name in PASSED_ENV if environ.get(name)})
    log = str(root / "server.log")
    return {
        "Label": LABEL,
        "ProgramArguments": [str(root / "scripts" / "doc-translator")],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": THROTTLE,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _loaded() -> bool:
    return subprocess.run(["launchctl", "print", f"{_domain()}/{LABEL}"], capture_output=True).returncode == 0


def _unload() -> None:
    if _loaded():
        subprocess.run(["launchctl", "bootout", f"{_domain()}/{LABEL}"], check=True)


def _answers(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _wait(host: str, port: int, up: bool, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _answers(host, port) == up:
            return True
        time.sleep(0.5)
    return _answers(host, port) == up


def install(host: str, port: int) -> int:
    if not (ROOT / ".venv" / "bin" / "python").exists():
        print(f"没有找到 {ROOT / '.venv'}，请先按 README 创建虚拟环境并安装依赖", file=sys.stderr)
        return 1
    # An earlier copy of this agent is replaced; the port must then be free, or another server holds it
    _unload()
    if not _wait(host, port, up=False, timeout=10):
        print(f"{host}:{port} 已被其他程序占用（比如手动启动的服务），请先停止它", file=sys.stderr)
        return 1
    AGENT.parent.mkdir(parents=True, exist_ok=True)
    AGENT.write_bytes(plistlib.dumps(agent(ROOT, host, port, os.environ)))
    AGENT.chmod(0o600)  # may hold ACCESS_TOKEN
    subprocess.run(["launchctl", "bootstrap", _domain(), str(AGENT)], check=True)
    if not _wait(host, port, up=True, timeout=START_TIMEOUT):
        print(f"服务没有在 {START_TIMEOUT} 秒内启动，请查看 {ROOT / 'server.log'}", file=sys.stderr)
        return 1
    print(f"已安装 {AGENT}，服务运行在 http://{host}:{port}，登录后自动启动")
    return 0


def uninstall() -> int:
    _unload()
    AGENT.unlink(missing_ok=True)
    print(f"已停止服务并删除 {AGENT}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="在 macOS 上用 launchd 让服务在后台常驻")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("install", help="安装并启动（再次运行会按当前设置重装）")
    setup.add_argument("--host", default="127.0.0.1")
    setup.add_argument("--port", type=int, default=8765)
    commands.add_parser("uninstall", help="停止并删除")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("launchd 只在 macOS 上可用", file=sys.stderr)
        return 1
    if args.command == "install":
        return install(args.host, args.port)
    return uninstall()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
