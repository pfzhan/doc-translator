"""The launchd agent that keeps the server running on macOS (scripts/launch_agent.py)."""

import os
import plistlib
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import launch_agent


def test_agent_runs_the_named_wrapper_of_the_checkout() -> None:
    """The job starts scripts/doc-translator, which gives the login item its name, keeps it alive and logs to
    server.log; it carries the address and nothing of the shell beyond the server's own settings."""
    root = Path("/opt/doc-translator")
    environ = {"ACCESS_TOKEN": "secret", "ALLOWED_HOSTS": "", "HOME": "/Users/x", "OPENAI_API_KEY": "sk-x"}
    job = launch_agent.agent(root, "0.0.0.0", 9000, environ)
    assert job["Label"] == "com.doc-translator.server"
    assert job["ProgramArguments"] == ["/opt/doc-translator/scripts/doc-translator"]
    assert job["WorkingDirectory"] == "/opt/doc-translator"
    assert job["EnvironmentVariables"] == {
        "PATH": launch_agent.PATH,
        "HOST": "0.0.0.0",
        "PORT": "9000",
        "ACCESS_TOKEN": "secret",
    }
    assert job["KeepAlive"] is True and job["RunAtLoad"] is True and job["ThrottleInterval"] == 30
    assert job["StandardOutPath"] == job["StandardErrorPath"] == "/opt/doc-translator/server.log"
    # launchd reads it back as written
    assert plistlib.loads(plistlib.dumps(job)) == job


def test_wrapper_starts_uvicorn_from_the_venv() -> None:
    wrapper = launch_agent.ROOT / "scripts" / "doc-translator"
    assert os.access(wrapper, os.X_OK)
    text = wrapper.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert '.venv/bin/python -m uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"' in text
    assert subprocess.run(["sh", "-n", str(wrapper)]).returncode == 0


class FakeLaunchd:
    """launchctl and the port, as install and uninstall see them. The agent's server holds the port while it is
    loaded; other holds it for a program that is not the agent."""

    def __init__(self, loaded: bool = False, other: bool = False, starts: bool = True) -> None:
        self.loaded = loaded
        self.other = other
        self.starts = starts
        self.calls: list[list[str]] = []

    def run(self, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(args)
        command = args[1]
        if command == "print":
            return subprocess.CompletedProcess(args, 0 if self.loaded else 113)
        if command == "bootout":
            self.loaded = False
        elif command == "bootstrap":
            self.loaded = True
        return subprocess.CompletedProcess(args, 0)

    def answers(self, _host: str, _port: int) -> bool:
        return self.other or (self.loaded and self.starts)

    def commands(self) -> list[str]:
        return [args[1] for args in self.calls if args[1] != "print"]


@pytest.fixture
def launchd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeLaunchd]:
    """A checkout with a .venv in tmp_path, the agent written under tmp_path, and launchctl faked."""
    root = tmp_path / "checkout"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").touch()
    fake = FakeLaunchd()
    monkeypatch.setattr(launch_agent, "ROOT", root)
    monkeypatch.setattr(launch_agent, "AGENT", tmp_path / "LaunchAgents" / f"{launch_agent.LABEL}.plist")
    monkeypatch.setattr("subprocess.run", fake.run)
    monkeypatch.setattr(launch_agent, "_answers", fake.answers)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr(launch_agent, "START_TIMEOUT", 0)
    monkeypatch.setattr(launch_agent, "FREE_TIMEOUT", 0)
    yield fake


def test_install_writes_the_agent_for_the_user_only_and_loads_it(
    launchd: FakeLaunchd, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "a-token-of-sixteen")
    assert launch_agent.install("127.0.0.1", 8765) == 0
    agent = launch_agent.AGENT
    assert stat.S_IMODE(agent.stat().st_mode) == 0o600  # it holds ACCESS_TOKEN
    job = plistlib.loads(agent.read_bytes())
    assert job["EnvironmentVariables"]["ACCESS_TOKEN"] == "a-token-of-sixteen"
    assert job["WorkingDirectory"] == str(launch_agent.ROOT)
    assert launchd.calls[-1] == ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(agent)]


def test_install_again_replaces_the_loaded_agent(launchd: FakeLaunchd) -> None:
    """The agent's own server is stopped first, so the port it frees is not taken for another program's."""
    launchd.loaded = True
    assert launch_agent.install("127.0.0.1", 9000) == 0
    assert launchd.commands() == ["bootout", "bootstrap"]
    assert plistlib.loads(launch_agent.AGENT.read_bytes())["EnvironmentVariables"]["PORT"] == "9000"


def test_install_refuses_a_port_another_program_holds(launchd: FakeLaunchd) -> None:
    launchd.other = True
    assert launch_agent.install("127.0.0.1", 8765) == 1
    assert not launch_agent.AGENT.exists()
    assert "bootstrap" not in launchd.commands()


def test_install_reports_a_server_that_does_not_start(launchd: FakeLaunchd) -> None:
    """The agent stays installed, so launchd keeps trying; the message points at the log."""
    launchd.starts = False
    assert launch_agent.install("127.0.0.1", 8765) == 1
    assert launch_agent.AGENT.exists() and launchd.loaded


def test_install_needs_the_venv(launchd: FakeLaunchd) -> None:
    (launch_agent.ROOT / ".venv" / "bin" / "python").unlink()
    assert launch_agent.install("127.0.0.1", 8765) == 1
    assert launchd.commands() == []


def test_uninstall_stops_and_removes_the_agent(launchd: FakeLaunchd) -> None:
    assert launch_agent.install("127.0.0.1", 8765) == 0
    assert launch_agent.uninstall() == 0
    assert not launch_agent.AGENT.exists() and not launchd.loaded
    # Nothing to do the second time
    assert launch_agent.uninstall() == 0
    assert launchd.commands() == ["bootstrap", "bootout"]


def test_only_on_macos(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    assert launch_agent.main(["install"]) == 1
    assert "only available on macOS" in capsys.readouterr().err
