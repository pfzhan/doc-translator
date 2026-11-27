"""The launchd agent that keeps the server running on macOS (scripts/launch_agent.py)."""

import os
import plistlib
import subprocess
from pathlib import Path

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
