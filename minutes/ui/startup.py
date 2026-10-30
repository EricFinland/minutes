"""Start at login.

Windows: a shortcut to `pythonw.exe run.pyw` in the user's Startup folder.
macOS: a LaunchAgent plist in ~/Library/LaunchAgents that runs app.py at login.
"""
from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path
from typing import Optional

SHORTCUT_NAME = "Minutes.lnk"
LAUNCH_AGENT_LABEL = "com.minutes.recorder"
LAUNCH_AGENT_NAME = LAUNCH_AGENT_LABEL + ".plist"


def label() -> str:
    return "Open at login" if sys.platform == "darwin" else "Start with Windows"


def _entry_name() -> str:
    return LAUNCH_AGENT_NAME if sys.platform == "darwin" else SHORTCUT_NAME


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def launcher_script() -> Path:
    return project_root() / "run.pyw"


def pythonw_path() -> Path:
    exe = Path(sys.executable)
    candidate = exe.with_name("pythonw.exe")
    return candidate if candidate.exists() else exe


def startup_folder() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents"
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def create_shortcut(folder: Path, name: str = SHORTCUT_NAME) -> Path:
    """Write a .lnk that launches the app without a console window."""
    import win32com.client

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    shell = win32com.client.Dispatch("WScript.Shell")
    sc = shell.CreateShortcut(str(path))
    sc.TargetPath = str(pythonw_path())
    sc.Arguments = f'"{launcher_script()}"'
    sc.WorkingDirectory = str(project_root())
    sc.Description = "Minutes meeting transcriber"
    sc.IconLocation = f"{pythonw_path()},0"
    sc.Save()
    return path


def read_shortcut(path: Path) -> dict:
    import win32com.client

    sc = win32com.client.Dispatch("WScript.Shell").CreateShortcut(str(path))
    return {"target": sc.TargetPath, "arguments": sc.Arguments, "workdir": sc.WorkingDirectory}


def write_launch_agent(folder: Path, python: Optional[str] = None,
                       log_path: Optional[Path] = None) -> Path:
    """Write the macOS LaunchAgent that starts Minutes at login."""
    from minutes import config

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    log_file = str(log_path or config.data_dir() / "minutes.log")
    agent = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [python or sys.executable, str(project_root() / "app.py")],
        "WorkingDirectory": str(project_root()),
        "RunAtLoad": True,
        "ProcessType": "Interactive",
        "StandardOutPath": log_file,
        "StandardErrorPath": log_file,
    }
    path = folder / LAUNCH_AGENT_NAME
    with open(path, "wb") as f:
        plistlib.dump(agent, f)
    return path


def is_enabled(folder: Optional[Path] = None) -> bool:
    return (Path(folder or startup_folder()) / _entry_name()).exists()


def set_enabled(enabled: bool, folder: Optional[Path] = None) -> bool:
    folder = Path(folder or startup_folder())
    path = folder / _entry_name()
    if enabled and sys.platform == "darwin":
        write_launch_agent(folder)
    elif enabled:
        create_shortcut(folder)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return is_enabled(folder)
