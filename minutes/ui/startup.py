"""Start with Windows: a shortcut to `pythonw.exe run.pyw` in the user's Startup folder."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

SHORTCUT_NAME = "Minutes.lnk"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def launcher_script() -> Path:
    return project_root() / "run.pyw"


def pythonw_path() -> Path:
    exe = Path(sys.executable)
    candidate = exe.with_name("pythonw.exe")
    return candidate if candidate.exists() else exe


def startup_folder() -> Path:
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


def is_enabled(folder: Optional[Path] = None) -> bool:
    return (Path(folder or startup_folder()) / SHORTCUT_NAME).exists()


def set_enabled(enabled: bool, folder: Optional[Path] = None) -> bool:
    folder = Path(folder or startup_folder())
    path = folder / SHORTCUT_NAME
    if enabled:
        create_shortcut(folder)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return is_enabled(folder)
