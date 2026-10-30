"""Create a "Minutes" shortcut on the Desktop that launches run.pyw with pythonw.

    python make_shortcut.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minutes.ui.startup import SHORTCUT_NAME, create_shortcut  # noqa: E402


def desktop_folder() -> Path:
    import win32com.client

    # Honors a Desktop redirected to OneDrive.
    return Path(win32com.client.Dispatch("WScript.Shell").SpecialFolders("Desktop"))


def main() -> None:
    path = create_shortcut(desktop_folder(), SHORTCUT_NAME)
    print(f"Created {path}")


if __name__ == "__main__":
    main()
