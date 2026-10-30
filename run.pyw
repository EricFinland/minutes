"""Windowless launcher for Minutes.

pythonw.exe gives the process NULL stdin/stdout/stderr handles, and ctranslate2 (the
Whisper backend) crashes when it writes to them. So this script starts python.exe
with app.py under a hidden console: the child gets real handles, its output goes to
<data dir>/minutes.log, and no window ever appears.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app.py")
PYTHON = os.path.join(os.path.dirname(sys.executable), "python.exe")
if not os.path.exists(PYTHON):
    PYTHON = "python.exe"

sys.path.insert(0, HERE)
from minutes import config  # noqa: E402

LOG = os.path.join(str(config.data_dir()), "minutes.log")
MAX_LOG_BYTES = 2_000_000


def _rotate_log() -> None:
    try:
        if os.path.getsize(LOG) > MAX_LOG_BYTES:
            os.replace(LOG, LOG + ".old")
    except OSError:
        pass


def _main_posix() -> None:
    """macOS: start app.py detached so the Terminal window that ran this can close."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(LOG, "a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, APP, *sys.argv[1:]], cwd=HERE,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)


def main() -> None:
    _rotate_log()
    if sys.platform != "win32":
        _main_posix()
        return
    # STARTF_USESHOWWINDOW + SW_HIDE: python.exe still owns a real console (valid fd
    # 0/1/2 handles), it is just never shown.
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"  # log lines reach the file even if the process dies

    with open(LOG, "a", encoding="utf-8") as log:
        subprocess.Popen(
            [PYTHON, APP, *sys.argv[1:]],
            cwd=HERE,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            startupinfo=si,
            env=env,
        )


if __name__ == "__main__":
    main()
