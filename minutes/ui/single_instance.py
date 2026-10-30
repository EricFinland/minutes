"""One app per data folder.

A named mutex decides who is first (atomic, released by the OS if we crash). The
first instance also listens on a local socket so a second launch can ask it to show
its window before exiting.
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger(__name__)

ERROR_ALREADY_EXISTS = 183


def instance_key(data_dir: Path) -> str:
    digest = hashlib.sha1(str(Path(data_dir).resolve()).lower().encode("utf-8")).hexdigest()
    return f"Minutes-{digest[:12]}"


class SingleInstance(QObject):
    activated = pyqtSignal()

    def __init__(self, key: str, parent: QObject | None = None):
        super().__init__(parent)
        self.key = key
        self._mutex = None
        self._server: QLocalServer | None = None

    def acquire(self) -> bool:
        """True if this process is the primary instance. Otherwise pings the primary."""
        if not self._take_mutex():
            self.notify_primary()
            return False
        QLocalServer.removeServer(self.key)
        self._server = QLocalServer(self)
        if not self._server.listen(self.key):
            log.warning("single instance server failed: %s", self._server.errorString())
        self._server.newConnection.connect(self._on_connection)
        return True

    def _take_mutex(self) -> bool:
        if sys.platform != "win32":
            return True
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        handle = kernel32.CreateMutexW(None, False, "Local\\" + self.key)
        already = ctypes.get_last_error() == ERROR_ALREADY_EXISTS
        if not handle:
            return True
        if already:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            return False
        self._mutex = handle  # held for the life of the process
        return True

    def notify_primary(self) -> bool:
        sock = QLocalSocket()
        sock.connectToServer(self.key)
        if not sock.waitForConnected(1500):
            return False
        sock.write(b"show\n")
        sock.flush()
        sock.waitForBytesWritten(1500)
        sock.disconnectFromServer()
        return True

    def _on_connection(self) -> None:
        # Any connection is a "show yourself" request. Reading a payload is not
        # reliable on Windows when the client disconnects right after writing.
        while self._server is not None and self._server.hasPendingConnections():
            conn = self._server.nextPendingConnection()
            conn.disconnected.connect(conn.deleteLater)
            conn.disconnectFromServer()
        self.activated.emit()

    def release(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._mutex:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._mutex))
            self._mutex = None
