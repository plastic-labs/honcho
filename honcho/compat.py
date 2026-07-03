"""
Compatibility layer and utilities, mostly for proper Windows and Python 3
support.
"""
import errno
import os
import signal
import sys

# This works for both 32 and 64 bit Windows
ON_WINDOWS = 'win32' in str(sys.platform).lower()

if ON_WINDOWS:
    import ctypes


class ProcessManager(object):
    if ON_WINDOWS:
        def terminate(self, pid):
            # The first argument to OpenProcess represents the desired access
            # to the process. 1 represents the PROCESS_TERMINATE access right.
            handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)
            ctypes.windll.kernel32.TerminateProcess(handle, -1)
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        def terminate(self, pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError as e:
                if e.errno not in [errno.EPERM, errno.ESRCH]:
                    raise

    if ON_WINDOWS:
        def kill(self, pid):
            # There's no SIGKILL on Win32...
            self.terminate(pid)
    else:
        def kill(self, pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError as e:
                if e.errno not in [errno.EPERM, errno.ESRCH]:
                    raise
