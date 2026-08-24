"""Own benchmark subprocess trees with a Windows kill-on-close Job Object."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
# CPython does not expose this Win32 creation flag on every supported build.
# A worker must start suspended so it cannot create descendants before the
# benchmark assigns it to the kill-on-close Job Object.
CREATE_SUSPENDED = 0x00000004


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_ulonglong),
        ("write_operation_count", ctypes.c_ulonglong),
        ("other_operation_count", ctypes.c_ulonglong),
        ("read_transfer_count", ctypes.c_ulonglong),
        ("write_transfer_count", ctypes.c_ulonglong),
        ("other_transfer_count", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_longlong),
        ("per_job_user_time_limit", ctypes.c_longlong),
        ("limit_flags", wintypes.DWORD),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _BasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class WindowsKillOnCloseJob:
    """Keep one process tree in a Job Object that dies when the handle closes."""

    def __init__(self) -> None:
        self._handle: int | None = None
        if os.name != "nt":
            raise RuntimeError("Windows Job Objects are unavailable on this host")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise RuntimeError("Windows Job Object creation failed")
        self._kernel32 = kernel32
        self._ntdll = ntdll
        self._handle = int(handle)
        information = _ExtendedLimitInformation()
        information.basic_limit_information.limit_flags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            wintypes.HANDLE(self._handle),
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            self._close_without_raising()
            raise RuntimeError("Windows Job Object safety policy setup failed")

    def assign(self, process) -> None:
        if self._handle is None:
            raise RuntimeError("Windows Job Object is already closed")
        process_handle = getattr(process, "_handle", None)
        if process_handle is None or not self._kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(self._handle),
            wintypes.HANDLE(int(process_handle)),
        ):
            raise RuntimeError("benchmark worker Job Object assignment failed")

    def close(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        if not self._kernel32.CloseHandle(wintypes.HANDLE(handle)):
            raise RuntimeError("benchmark worker Job Object close failed")
        self._handle = None

    def resume(self, process) -> None:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None or self._ntdll.NtResumeProcess(
            wintypes.HANDLE(int(process_handle))
        ) != 0:
            raise RuntimeError("suspended benchmark worker could not be resumed")

    def _close_without_raising(self) -> None:
        handle = getattr(self, "_handle", None)
        kernel32 = getattr(self, "_kernel32", None)
        if handle is None or kernel32 is None:
            return
        kernel32.CloseHandle(wintypes.HANDLE(handle))
        self._handle = None

    def __del__(self) -> None:
        self._close_without_raising()
