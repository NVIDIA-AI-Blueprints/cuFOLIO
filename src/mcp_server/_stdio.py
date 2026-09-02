# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect the MCP stdio transport from plotting and native-library output."""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import sys
import threading

os.environ.setdefault("MPLBACKEND", "Agg")

_STDOUT_LOCK = threading.Lock()


@contextlib.contextmanager
def suppress_stdout():
    """Suppress Python and file-descriptor writes to stdout during compute."""

    with _STDOUT_LOCK:
        try:
            saved_fd = os.dup(1)
        except OSError:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
            return
        try:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                sys.stdout.flush()
                os.dup2(sink.fileno(), 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    yield
                try:
                    ctypes.CDLL(None).fflush(None)
                except Exception:
                    pass
        finally:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
