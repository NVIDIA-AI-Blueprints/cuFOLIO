# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional local stdio MCP integration for portfolio optimization."""

from .server import create_server

__all__ = ["create_server"]
