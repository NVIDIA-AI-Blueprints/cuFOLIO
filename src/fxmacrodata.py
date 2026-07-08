# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FXMacroData release-calendar helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def load_fxmacrodata_calendar(
    currency: str = "usd",
    *,
    limit: int = 100,
    min_tier: Optional[int] = 2,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Load official macro release events for portfolio risk overlays."""

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    token = api_key or os.getenv("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{FXMACRODATA_BASE_URL}/calendar/{currency.lower()}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "cufolio-fxmacrodata/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)

    rows: list[dict[str, Any]] = payload.get("data", [])
    if min_tier is not None:
        rows = [
            row
            for row in rows
            if int(row.get("market_tier") or 99) <= min_tier
        ]

    frame = pd.DataFrame(rows[:limit_count])
    if not frame.empty and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
