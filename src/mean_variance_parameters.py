# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import field_validator

from .base_parameters import BaseParameters


class MeanVarianceParameters(BaseParameters):
    """
    User-tunable parameters and constraint limits for Mean-Variance optimization.

    Extends BaseParameters with a Mean-Variance-specific optional hard limit
    on portfolio variance (``var_limit``).
    """

    # Mean-Variance-specific field
    var_limit: Optional[float] = None

    @field_validator("var_limit")
    @classmethod
    def validate_var_limit(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("Variance limit must be positive.")
        return value
