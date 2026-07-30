# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pydantic import BaseModel, ConfigDict


class CvarData(BaseModel):
    """
    Data structure holding all scenario and statistical information required
    for CVaR optimization.

    Attributes
    ----------
    mean : np.ndarray
        Array of shape (n_assets,) of expected asset returns.
    R : np.ndarray
        Scenario deviations, shape (num_scenarios, n_assets),
        each row is asset return deviation for a scenario.
    p : np.ndarray
        Scenario probabilities, shape (num_scenarios,), summing to 1.

    Examples
    --------
    >>> import numpy as np
    >>> # Create data for 3 assets with 5 scenarios
    >>> mean = np.array([0.08, 0.10, 0.12])
    >>> R = np.array([
    ...     [-0.02, 0.01, 0.03],
    ...     [0.01, -0.01, 0.02],
    ...     [0.03, 0.02, -0.01],
    ...     [-0.01, 0.02, 0.01],
    ...     [0.02, -0.02, 0.00]
    ... ])
    >>> p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> data = CvarData(mean=mean, R=R, p=p)
    >>> print(data.mean.shape)  # (3,)
    >>> print(data.R.shape)  # (5, 3)
    >>> print(data.p.sum())  # 1.0

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: np.ndarray
    R: np.ndarray
    p: np.ndarray
