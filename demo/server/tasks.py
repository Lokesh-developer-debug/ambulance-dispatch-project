# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions for the Ambulance Dispatch Environment.

Defines preset scenarios of varying difficulty.
"""

from typing import Any, Dict, List


TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "ambulance_dispatch_easy",
        "description": (
            "Easy dispatch scenario: 3 ambulances, 2 hospitals, "
            "4 patients, 60 steps. Good for initial testing."
        ),
        "env_config": {
            "num_ambulances": 3,
            "num_hospitals": 2,
            "num_patients": 4,
            "max_steps": 60,
            "seed": 42,
        },
        "success_criteria": {
            "min_delivery_rate": 0.75,
            "max_deaths": 1,
        },
    },
    {
        "task_id": "ambulance_dispatch_medium",
        "description": (
            "Medium dispatch scenario: 3 ambulances, 3 hospitals, "
            "6 patients, 60 steps."
        ),
        "env_config": {
            "num_ambulances": 3,
            "num_hospitals": 3,
            "num_patients": 6,
            "max_steps": 60,
            "seed": None,
        },
        "success_criteria": {
            "min_delivery_rate": 0.80,
            "max_deaths": 1,
        },
    },
    {
        "task_id": "ambulance_dispatch_hard",
        "description": (
            "Hard dispatch scenario: 2 ambulances, 2 hospitals, "
            "8 patients, 50 steps. Tests prioritisation under pressure."
        ),
        "env_config": {
            "num_ambulances": 2,
            "num_hospitals": 2,
            "num_patients": 8,
            "max_steps": 50,
            "seed": None,
        },
        "success_criteria": {
            "min_delivery_rate": 0.85,
            "max_deaths": 0,
        },
    },
]


def get_task(task_id: str) -> Dict[str, Any]:
    """Retrieve a task config by its task_id."""
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Task '{task_id}' not found. Available: {[t['task_id'] for t in TASKS]}")