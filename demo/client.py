# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ambulance Dispatch Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AmbulanceDispatchAction, AmbulanceDispatchObservation


class AmbulanceDispatchEnv(
    EnvClient[AmbulanceDispatchAction, AmbulanceDispatchObservation, State]
):
    """
    Client for the Ambulance Dispatch Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AmbulanceDispatchEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.ambulances)
        ...
        ...     result = client.step(AmbulanceDispatchAction(
        ...         action_type="dispatch",
        ...         ambulance_id=0,
        ...         patient_id=0
        ...     ))
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AmbulanceDispatchEnv.from_docker_image("ambulance_dispatch-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AmbulanceDispatchAction(
        ...         action_type="dispatch",
        ...         ambulance_id=0,
        ...         patient_id=0
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AmbulanceDispatchAction) -> Dict:
        """
        Convert AmbulanceDispatchAction to JSON payload for step message.

        Args:
            action: AmbulanceDispatchAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value,
            "ambulance_id": action.ambulance_id,
            "patient_id": action.patient_id,
            "hospital_id": action.hospital_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AmbulanceDispatchObservation]:
        """
        Parse server response into StepResult[AmbulanceDispatchObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AmbulanceDispatchObservation
        """
        obs_data = payload.get("observation", {})
        observation = AmbulanceDispatchObservation(
            ambulances=obs_data.get("ambulances", []),
            patients=obs_data.get("patients", []),
            hospitals=obs_data.get("hospitals", []),
            time_step=obs_data.get("time_step", 0),
            max_steps=obs_data.get("max_steps", 60),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )