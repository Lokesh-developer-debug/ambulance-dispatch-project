# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Ambulance Dispatch Environment.

Models the real-world 108 emergency dispatch problem:
- Ambulances with location and status
- Patients with severity and visibility (partial observability)
- Hospitals with bed and ICU availability
- Actions: DISPATCH, ROUTE_TO_HOSPITAL, WAIT
- Observations: agent's view of the world each step
"""

from enum import Enum
from typing import List, Optional

from pydantic import Field, BaseModel
class AmbulanceStatus(str, Enum):
    """Possible states an ambulance can be in."""
    IDLE = "idle"
    EN_ROUTE_PATIENT = "en_route_patient"
    EN_ROUTE_HOSPITAL = "en_route_hospital"

class Severity(str, Enum):
    """Patient severity levels - determines urgency and death threshold."""
    LOW = "low"
    MEDIUM = "medium"
    CRITICAL = "critical"

class Ambulance(BaseModel):
    """Represents one ambulance vehicle on the map."""

    id: int = Field(..., description="Unique ambulance identifier")
    x: float = Field(..., description="Current X position on map")
    y: float = Field(..., description="Current Y position on map")
    status: AmbulanceStatus = Field(
        default=AmbulanceStatus.IDLE,
        description="Current status of ambulance"
    )
    assigned_patient_id: Optional[int] = Field(
        default=None,
        description="Primary patient this ambulance is going to"
    )
    secondary_patient_id: Optional[int] = Field(
        default=None,
        description="Secondary patient for multi-pickup. None if not doing multi-pickup"
    )
    assigned_hospital_id: Optional[int] = Field(
        default=None,
        description="Hospital this ambulance is heading to"
    )

class Patient(BaseModel):
    """Represents one patient needing emergency help."""

    id: int = Field(..., description="Unique patient identifier")
    x: float = Field(..., description="Patient X position on map")
    y: float = Field(..., description="Patient Y position on map")
    severity: Severity = Field(..., description="How critical this patient is")
    waiting_steps: int = Field(
        default=0,
        description="How many steps this patient has been waiting"
    )
    is_picked_up: bool = Field(
        default=False,
        description="True when ambulance has reached this patient"
    )
    is_delivered: bool = Field(
        default=False,
        description="True when patient has been delivered to hospital"
    )
    is_dead: bool = Field(
        default=False,
        description="True if patient waited too long and died"
    )
    is_visible: bool = Field(
        default=False,
        description="True if agent can see this patient (partial observability)"
    )

class Hospital(BaseModel):
    """Represents one hospital where patients get delivered."""

    id: int = Field(..., description="Unique hospital identifier")
    x: float = Field(..., description="Hospital X position on map")
    y: float = Field(..., description="Hospital Y position on map")
    available_beds: int = Field(
        default=10,
        description="Number of regular beds available"
    )
    available_icu: int = Field(
        default=2,
        description="Number of ICU beds available"
    )
    incoming_ambulances: int = Field(
        default=0,
        description="How many ambulances are currently heading to this hospital"
    )

class ActionType(str, Enum):
    """Possible action types the agent can take."""
    DISPATCH = "dispatch"
    ROUTE_TO_HOSPITAL = "route_to_hospital"
    WAIT = "wait"


class AmbulanceDispatchAction(BaseModel):
    """What the agent decides each step."""

    action_type: ActionType = Field(..., description="Type of action to take")
    ambulance_id: Optional[int] = Field(
        default=None,
        description="Which ambulance to act on. None for WAIT"
    )
    patient_id: Optional[int] = Field(
        default=None,
        description="Which patient to dispatch to. Only for DISPATCH"
    )
    hospital_id: Optional[int] = Field(
        default=None,
        description="Which hospital to route to. Only for ROUTE_TO_HOSPITAL"
    )

class AmbulanceDispatchObservation(BaseModel):
    """What the agent sees every step."""

    ambulances: List[Ambulance] = Field(
        default_factory=list,
        description="All ambulances with current status"
    )
    patients: List[Patient] = Field(
        default_factory=list,
        description="Only visible patients (partial observability)"
    )
    hospitals: List[Hospital] = Field(
        default_factory=list,
        description="All hospitals with current bed availability"
    )
    time_step: int = Field(
        default=0,
        description="Current step number"
    )
    max_steps: int = Field(
        default=60,
        description="Maximum steps in this episode"
    )
    total_patients_spawned: int = Field(
        default=0,
        description="Total patients spawned so far including dynamic"
    )
    current_phase: str = Field(
        default="early",
        description="Current episode phase: early, middle, late"
    )