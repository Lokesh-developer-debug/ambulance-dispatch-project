# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Dispatcher for Ambulance Dispatch Environment.

Handles automatic dispatching logic:
- Priority based dispatching (CRITICAL > MEDIUM > LOW)
- Nearest ambulance assignment
- Multi-pickup when 2 patients are close enough
- Nearest hospital with available beds/ICU
- Auto reassignment after delivery
"""

import math
from typing import List, Optional, Tuple, Dict, Any

from models import (
    Ambulance,
    AmbulanceStatus,
    AmbulanceDispatchAction,
    ActionType,
    Hospital,
    Patient,
    Severity,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MULTI_PICKUP_RADIUS = 15.0
SAFETY_MARGIN = 2
CRITICAL_THRESHOLD = 5
MEDIUM_THRESHOLD = 10
LOW_THRESHOLD = 20

DEATH_THRESHOLD = {
    Severity.CRITICAL: CRITICAL_THRESHOLD,
    Severity.MEDIUM: MEDIUM_THRESHOLD,
    Severity.LOW: LOW_THRESHOLD,
}

SEVERITY_PRIORITY = {
    Severity.CRITICAL: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
}


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _remaining_steps(patient: Patient) -> int:
    threshold = DEATH_THRESHOLD[patient.severity]
    return threshold - patient.waiting_steps


def _nearest_hospital(
    amb: Ambulance,
    hospitals: List[Hospital],
    needs_icu: bool
) -> Optional[Hospital]:
    valid_hospitals = []
    for h in hospitals:
        if needs_icu and h.available_icu > 0:
            valid_hospitals.append(h)
        elif not needs_icu and h.available_beds > 0:
            valid_hospitals.append(h)

    if not valid_hospitals:
        valid_hospitals = [h for h in hospitals if h.available_beds > 0 or h.available_icu > 0]

    if not valid_hospitals:
        return None

    return min(
        valid_hospitals,
        key=lambda h: _distance(amb.x, amb.y, h.x, h.y)
    )


def _can_multi_pickup(
    primary_patient: Patient,
    secondary_patient: Patient,
) -> bool:
    """
    Check if ambulance going to primary_patient can also pick up secondary_patient.

    Rules:
    1. Secondary patient must be within MULTI_PICKUP_RADIUS of primary patient
    2. Secondary patient must have enough time remaining (> SAFETY_MARGIN)
    3. If secondary is CRITICAL with waiting_steps >= 3 → deny (too risky)
    """
    dist = _distance(
        primary_patient.x, primary_patient.y,
        secondary_patient.x, secondary_patient.y
    )
    if dist > MULTI_PICKUP_RADIUS:
        return False

    remaining = _remaining_steps(secondary_patient)
    if remaining <= SAFETY_MARGIN:
        return False

    if secondary_patient.severity == Severity.CRITICAL and secondary_patient.waiting_steps >= 3:
        return False

    return True


def _get_active_patients(patients: List[Patient]) -> List[Patient]:
    return [
        p for p in patients
        if not p.is_picked_up and not p.is_delivered and not p.is_dead and p.is_visible
    ]


def _get_idle_ambulances(ambulances: List[Ambulance]) -> List[Ambulance]:
    return [a for a in ambulances if a.status == AmbulanceStatus.IDLE]


def _priority_score(patient: Patient) -> Tuple[int, int]:
    severity_score = SEVERITY_PRIORITY[patient.severity]
    urgency = -_remaining_steps(patient)
    return (severity_score, urgency)


def compute_dispatch_actions(
    ambulances: List[Ambulance],
    patients: List[Patient],
    hospitals: List[Hospital],
) -> List[Dict[str, Any]]:
    """
    Core smart dispatch logic.

    Priority rules:
    - CRITICAL patients MUST be assigned first
    - MEDIUM patients next
    - LOW patients only when no CRITICAL or MEDIUM waiting
    - Nearest ambulance always assigned to highest priority patient
    - Never dispatch to patients already being handled
    - Multi-pickup sets secondary_patient_id on ambulance directly
    """
    actions = []
    active_patients = _get_active_patients(patients)
    idle_ambulances = _get_idle_ambulances(ambulances)

    if not active_patients or not idle_ambulances:
        return actions

    # Remove patients already assigned to en_route ambulances
    already_assigned_patient_ids = {
        a.assigned_patient_id
        for a in ambulances
        if a.assigned_patient_id is not None
        and a.status != AmbulanceStatus.IDLE
    }

    # Also include secondary patients already assigned
    already_assigned_patient_ids.update({
        a.secondary_patient_id
        for a in ambulances
        if a.secondary_patient_id is not None
        and a.status != AmbulanceStatus.IDLE
    })

    active_patients = [
        p for p in active_patients
        if p.id not in already_assigned_patient_ids
    ]

    if not active_patients:
        return actions

    # Separate patients by severity
    critical_patients = [p for p in active_patients if p.severity == Severity.CRITICAL]
    medium_patients = [p for p in active_patients if p.severity == Severity.MEDIUM]
    low_patients = [p for p in active_patients if p.severity == Severity.LOW]

    if critical_patients:
        priority_patients = sorted(critical_patients, key=_priority_score, reverse=True)
        secondary_patients = medium_patients + low_patients
    elif medium_patients:
        priority_patients = sorted(medium_patients, key=_priority_score, reverse=True)
        secondary_patients = low_patients
    else:
        priority_patients = sorted(low_patients, key=_priority_score, reverse=True)
        secondary_patients = []

    assigned_patient_ids = set()
    assigned_ambulance_ids = set()

    for patient in priority_patients:
        if patient.id in assigned_patient_ids:
            continue

        available_ambulances = [
            a for a in idle_ambulances
            if a.id not in assigned_ambulance_ids
        ]
        if not available_ambulances:
            break

        nearest_amb = min(
            available_ambulances,
            key=lambda a: _distance(a.x, a.y, patient.x, patient.y)
        )

        # Check for multi-pickup opportunity from secondary patients
        multi_pickup_patient = None
        for other_patient in secondary_patients:
            if other_patient.id in assigned_patient_ids:
                continue
            if other_patient.id in already_assigned_patient_ids:
                continue
            if _can_multi_pickup(patient, other_patient):
                multi_pickup_patient = other_patient
                break

        # If multi-pickup planned → set secondary_patient_id on ambulance directly
        if multi_pickup_patient:
            nearest_amb.secondary_patient_id = multi_pickup_patient.id
        else:
            nearest_amb.secondary_patient_id = None

        actions.append({
            "action_type": ActionType.DISPATCH,
            "ambulance_id": nearest_amb.id,
            "patient_id": patient.id,
            "hospital_id": None,
            "multi_pickup_patient_id": multi_pickup_patient.id if multi_pickup_patient else None,
        })

        assigned_patient_ids.add(patient.id)
        assigned_ambulance_ids.add(nearest_amb.id)

        if multi_pickup_patient:
            assigned_patient_ids.add(multi_pickup_patient.id)

    # Assign remaining idle ambulances to secondary patients
    remaining_ambulances = [
        a for a in idle_ambulances
        if a.id not in assigned_ambulance_ids
    ]
    remaining_patients = [
        p for p in secondary_patients
        if p.id not in assigned_patient_ids
        and p.id not in already_assigned_patient_ids
    ]

    for patient in sorted(remaining_patients, key=_priority_score, reverse=True):
        if not remaining_ambulances:
            break

        nearest_amb = min(
            remaining_ambulances,
            key=lambda a: _distance(a.x, a.y, patient.x, patient.y)
        )

        nearest_amb.secondary_patient_id = None

        actions.append({
            "action_type": ActionType.DISPATCH,
            "ambulance_id": nearest_amb.id,
            "patient_id": patient.id,
            "hospital_id": None,
            "multi_pickup_patient_id": None,
        })

        assigned_patient_ids.add(patient.id)
        remaining_ambulances.remove(nearest_amb)

    return actions


def compute_hospital_routing(
    ambulances: List[Ambulance],
    patients: List[Patient],
    hospitals: List[Hospital],
) -> List[Dict[str, Any]]:
    """
    For ambulances that have picked up patients, route them to nearest hospital.
    Waits for secondary patient pickup before routing to hospital.
    """
    actions = []

    for amb in ambulances:
        if amb.status != AmbulanceStatus.EN_ROUTE_PATIENT:
            continue
        if amb.assigned_patient_id is None:
            continue

        # Find the primary patient
        patient = next((p for p in patients if p.id == amb.assigned_patient_id), None)
        if patient is None or not patient.is_picked_up:
            continue

        # If secondary patient exists and not yet picked up → wait
        if amb.secondary_patient_id is not None:
            secondary = next((p for p in patients if p.id == amb.secondary_patient_id), None)
            if secondary and not secondary.is_picked_up and not secondary.is_dead:
                continue  # Wait until secondary is picked up or dead

        # Find nearest hospital
        needs_icu = patient.severity == Severity.CRITICAL
        hospital = _nearest_hospital(amb, hospitals, needs_icu)
        if hospital is None:
            continue

        actions.append({
            "action_type": ActionType.ROUTE_TO_HOSPITAL,
            "ambulance_id": amb.id,
            "patient_id": None,
            "hospital_id": hospital.id,
            "multi_pickup_patient_id": None,
        })

    return actions