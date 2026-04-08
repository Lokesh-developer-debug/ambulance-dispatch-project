# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Core environment logic for the Ambulance Dispatch Environment.

Simulates the 108 emergency dispatch problem with:
- Partial observability (patients become visible within radius)
- Ambulance movement toward targets each step
- Patient death if waiting too long
- Hospital bed management
- Reward shaping based on outcomes
- Dynamic patient spawning (early/middle/late phases)
- Cluster spawning (accident scenes)
- Multi-pickup support
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Ambulance,
    AmbulanceDispatchAction,
    AmbulanceDispatchObservation,
    AmbulanceStatus,
    Hospital,
    Patient,
    Severity,
    ActionType,
)


# ── Tunable constants ──────────────────────────────────────────────────────────
MAP_SIZE = 100.0
AMBULANCE_SPEED = 5.0
VISIBILITY_RADIUS = 100.0

# Death thresholds
DEATH_THRESHOLD: Dict[Severity, int] = {
    Severity.CRITICAL: 5,
    Severity.MEDIUM: 10,
    Severity.LOW: 20,
}

# Reward values
REWARD_DELIVERED_CRITICAL = 30
REWARD_DELIVERED_MEDIUM = 20
REWARD_DELIVERED_LOW = 10
PENALTY_DEATH = -50
PENALTY_NO_BED = -10
STEP_PENALTY = -1

# ── Dynamic spawning constants ─────────────────────────────────────────────────
SPAWN_INTERVAL = 3

SPAWN_PROBABILITY = {
    "early":  0.2,
    "middle": 0.5,
    "late":   0.7,
}

SEVERITY_WEIGHTS = {
    "early":  {Severity.LOW: 0.5, Severity.MEDIUM: 0.4, Severity.CRITICAL: 0.1},
    "middle": {Severity.LOW: 0.2, Severity.MEDIUM: 0.5, Severity.CRITICAL: 0.3},
    "late":   {Severity.LOW: 0.1, Severity.MEDIUM: 0.4, Severity.CRITICAL: 0.5},
}

CLUSTER_PROBABILITY = 0.1
CLUSTER_SIZE_MIN = 2
CLUSTER_SIZE_MAX = 3
CLUSTER_RADIUS = 10.0
MAX_TOTAL_PATIENTS = 20


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _move_toward(
    ax: float, ay: float, tx: float, ty: float, speed: float
) -> Tuple[float, float]:
    dist = _distance(ax, ay, tx, ty)
    if dist <= speed:
        return tx, ty
    ratio = speed / dist
    return ax + ratio * (tx - ax), ay + ratio * (ty - ay)


def _get_phase(time_step: int, max_steps: int) -> str:
    """Determine current episode phase based on time step."""
    progress = time_step / max_steps
    if progress < 0.33:
        return "early"
    elif progress < 0.66:
        return "middle"
    else:
        return "late"


class AmbulanceDispatchEnvironment:
    """
    Environment for ambulance dispatch simulation with dynamic spawning
    and multi-pickup support.
    """

    def __init__(
        self,
        num_ambulances: int = 3,
        num_hospitals: int = 2,
        num_patients: int = 6,
        max_steps: int = 60,
        seed: Optional[int] = None,
        enable_dynamic_spawning: bool = True,
        max_total_patients: int = MAX_TOTAL_PATIENTS,
    ) -> None:
        self.num_ambulances = num_ambulances
        self.num_hospitals = num_hospitals
        self.num_patients = num_patients
        self.max_steps = max_steps
        self.seed = seed
        self.enable_dynamic_spawning = enable_dynamic_spawning
        self.max_total_patients = max_total_patients

        self.ambulances: List[Ambulance] = []
        self.hospitals: List[Hospital] = []
        self.patients: List[Patient] = []
        self.time_step: int = 0
        self._next_patient_id: int = 0
        self._rng = random.Random(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> AmbulanceDispatchObservation:
        self._rng = random.Random(self.seed)
        self.time_step = 0
        self._next_patient_id = 0

        self.hospitals = self._spawn_hospitals()
        self.ambulances = self._spawn_ambulances()
        self.patients = self._spawn_patients()

        self._update_visibility()
        return self._build_observation()

    def step(
        self, action: AmbulanceDispatchAction
    ) -> Tuple[AmbulanceDispatchObservation, float, bool, Dict[str, Any]]:
        reward = STEP_PENALTY
        info: Dict[str, Any] = {}

        action_reward, action_info = self._apply_action(action)
        reward += action_reward
        info.update(action_info)

        self._move_ambulances()

        arrival_reward, arrival_info = self._check_arrivals()
        reward += arrival_reward
        info.update(arrival_info)

        death_penalty, death_info = self._tick_patients()
        reward += death_penalty
        info.update(death_info)

        if self.enable_dynamic_spawning:
            spawn_info = self._maybe_spawn_patients()
            info.update(spawn_info)

        self._update_visibility()
        self.time_step += 1

        done = self._is_done()
        obs = self._build_observation()
        return obs, reward, done, info

    def step_multiple(
        self, actions: List[AmbulanceDispatchAction]
    ) -> Tuple[AmbulanceDispatchObservation, float, bool, Dict[str, Any]]:
        reward = STEP_PENALTY
        info: Dict[str, Any] = {}

        for action in actions:
            action_reward, action_info = self._apply_action(action)
            reward += action_reward
            info.update(action_info)

        self._move_ambulances()

        arrival_reward, arrival_info = self._check_arrivals()
        reward += arrival_reward
        info.update(arrival_info)

        death_penalty, death_info = self._tick_patients()
        reward += death_penalty
        info.update(death_info)

        if self.enable_dynamic_spawning:
            spawn_info = self._maybe_spawn_patients()
            info.update(spawn_info)

        self._update_visibility()
        self.time_step += 1

        done = self._is_done()
        obs = self._build_observation()
        return obs, reward, done, info

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "action_type": [a.value for a in ActionType],
            "ambulance_id": list(range(self.num_ambulances)),
            "patient_id": list(range(self.max_total_patients)),
            "hospital_id": list(range(self.num_hospitals)),
        }

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "ambulances": self.num_ambulances,
            "patients": self.max_total_patients,
            "hospitals": self.num_hospitals,
            "map_size": MAP_SIZE,
            "dynamic_spawning": self.enable_dynamic_spawning,
        }

    # ── Spawning helpers ───────────────────────────────────────────────────────

    def _spawn_hospitals(self) -> List[Hospital]:
        hospitals = []
        for i in range(self.num_hospitals):
            hospitals.append(
                Hospital(
                    id=i,
                    x=self._rng.uniform(10, MAP_SIZE - 10),
                    y=self._rng.uniform(10, MAP_SIZE - 10),
                    available_beds=self._rng.randint(8, 15),
                    available_icu=self._rng.randint(2, 5),
                )
            )
        return hospitals

    def _spawn_ambulances(self) -> List[Ambulance]:
        ambulances = []
        for i in range(self.num_ambulances):
            hospital = self._rng.choice(self.hospitals)
            ambulances.append(
                Ambulance(
                    id=i,
                    x=hospital.x + self._rng.uniform(-5, 5),
                    y=hospital.y + self._rng.uniform(-5, 5),
                    status=AmbulanceStatus.IDLE,
                )
            )
        return ambulances

    def _spawn_patients(self) -> List[Patient]:
        severities = [Severity.CRITICAL, Severity.MEDIUM, Severity.LOW]
        weights = [0.3, 0.4, 0.3]
        patients = []
        for i in range(self.num_patients):
            patients.append(
                Patient(
                    id=self._next_patient_id,
                    x=self._rng.uniform(0, MAP_SIZE),
                    y=self._rng.uniform(0, MAP_SIZE),
                    severity=self._rng.choices(severities, weights=weights)[0],
                    is_visible=False,
                )
            )
            self._next_patient_id += 1
        return patients

    def _maybe_spawn_patients(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}

        if self.time_step % SPAWN_INTERVAL != 0:
            return info

        if len(self.patients) >= self.max_total_patients:
            return info

        phase = _get_phase(self.time_step, self.max_steps)
        spawn_prob = SPAWN_PROBABILITY[phase]

        if self._rng.random() > spawn_prob:
            return info

        if self._rng.random() < CLUSTER_PROBABILITY:
            new_patients = self._spawn_cluster(phase)
            info["cluster_spawn"] = True
        else:
            new_patients = [self._spawn_single_patient(phase)]
            info["cluster_spawn"] = False

        for patient in new_patients:
            if len(self.patients) < self.max_total_patients:
                self.patients.append(patient)
                info[f"new_patient_{patient.id}_spawned"] = {
                    "severity": patient.severity.value,
                    "phase": phase,
                    "position": (round(patient.x, 2), round(patient.y, 2))
                }

        return info

    def _spawn_single_patient(self, phase: str) -> Patient:
        weights = SEVERITY_WEIGHTS[phase]
        severities = list(weights.keys())
        weights_list = list(weights.values())

        patient = Patient(
            id=self._next_patient_id,
            x=self._rng.uniform(0, MAP_SIZE),
            y=self._rng.uniform(0, MAP_SIZE),
            severity=self._rng.choices(severities, weights=weights_list)[0],
            is_visible=False,
        )
        self._next_patient_id += 1
        return patient

    def _spawn_cluster(self, phase: str) -> List[Patient]:
        cluster_size = self._rng.randint(CLUSTER_SIZE_MIN, CLUSTER_SIZE_MAX)
        center_x = self._rng.uniform(10, MAP_SIZE - 10)
        center_y = self._rng.uniform(10, MAP_SIZE - 10)

        patients = []
        for _ in range(cluster_size):
            if len(self.patients) + len(patients) >= self.max_total_patients:
                break

            x = center_x + self._rng.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
            y = center_y + self._rng.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
            x = max(0, min(MAP_SIZE, x))
            y = max(0, min(MAP_SIZE, y))

            weights = SEVERITY_WEIGHTS[phase]
            severities = list(weights.keys())
            weights_list = list(weights.values())

            patient = Patient(
                id=self._next_patient_id,
                x=x,
                y=y,
                severity=self._rng.choices(severities, weights=weights_list)[0],
                is_visible=False,
            )
            self._next_patient_id += 1
            patients.append(patient)

        return patients

    # ── Action application ─────────────────────────────────────────────────────

    def _apply_action(
        self, action: AmbulanceDispatchAction
    ) -> Tuple[float, Dict[str, Any]]:
        reward = 0.0
        info: Dict[str, Any] = {}

        if action.action_type == ActionType.WAIT:
            info["action"] = "wait"
            return reward, info

        if action.ambulance_id is None:
            info["error"] = "ambulance_id required"
            return reward, info

        amb = self._get_ambulance(action.ambulance_id)
        if amb is None:
            info["error"] = f"ambulance {action.ambulance_id} not found"
            return reward, info

        if action.action_type == ActionType.DISPATCH:
            if action.patient_id is None:
                info["error"] = "patient_id required for DISPATCH"
                return reward, info
            patient = self._get_patient(action.patient_id)
            if patient is None or patient.is_picked_up or patient.is_dead or patient.is_delivered:
                info["error"] = "invalid patient for dispatch"
                return reward, info
            if amb.status != AmbulanceStatus.IDLE:
                info["warning"] = f"ambulance {amb.id} already busy"
                return reward, info

            amb.status = AmbulanceStatus.EN_ROUTE_PATIENT
            amb.assigned_patient_id = patient.id
            info[f"dispatched_amb_{amb.id}"] = f"patient {patient.id}"

        elif action.action_type == ActionType.ROUTE_TO_HOSPITAL:
            if action.hospital_id is None:
                info["error"] = "hospital_id required for ROUTE_TO_HOSPITAL"
                return reward, info
            hospital = self._get_hospital(action.hospital_id)
            if hospital is None:
                info["error"] = "invalid hospital"
                return reward, info
            if amb.status != AmbulanceStatus.EN_ROUTE_PATIENT or amb.assigned_patient_id is None:
                info["warning"] = "ambulance not carrying a patient"
                return reward, info
            patient = self._get_patient(amb.assigned_patient_id)
            if patient and patient.severity == Severity.CRITICAL:
                if hospital.available_icu <= 0:
                    reward += PENALTY_NO_BED
                    info["warning"] = f"no ICU beds at hospital {hospital.id}"
                    return reward, info
            elif hospital.available_beds <= 0:
                reward += PENALTY_NO_BED
                info["warning"] = f"no beds at hospital {hospital.id}"
                return reward, info

            amb.status = AmbulanceStatus.EN_ROUTE_HOSPITAL
            amb.assigned_hospital_id = hospital.id
            hospital.incoming_ambulances += 1
            info[f"routing_amb_{amb.id}"] = f"hospital {hospital.id}"

        return reward, info

    # ── Movement ───────────────────────────────────────────────────────────────

    def _move_ambulances(self) -> None:
        for amb in self.ambulances:
            if amb.status == AmbulanceStatus.EN_ROUTE_PATIENT:
                # If primary patient picked up and secondary exists → go to secondary
                primary = self._get_patient(amb.assigned_patient_id)
                if primary and primary.is_picked_up and amb.secondary_patient_id is not None:
                    secondary = self._get_patient(amb.secondary_patient_id)
                    if secondary and not secondary.is_picked_up and not secondary.is_dead:
                        amb.x, amb.y = _move_toward(amb.x, amb.y, secondary.x, secondary.y, AMBULANCE_SPEED)
                        continue
                # Otherwise move toward primary patient
                if primary and not primary.is_picked_up:
                    amb.x, amb.y = _move_toward(amb.x, amb.y, primary.x, primary.y, AMBULANCE_SPEED)
            elif amb.status == AmbulanceStatus.EN_ROUTE_HOSPITAL:
                hospital = self._get_hospital(amb.assigned_hospital_id)
                if hospital:
                    amb.x, amb.y = _move_toward(amb.x, amb.y, hospital.x, hospital.y, AMBULANCE_SPEED)

    # ── Arrival checks ─────────────────────────────────────────────────────────

    def _check_arrivals(self) -> Tuple[float, Dict[str, Any]]:
        reward = 0.0
        info: Dict[str, Any] = {}

        for amb in self.ambulances:
            if amb.status == AmbulanceStatus.EN_ROUTE_PATIENT:
                primary = self._get_patient(amb.assigned_patient_id)

                # Check primary patient pickup
                if primary and not primary.is_picked_up:
                    if _distance(amb.x, amb.y, primary.x, primary.y) < 1.0:
                        primary.is_picked_up = True
                        info[f"amb_{amb.id}_picked_up_patient_{primary.id}"] = True

                # Check secondary patient pickup
                if primary and primary.is_picked_up and amb.secondary_patient_id is not None:
                    secondary = self._get_patient(amb.secondary_patient_id)
                    if secondary and not secondary.is_picked_up and not secondary.is_dead:
                        if _distance(amb.x, amb.y, secondary.x, secondary.y) < 1.0:
                            secondary.is_picked_up = True
                            info[f"amb_{amb.id}_picked_up_secondary_patient_{secondary.id}"] = True

            elif amb.status == AmbulanceStatus.EN_ROUTE_HOSPITAL:
                hospital = self._get_hospital(amb.assigned_hospital_id)
                if hospital and _distance(amb.x, amb.y, hospital.x, hospital.y) < 1.0:
                    # Deliver primary patient
                    primary = self._get_patient(amb.assigned_patient_id)
                    if primary and primary.is_picked_up and not primary.is_delivered:
                        primary.is_delivered = True
                        hospital.incoming_ambulances = max(0, hospital.incoming_ambulances - 1)
                        if primary.severity == Severity.CRITICAL and hospital.available_icu > 0:
                            hospital.available_icu -= 1
                        elif hospital.available_beds > 0:
                            hospital.available_beds -= 1
                        r = {
                            Severity.CRITICAL: REWARD_DELIVERED_CRITICAL,
                            Severity.MEDIUM: REWARD_DELIVERED_MEDIUM,
                            Severity.LOW: REWARD_DELIVERED_LOW,
                        }[primary.severity]
                        reward += r
                        info[f"delivered_patient_{primary.id}"] = r

                    # Deliver secondary patient if picked up
                    if amb.secondary_patient_id is not None:
                        secondary = self._get_patient(amb.secondary_patient_id)
                        if secondary and secondary.is_picked_up and not secondary.is_delivered:
                            secondary.is_delivered = True
                            if secondary.severity == Severity.CRITICAL and hospital.available_icu > 0:
                                hospital.available_icu -= 1
                            elif hospital.available_beds > 0:
                                hospital.available_beds -= 1
                            r = {
                                Severity.CRITICAL: REWARD_DELIVERED_CRITICAL,
                                Severity.MEDIUM: REWARD_DELIVERED_MEDIUM,
                                Severity.LOW: REWARD_DELIVERED_LOW,
                            }[secondary.severity]
                            reward += r
                            info[f"delivered_secondary_patient_{secondary.id}"] = r

                    # Reset ambulance
                    amb.status = AmbulanceStatus.IDLE
                    amb.assigned_patient_id = None
                    amb.secondary_patient_id = None
                    amb.assigned_hospital_id = None

        return reward, info

    # ── Patient ticking ────────────────────────────────────────────────────────

    def _tick_patients(self) -> Tuple[float, Dict[str, Any]]:
        penalty = 0.0
        info: Dict[str, Any] = {}

        for patient in self.patients:
            if patient.is_delivered or patient.is_dead:
                continue
            if not patient.is_picked_up:
                patient.waiting_steps += 1
                threshold = DEATH_THRESHOLD[patient.severity]
                if patient.waiting_steps >= threshold:
                    patient.is_dead = True
                    penalty += PENALTY_DEATH
                    info[f"patient_{patient.id}_died"] = True

        return penalty, info

    # ── Visibility ─────────────────────────────────────────────────────────────

    def _update_visibility(self) -> None:
        for patient in self.patients:
            if patient.is_visible:
                continue
            for amb in self.ambulances:
                if _distance(amb.x, amb.y, patient.x, patient.y) <= VISIBILITY_RADIUS:
                    patient.is_visible = True
                    break

    # ── Observation builder ────────────────────────────────────────────────────

    def _build_observation(self) -> AmbulanceDispatchObservation:
        visible_patients = [p for p in self.patients if p.is_visible]
        phase = _get_phase(self.time_step, self.max_steps)
        return AmbulanceDispatchObservation(
            ambulances=list(self.ambulances),
            patients=visible_patients,
            hospitals=list(self.hospitals),
            time_step=self.time_step,
            max_steps=self.max_steps,
            total_patients_spawned=self._next_patient_id,
            current_phase=phase,
        )

    # ── Done condition ─────────────────────────────────────────────────────────

    def _is_done(self) -> bool:
        if self.time_step >= self.max_steps:
            return True
        if not self.enable_dynamic_spawning:
            return all(p.is_delivered or p.is_dead for p in self.patients)
        return False

    # ── Lookup helpers ─────────────────────────────────────────────────────────

    def _get_ambulance(self, aid: Optional[int]) -> Optional[Ambulance]:
        if aid is None:
            return None
        return next((a for a in self.ambulances if a.id == aid), None)

    def _get_patient(self, pid: Optional[int]) -> Optional[Patient]:
        if pid is None:
            return None
        return next((p for p in self.patients if p.id == pid), None)

    def _get_hospital(self, hid: Optional[int]) -> Optional[Hospital]:
        if hid is None:
            return None
        return next((h for h in self.hospitals if h.id == hid), None)