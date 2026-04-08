---
title: Ambulance Dispatch Environment
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - reinforcement-learning
  - openenv
  - simulation
  - emergency-dispatch
---

# Ambulance Dispatch Environment

A reinforcement learning environment simulating the real-world 108 emergency dispatch problem. Ambulances must be dispatched to patients and routed to hospitals efficiently under partial observability.

## Quick Start
```python
from ambulance_dispatch import AmbulanceDispatchAction, AmbulanceDispatchEnv
from models import ActionType

try:
    # Create environment from Docker image
    env = AmbulanceDispatchEnv.from_docker_image("ambulance_dispatch-env:latest")

    # Reset the environment
    obs = env.reset()
    print(f"Ambulances: {len(obs.observation.ambulances)}")
    print(f"Visible patients: {len(obs.observation.patients)}")
    print(f"Hospitals: {len(obs.observation.hospitals)}")

    # Dispatch ambulance 0 to patient 0
    result = env.step(AmbulanceDispatchAction(
        action_type=ActionType.DISPATCH,
        ambulance_id=0,
        patient_id=0
    ))
    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")

finally:
    env.close()
```

## Building the Docker Image
```bash
docker build -t ambulance_dispatch-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces
```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment
- `--repo-id`, `-r`: Repository ID in format `username/repo-name`
- `--base-image`, `-b`: Base Docker image to use
- `--private`: Deploy the space as private (default: public)

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

## Environment Details

### Action — `AmbulanceDispatchAction`
| Field | Type | Description |
|---|---|---|
| `action_type` | `ActionType` | `DISPATCH`, `ROUTE_TO_HOSPITAL`, or `WAIT` |
| `ambulance_id` | `int` | Which ambulance to act on |
| `patient_id` | `int` | Target patient (DISPATCH only) |
| `hospital_id` | `int` | Target hospital (ROUTE_TO_HOSPITAL only) |

### Observation — `AmbulanceDispatchObservation`
| Field | Type | Description |
|---|---|---|
| `ambulances` | `List[Ambulance]` | All ambulances with current status |
| `patients` | `List[Patient]` | Only visible patients (partial observability) |
| `hospitals` | `List[Hospital]` | All hospitals with bed availability |
| `time_step` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps in this episode |

### Reward
| Event | Reward |
|---|---|
| Critical patient delivered | +30 |
| Medium patient delivered | +20 |
| Low patient delivered | +10 |
| Patient death | -50 |
| No bed available | -10 |
| Per step | -1 |

### Patient Death Thresholds
| Severity | Steps before death |
|---|---|
| CRITICAL | 5 |
| MEDIUM | 10 |
| LOW | 20 |

## Tasks

Three preset difficulty levels are available in `tasks.py`:

| Task ID | Ambulances | Hospitals | Patients | Max Steps |
|---|---|---|---|---|
| `ambulance_dispatch_easy` | 3 | 2 | 4 | 60 |
| `ambulance_dispatch_medium` | 3 | 3 | 6 | 60 |
| `ambulance_dispatch_hard` | 2 | 2 | 8 | 50 |

## Running Locally
```bash
uvicorn server.app:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| POST | `/reset` | Reset environment |
| POST | `/step` | Send action, get observation |
| GET | `/observation` | Get current observation |
| GET | `/action_space` | Get valid action space |
| GET | `/observation_space` | Get observation space metadata |

## Project Structure
```
ambulance-openenv/
└── ambulance_dispatch/
    ├── __init__.py
    ├── client.py
    ├── models.py
    ├── tasks.py
    ├── openenv.yaml
    ├── pyproject.toml
    ├── README.md
    └── server/
        ├── __init__.py
        ├── ambulance_dispatch_environment.py
        ├── app.py
        ├── requirements.txt
        └── Dockerfile



        