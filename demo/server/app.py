# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI server for the Ambulance Dispatch Environment.

Endpoints
---------
POST /reset              → reset the environment, get first observation
POST /step               → send an action, get back obs + reward + done + info
POST /auto_dispatch      → run full episode with smart automatic dispatching
GET  /observation        → get the current observation without stepping
GET  /state              → get current episode_id and step_count
GET  /action_space       → get the valid action space
GET  /observation_space  → get the observation space metadata
GET  /health             → liveness probe
GET  /                   → landing page (falls back to /docs if file missing)
GET  /docs               → custom dark themed swagger UI
"""

from typing import Any, Dict, List
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from server.ambulance_dispatch_environment import AmbulanceDispatchEnvironment, _get_phase
from server.smart_dispatcher import compute_dispatch_actions, compute_hospital_routing
from models import (
    AmbulanceDispatchAction,
    AmbulanceDispatchObservation,
)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🚑 Ambulance Dispatch Environment",
    description="""
## 108 Emergency Dispatch Simulation

A production-grade **Reinforcement Learning environment** simulating real-world emergency medical dispatch.

---

### 🧠 What is this?
An AI agent learns to optimally assign ambulances to patients, prioritize critical cases, and maximize patient survival rates across a dynamic 100×100 map.

---

### ⚡ Key Features
| Feature | Description |
|---|---|
| **Smart Auto Dispatch** | Priority-based assignment (CRITICAL → MEDIUM → LOW) |
| **Multi-Pickup** | Single ambulance picks up 2 nearby patients in one trip |
| **Dynamic Spawning** | New patients appear during episode like real 108 calls |
| **Cluster Spawning** | Accident scenes with 2-3 patients at same location |
| **3 Episode Phases** | Early (calm) → Middle (busy) → Late (critical surge) |
| **Simultaneous Actions** | All ambulances dispatched in one step |

---

### 🚦 Quick Start
1. Call **POST /reset** to start a new episode
2. Call **POST /auto_dispatch** to run a full automatic episode
3. Or use **POST /step** manually with your own actions

---

### 🏆 Reward System
- ✅ **+30** CRITICAL patient delivered
- ✅ **+20** MEDIUM patient delivered
- ✅ **+10** LOW patient delivered
- ❌ **-50** Patient death
- ❌ **-10** No hospital bed
- ⏱️ **-1** Per step (urgency pressure)

---

### 📦 Built With
`FastAPI` · `Pydantic` · `OpenEnv` · `Docker` · `Python 3.11`
    """,
    version="1.0.0",
    contact={
        "name": "Ambulance Dispatch Team",
    },
    license_info={
        "name": "BSD-3-Clause",
    },
    docs_url=None,
    redoc_url=None,
    openapi_tags=[
        {
            "name": "meta",
            "description": "Server health and navigation endpoints",
        },
        {
            "name": "env",
            "description": "Core environment control — reset, step, observe",
        },
        {
            "name": "smart",
            "description": "🧠 Smart auto dispatch — full episode with AI-powered routing",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton environment ──────────────────────────────────────────────────────
_env: AmbulanceDispatchEnvironment = AmbulanceDispatchEnvironment()
_last_obs: AmbulanceDispatchObservation | None = None

# ── Helper to resolve file paths ───────────────────────────────────────────────
def _root_file(filename: str) -> str:
    """Resolve path to a file in the ambulance_dispatch root folder."""
    # Get the directory where this app.py file is located (server/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the root (ambulance_dispatch/)
    root_dir = os.path.dirname(current_dir)
    return os.path.join(root_dir, filename)


# ── Request / Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    num_ambulances: int = 3
    num_hospitals: int = 2
    num_patients: int = 6
    max_steps: int = 60
    seed: int | None = None
    enable_dynamic_spawning: bool = True
    max_total_patients: int = 20


class StepResponse(BaseModel):
    observation: AmbulanceDispatchObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class AutoDispatchRequest(BaseModel):
    num_ambulances: int = 3
    num_hospitals: int = 2
    num_patients: int = 6
    max_steps: int = 60
    seed: int | None = None
    enable_dynamic_spawning: bool = True
    max_total_patients: int = 20


class AutoDispatchResponse(BaseModel):
    total_reward: float
    steps_taken: int
    patients_saved: int
    patients_dead: int
    patients_unresolved: int
    patients_total: int
    delivery_rate: float
    episode_log: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"], include_in_schema=False)
def landing_page():
    """Serves the landing page. Falls back to /docs redirect if file is missing."""
    path = _root_file("landing_page.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Custom dark themed Swagger UI."""
    return HTMLResponse("""<!DOCTYPE html>
<html>
<head>
  <title>🚑 Ambulance Dispatch API</title>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #0d0d0d !important; font-family: 'Segoe UI', sans-serif; }
    .swagger-ui { background: #0d0d0d !important; }
    .swagger-ui .topbar { background: #0d0d0d !important; border-bottom: 1px solid rgba(230,57,70,0.3) !important; padding: 12px 24px !important; }
    .swagger-ui .topbar-wrapper img { display: none; }
    .swagger-ui .topbar-wrapper::before {
      content: '🚑 Ambulance Dispatch Environment — API Reference';
      color: #f0f0f0; font-size: 15px; font-weight: 600; letter-spacing: 0.03em;
    }
    .swagger-ui .topbar a { display: none; }
    .swagger-ui .info { background: #161616 !important; padding: 24px !important; border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.06) !important; margin: 20px 0 !important; }
    .swagger-ui .info .title { color: #f0f0f0 !important; font-size: 26px !important; }
    .swagger-ui .info .description p { color: #ccc !important; }
    .swagger-ui .info .description table { color: #ccc !important; width: 100%; }
    .swagger-ui .info .description th { color: #888 !important; background: #1f1f1f !important; padding: 6px 12px !important; }
    .swagger-ui .info .description td { padding: 6px 12px !important; border-bottom: 1px solid rgba(255,255,255,0.04) !important; }
    .swagger-ui .info .description a { color: #E63946 !important; }
    .swagger-ui .info .description strong { color: #f0f0f0 !important; }
    .swagger-ui .info .description h3 { color: #f0f0f0 !important; margin: 16px 0 8px !important; }
    .swagger-ui .info .description code { background: #252525 !important; padding: 2px 6px !important; border-radius: 4px !important; color: #4ecdc4 !important; }
    .swagger-ui .info .description hr { border-color: rgba(255,255,255,0.06) !important; }
    .swagger-ui .scheme-container { background: #161616 !important; border-bottom: 1px solid rgba(255,255,255,0.06) !important; padding: 12px 24px !important; }
    .swagger-ui .opblock-tag { color: #f0f0f0 !important; border-bottom: 1px solid rgba(255,255,255,0.08) !important; font-size: 15px !important; padding: 10px 0 !important; }
    .swagger-ui .opblock-tag:hover { background: rgba(255,255,255,0.02) !important; }
    .swagger-ui .opblock-tag-section { background: #0d0d0d !important; }
    .swagger-ui .opblock-tag small { color: #888 !important; }
    .swagger-ui .opblock { background: #161616 !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; margin-bottom: 8px !important; box-shadow: none !important; }
    .swagger-ui .opblock:hover { border-color: rgba(230,57,70,0.3) !important; }
    .swagger-ui .opblock .opblock-summary { border-radius: 8px !important; padding: 10px 16px !important; }
    .swagger-ui .opblock .opblock-summary-method { border-radius: 4px !important; font-weight: 600 !important; min-width: 60px !important; text-align: center !important; }
    .swagger-ui .opblock .opblock-summary-path { color: #f0f0f0 !important; font-weight: 500 !important; }
    .swagger-ui .opblock .opblock-summary-description { color: #888 !important; }
    .swagger-ui .opblock.opblock-post { border-color: rgba(230,57,70,0.25) !important; }
    .swagger-ui .opblock.opblock-post .opblock-summary { background: rgba(230,57,70,0.06) !important; }
    .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #E63946 !important; }
    .swagger-ui .opblock.opblock-get { border-color: rgba(29,158,117,0.25) !important; }
    .swagger-ui .opblock.opblock-get .opblock-summary { background: rgba(29,158,117,0.06) !important; }
    .swagger-ui .opblock.opblock-get .opblock-summary-method { background: #1D9E75 !important; }
    .swagger-ui .opblock-body { background: #1a1a1a !important; border-top: 1px solid rgba(255,255,255,0.06) !important; border-radius: 0 0 8px 8px !important; }
    .swagger-ui .opblock-description-wrapper p { color: #ccc !important; }
    .swagger-ui .opblock-section-header { background: #1f1f1f !important; border-bottom: 1px solid rgba(255,255,255,0.06) !important; }
    .swagger-ui .opblock-section-header label { color: #ccc !important; font-weight: 600 !important; }
    .swagger-ui .tab li { color: #888 !important; }
    .swagger-ui .tab li.active { color: #f0f0f0 !important; border-bottom: 2px solid #E63946 !important; }
    .swagger-ui textarea, .swagger-ui input[type=text], .swagger-ui input[type=password], .swagger-ui input[type=search], .swagger-ui input[type=email] {
      background: #252525 !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #f0f0f0 !important; border-radius: 6px !important;
    }
    .swagger-ui .btn { border-radius: 6px !important; font-weight: 600 !important; transition: all 0.2s !important; }
    .swagger-ui .btn.execute { background: #E63946 !important; border-color: #E63946 !important; color: white !important; }
    .swagger-ui .btn.execute:hover { background: #c1121f !important; }
    .swagger-ui .btn.cancel { background: transparent !important; border-color: rgba(255,255,255,0.2) !important; color: #ccc !important; }
    .swagger-ui .btn.try-out__btn { background: transparent !important; border-color: rgba(230,57,70,0.5) !important; color: #E63946 !important; }
    .swagger-ui .btn.try-out__btn:hover { background: rgba(230,57,70,0.1) !important; }
    .swagger-ui .btn.try-out__btn.cancel { border-color: rgba(255,255,255,0.2) !important; color: #ccc !important; }
    .swagger-ui .responses-inner { background: #1a1a1a !important; padding: 16px !important; }
    .swagger-ui .response-col_status { color: #4ecdc4 !important; font-weight: 600 !important; }
    .swagger-ui .response-col_description { color: #ccc !important; }
    .swagger-ui table thead tr td, .swagger-ui table thead tr th { color: #888 !important; border-bottom: 1px solid rgba(255,255,255,0.08) !important; background: #1f1f1f !important; padding: 8px 12px !important; }
    .swagger-ui table tbody tr td { color: #ccc !important; border-bottom: 1px solid rgba(255,255,255,0.04) !important; padding: 8px 12px !important; }
    .swagger-ui .model-box { background: #1f1f1f !important; border-radius: 6px !important; border: 1px solid rgba(255,255,255,0.06) !important; }
    .swagger-ui .model { color: #ccc !important; }
    .swagger-ui .model-title { color: #f0f0f0 !important; font-weight: 600 !important; }
    .swagger-ui .prop-type { color: #7eb3f7 !important; }
    .swagger-ui .prop-format { color: #888 !important; }
    .swagger-ui section.models { background: #161616 !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 8px !important; margin-top: 20px !important; }
    .swagger-ui section.models h4 { color: #f0f0f0 !important; }
    .swagger-ui section.models .model-container { background: #1a1a1a !important; border-color: rgba(255,255,255,0.06) !important; border-radius: 6px !important; }
    .swagger-ui .highlight-code { background: #252525 !important; border-radius: 6px !important; }
    .swagger-ui .microlight { color: #4ecdc4 !important; }
    .swagger-ui .servers > label select { background: #252525 !important; color: #f0f0f0 !important; border-color: rgba(255,255,255,0.1) !important; border-radius: 6px !important; }
    .swagger-ui .arrow { fill: #888 !important; }
    .swagger-ui .expand-methods svg, .swagger-ui .expand-operation svg { fill: #888 !important; }
    .swagger-ui .parameter__name { color: #f0f0f0 !important; }
    .swagger-ui .parameter__type { color: #7eb3f7 !important; }
    .swagger-ui .parameter__deprecated { color: #888 !important; }
    .swagger-ui .parameter__in { color: #888 !important; font-style: italic !important; }
    .swagger-ui .required > .parameter__name::after { color: #E63946 !important; }
    .swagger-ui .renderedMarkdown p { color: #ccc !important; }
    .swagger-ui .renderedMarkdown code { background: #252525 !important; color: #4ecdc4 !important; padding: 2px 6px !important; border-radius: 4px !important; }
    .swagger-ui .copy-to-clipboard { background: #252525 !important; border-color: rgba(255,255,255,0.1) !important; }
    .swagger-ui .copy-to-clipboard button { background: transparent !important; }
    #swagger-ui { max-width: 1200px; margin: 0 auto; padding: 0 24px 40px; }
  </style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  window.onload = () => {
    SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: '#swagger-ui',
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
      layout: "BaseLayout",
      deepLinking: true,
      defaultModelsExpandDepth: 1,
      defaultModelExpandDepth: 1,
    })
  }
</script>
</body>
</html>""")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe — always returns 200 OK."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=AmbulanceDispatchObservation, tags=["env"])
def reset(request: ResetRequest = ResetRequest()) -> AmbulanceDispatchObservation:
    """Reset the environment and return the first observation."""
    global _env, _last_obs
    _env = AmbulanceDispatchEnvironment(
        num_ambulances=request.num_ambulances,
        num_hospitals=request.num_hospitals,
        num_patients=request.num_patients,
        max_steps=request.max_steps,
        seed=request.seed,
        enable_dynamic_spawning=request.enable_dynamic_spawning,
        max_total_patients=request.max_total_patients,
    )
    _last_obs = _env.reset()
    return _last_obs


@app.post("/step", response_model=StepResponse, tags=["env"])
def step(action: AmbulanceDispatchAction) -> StepResponse:
    """Apply an action and advance the environment by one step."""
    global _last_obs
    if _last_obs is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first.",
        )
    obs, reward, done, info = _env.step(action)
    _last_obs = obs
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.post("/auto_dispatch", response_model=AutoDispatchResponse, tags=["smart"])
def auto_dispatch(request: AutoDispatchRequest = AutoDispatchRequest()) -> AutoDispatchResponse:
    """
    Run a full episode with smart automatic dispatching.

    Automatically:
    - Assigns nearest ambulance to highest priority patient
    - All ambulances dispatched simultaneously in one step
    - Routes to nearest hospital with available beds
    - Reassigns ambulance after delivery to next waiting patient
    - Multi-pickup when 2 patients are close enough
    - Dynamic patient spawning with early/middle/late phases
    - Cluster spawning for accident scenes
    """
    env = AmbulanceDispatchEnvironment(
        num_ambulances=request.num_ambulances,
        num_hospitals=request.num_hospitals,
        num_patients=request.num_patients,
        max_steps=request.max_steps,
        seed=request.seed,
        enable_dynamic_spawning=request.enable_dynamic_spawning,
        max_total_patients=request.max_total_patients,
    )
    obs = env.reset()

    total_reward = 0.0
    episode_log = []
    done = False

    while not done:
        step_actions = []

        hospital_actions = compute_hospital_routing(
            obs.ambulances,
            obs.patients,
            obs.hospitals,
        )
        step_actions.extend(hospital_actions)

        dispatch_actions = compute_dispatch_actions(
            obs.ambulances,
            obs.patients,
            obs.hospitals,
        )
        step_actions.extend(dispatch_actions)

        if not step_actions:
            action = AmbulanceDispatchAction(
                action_type="wait",
                ambulance_id=None,
                patient_id=None,
                hospital_id=None,
            )
            obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_log.append({
                "step": obs.time_step,
                "phase": obs.current_phase,
                "total_patients_spawned": obs.total_patients_spawned,
                "action": "wait",
                "reward": reward,
                "info": info,
            })
        else:
            action_objects = [
                AmbulanceDispatchAction(
                    action_type=a["action_type"],
                    ambulance_id=a["ambulance_id"],
                    patient_id=a["patient_id"],
                    hospital_id=a["hospital_id"],
                )
                for a in step_actions
            ]

            obs, reward, done, info = env.step_multiple(action_objects)
            total_reward += reward

            episode_log.append({
                "step": obs.time_step,
                "phase": obs.current_phase,
                "total_patients_spawned": obs.total_patients_spawned,
                "actions": [
                    {
                        "action": str(a["action_type"]),
                        "ambulance_id": a["ambulance_id"],
                        "patient_id": a.get("patient_id"),
                        "hospital_id": a.get("hospital_id"),
                        "multi_pickup": a.get("multi_pickup_patient_id"),
                    }
                    for a in step_actions
                ],
                "reward": reward,
                "info": info,
            })

    all_patients = env.patients
    patients_saved = sum(1 for p in all_patients if p.is_delivered)
    patients_dead = sum(1 for p in all_patients if p.is_dead and not p.is_delivered)
    patients_unresolved = sum(1 for p in all_patients if not p.is_delivered and not p.is_dead)
    patients_total = len(all_patients)
    delivery_rate = patients_saved / patients_total if patients_total > 0 else 0.0

    return AutoDispatchResponse(
        total_reward=total_reward,
        steps_taken=obs.time_step,
        patients_saved=patients_saved,
        patients_dead=patients_dead,
        patients_unresolved=patients_unresolved,
        patients_total=patients_total,
        delivery_rate=delivery_rate,
        episode_log=episode_log,
    )


@app.get("/observation", response_model=AmbulanceDispatchObservation, tags=["env"])
def get_observation() -> AmbulanceDispatchObservation:
    """Return the most recent observation without stepping."""
    if _last_obs is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first.",
        )
    return _last_obs


@app.get("/state", tags=["env"])
def get_state() -> Dict[str, Any]:
    """Return current episode state - episode_id and step_count."""
    return {
        "episode_id": id(_env),
        "step_count": _env.time_step,
        "current_phase": _get_phase(_env.time_step, _env.max_steps),
        "total_patients_spawned": _env._next_patient_id,
    }


@app.get("/action_space", tags=["env"])
def get_action_space() -> Dict[str, Any]:
    """Return the valid action space."""
    return _env.get_action_space()


@app.get("/observation_space", tags=["env"])
def get_observation_space() -> Dict[str, Any]:
    """Return the observation space metadata."""
    return _env.get_observation_space()


def main():
    """Main entry point for running the server."""
    import uvicorn
    import os
    
    # Get port from environment (Hugging Face uses PORT env var) or default to 7860
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
