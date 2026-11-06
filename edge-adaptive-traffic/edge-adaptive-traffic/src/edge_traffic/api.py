"""FastAPI service for storing, querying traffic telemetry, and real-time Digital Twin WebSocket hub."""
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from .storage import get_telemetry_collection, get_storage_mode
from .micro_sim import MultiJunctionCorridor, MicroIntersection
from .deep_rl import DeepQLearningController
from .controller import create_controller
from .analytics import TrafficAnalytics
from .vision_engine import EdgeVisionEngine
from .preemption import EmergencyVehicleManager

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sim_task
    if _sim_task is None or _sim_task.done():
        _sim_task = asyncio.create_task(_simulation_loop())
    yield
    if _sim_task and not _sim_task.done():
        _sim_task.cancel()

app = FastAPI(
    title="NEXUS-ITS Autonomous Edge Traffic Management API",
    description="Intelligent Transportation System with Real-Time Digital Twin, MARL & Emergency Preemption.",
    version="2.5.0",
    lifespan=lifespan,
)

# Shared Corridor Simulation & Controllers
corridor = MultiJunctionCorridor()
analytics = TrafficAnalytics()
vision_engine = EdgeVisionEngine()
emergency_manager = EmergencyVehicleManager()

active_policy_name = "dqn"
active_controllers = {
    jid: create_controller("dqn", {}) for jid in corridor.junctions
}
active_websockets: List[WebSocket] = []

# Background simulation runner
_sim_task: Optional[asyncio.Task] = None


class QueueCounts(BaseModel):
    N: int = Field(ge=0)
    S: int = Field(ge=0)
    E: int = Field(ge=0)
    W: int = Field(ge=0)


class TelemetryReading(BaseModel):
    intersection_id: str = Field(default="junction01", min_length=1, max_length=100)
    t: int = Field(ge=0)
    phase: Literal["NS_GREEN", "EW_GREEN"]
    queues: QueueCounts
    action: Literal[0, 1]
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmergencyDispatchRequest(BaseModel):
    vehicle_type: str = "ambulance"
    origin_junction: str = "junction01"
    approach: str = "W"
    direction: str = "eastbound"


class IncidentRequest(BaseModel):
    intersection_id: str = "junction01"
    approach: str = "E"
    duration_seconds: float = 30.0


class PolicySwitchRequest(BaseModel):
    policy: str  # "dqn", "green_wave", "actuated", "fixed"


class WeatherRequest(BaseModel):
    weather: str  # "clear", "rain", "fog", "night"


def _database_error(exc):
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=503, detail="Database is unavailable.") from exc


def _serialize(document):
    document["_id"] = str(document["_id"])
    return document





async def _simulation_loop():
    """High-frequency 20 Hz simulation & controller loop for real-time WebSocket streaming."""
    dt = 0.05
    step_accum = 0.0
    while True:
        try:
            # Advance microscopic physics
            corridor.step(dt)
            step_accum += dt

            # Run AI controllers periodically every 2.0 simulation seconds
            if step_accum >= 1.0:
                step_accum = 0.0
                for jid, j in corridor.junctions.items():
                    ctrl = active_controllers.get(jid)
                    if ctrl is not None and not j.preemption_active:
                        q_arr = j.get_queue_array()
                        action = ctrl.act(q_arr, j.phase, int(j.phase_timer / 5.0))
                        if action == 1:
                            j.request_phase_switch()

            # Broadcast to active WebSockets
            if active_websockets:
                snap = corridor.junctions["junction01"].get_state_snapshot()
                eco = analytics.compute_eco_metrics(snap["queues"], dt)
                dci = analytics.compute_congestion_index(snap["queues"])
                preds = analytics.predict_multi_horizon(snap["queues"], 8.5)

                payload = {
                    "type": "state_update",
                    "junction": snap,
                    "eco": eco,
                    "dci": dci,
                    "predictions": preds,
                    "policy": active_policy_name
                }
                msg = json.dumps(payload)
                dead_sockets = []
                for ws in active_websockets:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead_sockets.append(ws)
                for ws in dead_sockets:
                    if ws in active_websockets:
                        active_websockets.remove(ws)

        except Exception as e:
            pass
        await asyncio.sleep(dt)


# --- FRONTEND MISSION CONTROL DASHBOARD ---
@app.get("/", response_class=HTMLResponse)
def get_mission_control_dashboard():
    web_file = Path(__file__).parent / "web" / "index.html"
    if web_file.exists():
        return HTMLResponse(content=web_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>NEXUS-ITS Smart Traffic Digital Twin</h1><p>Web template loading...</p>")


# --- WEBSOCKET REAL-TIME ENDPOINT ---
@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket, junction: str = "junction01"):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                action = cmd.get("action")
                if action == "dispatch_ambulance":
                    j = corridor.junctions.get(cmd.get("junction", "junction01"))
                    if j:
                        j.spawn_vehicle(cmd.get("approach", "W"), is_emergency=True)
                elif action == "set_weather":
                    for j in corridor.junctions.values():
                        j.set_weather(cmd.get("weather", "clear"))
                elif action == "set_policy":
                    global active_policy_name, active_controllers
                    active_policy_name = cmd.get("policy", "dqn")
                    active_controllers = {
                        jid: create_controller(active_policy_name, {}) for jid in corridor.junctions
                    }
                elif action == "switch_phase":
                    j = corridor.junctions.get(cmd.get("junction", "junction01"))
                    if j:
                        j.request_phase_switch()
            except Exception:
                pass
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# --- REST API ENDPOINTS ---
@app.get("/health")
def health():
    try:
        get_telemetry_collection().database.client.admin.command("ping")
    except (RuntimeError, PyMongoError) as exc:
        _database_error(exc)
    return {
        "status": "ok",
        "database": "connected",
        "storage_mode": get_storage_mode(),
        "active_policy": active_policy_name,
        "intersections_online": len(corridor.junctions)
    }


@app.post("/api/telemetry", status_code=status.HTTP_201_CREATED)
def create_telemetry(reading: TelemetryReading):
    try:
        collection = get_telemetry_collection()
        document = reading.model_dump()
        result = collection.insert_one(document)
    except (RuntimeError, PyMongoError) as exc:
        _database_error(exc)
    document["_id"] = str(result.inserted_id)
    return document


@app.get("/api/telemetry")
def list_telemetry(
    intersection_id: str = Query(default="junction01", min_length=1, max_length=100),
    limit: int = Query(default=50, ge=1, le=500),
):
    try:
        documents = (
            get_telemetry_collection()
            .find({"intersection_id": intersection_id})
            .sort("recorded_at", -1)
            .limit(limit)
        )
        return [_serialize(document) for document in documents]
    except (RuntimeError, PyMongoError) as exc:
        _database_error(exc)


@app.get("/api/intersections/{intersection_id}/latest")
def latest_telemetry(intersection_id: str):
    try:
        document = get_telemetry_collection().find_one(
            {"intersection_id": intersection_id},
            sort=[("recorded_at", -1)],
        )
    except (RuntimeError, PyMongoError) as exc:
        _database_error(exc)
    if document is None:
        raise HTTPException(status_code=404, detail="No telemetry found for this intersection.")
    return _serialize(document)


# --- MISSION CONTROL & ITS ACTION APIS ---
@app.post("/api/emergency/dispatch")
def dispatch_emergency(req: EmergencyDispatchRequest):
    j = corridor.junctions.get(req.origin_junction)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found.")
    veh = j.spawn_vehicle(req.approach, is_emergency=True)
    ev = emergency_manager.dispatch_emergency(
        vehicle_type=req.vehicle_type,
        origin_junction=req.origin_junction,
        approach=req.approach,
        direction=req.direction
    )
    return {"status": "dispatched", "vehicle_id": veh.id, "event": ev}


@app.post("/api/incident/trigger")
def trigger_incident(req: IncidentRequest):
    j = corridor.junctions.get(req.intersection_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found.")
    j.trigger_incident(req.approach, req.duration_seconds)
    return {"status": "incident_active", "intersection_id": req.intersection_id, "approach": req.approach}


@app.post("/api/controller/policy")
def set_policy(req: PolicySwitchRequest):
    global active_policy_name, active_controllers
    active_policy_name = req.policy.lower()
    active_controllers = {
        jid: create_controller(active_policy_name, {}) for jid in corridor.junctions
    }
    return {"status": "policy_updated", "policy": active_policy_name}


@app.post("/api/controller/switch_phase")
def switch_phase(intersection_id: str = "junction01"):
    j = corridor.junctions.get(intersection_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found.")
    switched = j.request_phase_switch()
    return {"status": "switch_requested", "success": switched, "phase_state": j.phase_state}


@app.post("/api/environment/weather")
def set_weather(req: WeatherRequest):
    for j in corridor.junctions.values():
        j.set_weather(req.weather)
    return {"status": "weather_updated", "weather": req.weather}


@app.get("/api/analytics/corridor")
def get_corridor_status():
    return corridor.get_corridor_snapshot()
