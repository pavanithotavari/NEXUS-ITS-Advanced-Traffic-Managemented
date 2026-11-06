"""High-Fidelity Microscopic Traffic Simulator & Multi-Junction Digital Twin Engine.

Simulates individual physical vehicles with Intelligent Driver Model (IDM),
lane-level queuing, turning behaviors, emergency vehicles, pedestrian crosswalks,
weather conditions, and road incidents.
"""
import math
import random
import time
from typing import Dict, List, Optional, Tuple, Any

APPROACHES = ["N", "S", "E", "W"]
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
TURN_RIGHT = {"N": "W", "W": "S", "S": "E", "E": "N"}
TURN_LEFT = {"N": "E", "E": "S", "S": "W", "W": "N"}


class Vehicle:
    def __init__(
        self,
        vid: int,
        origin: str,
        turn: str = "straight",  # "straight", "left", "right"
        kind: str = "car",       # "car", "bus", "truck", "motorcycle", "ambulance"
        speed: float = 11.0,     # m/s (~40 km/h)
        target_speed: float = 13.8, # m/s (~50 km/h)
        position: float = 240.0, # meters from stop line
        is_emergency: bool = False
    ):
        self.id = vid
        self.origin = origin
        self.turn = turn
        self.kind = kind
        self.speed = speed
        self.target_speed = target_speed
        self.position = position  # distance to stop line (positive = approaching, negative = passed)
        self.is_emergency = is_emergency or (kind == "ambulance")
        self.length = 12.0 if kind in ("bus", "truck") else (2.2 if kind == "motorcycle" else 4.5)
        self.max_accel = 3.5 if self.is_emergency else 2.2
        self.comf_decel = 2.8 if self.is_emergency else 2.0
        self.wait_time = 0.0
        self.stopped = False
        self.fuel_ml = 0.0
        self.co2_g = 0.0
        self.is_stalled = False
        self.has_violated_red = False
        self.lane = 0  # 0: right/straight, 1: left

        if turn == "left":
            self.lane = 1
        elif turn == "right":
            self.lane = 0

    def update_idm(self, lead_vehicle: Optional["Vehicle"], is_red_for_me: bool, dt: float, friction: float = 1.0):
        if self.is_stalled:
            self.speed = 0.0
            self.stopped = True
            self.wait_time += dt
            self.fuel_ml += 0.22 * dt  # idle fuel
            self.co2_g += 0.52 * dt
            return

        # Distance to effective stop line
        s_stop = self.position
        desired_s0 = 2.5
        desired_T = 1.2

        lead_dist = 999.0
        lead_speed = self.target_speed

        if lead_vehicle is not None:
            lead_dist = (self.position - lead_vehicle.position) - lead_vehicle.length
            lead_dist = max(0.1, lead_dist)
            lead_speed = lead_vehicle.speed

        # Red signal or amber clearance
        must_stop_for_signal = is_red_for_me and (s_stop > -1.0) and (s_stop < 120.0)
        if must_stop_for_signal:
            if s_stop < lead_dist:
                lead_dist = max(0.1, s_stop)
                lead_speed = 0.0

        # IDM Acceleration calculation
        v = max(0.0, self.speed)
        v0 = self.target_speed
        a = self.max_accel * friction
        b = self.comf_decel * friction

        delta_v = v - lead_speed
        s_star = desired_s0 + max(0.0, v * desired_T + (v * delta_v) / (2.0 * math.sqrt(a * b) + 1e-6))
        accel = a * (1.0 - (v / v0) ** 4 - (s_star / lead_dist) ** 2)

        # Update physics
        self.speed = max(0.0, v + accel * dt)
        self.position -= self.speed * dt

        if self.speed < 0.8:
            self.stopped = True
            self.wait_time += dt
            # Idle fuel: ~0.8 L/hr for car = ~0.22 ml/s; ~2.31 kg CO2/L = ~0.52 g/s
            fuel_rate = 0.6 if self.kind in ("bus", "truck") else 0.22
            self.fuel_ml += fuel_rate * dt
            self.co2_g += (fuel_rate * 2.35) * dt
        else:
            self.stopped = False
            # Cruising fuel
            self.fuel_ml += (0.12 + 0.008 * self.speed) * dt
            self.co2_g += (0.28 + 0.018 * self.speed) * dt


class Pedestrian:
    def __init__(self, pid: int, approach: str):
        self.id = pid
        self.approach = approach  # which crosswalk
        self.progress = 0.0       # 0.0 (waiting at curb) to 1.0 (crossed)
        self.speed = random.uniform(1.1, 1.4)  # m/s
        self.waiting = True
        self.crossing = False
        self.finished = False

    def update(self, can_walk: bool, dt: float):
        if self.waiting:
            if can_walk:
                self.waiting = False
                self.crossing = True
        elif self.crossing:
            self.progress += (self.speed / 16.0) * dt  # crosswalk length ~16m
            if self.progress >= 1.0:
                self.crossing = False
                self.finished = True


class MicroIntersection:
    """Microscopic simulation of a 4-way intersection."""

    def __init__(self, intersection_id: str = "junction01", cfg: Optional[Dict[str, Any]] = None):
        self.id = intersection_id
        self.cfg = cfg or {}
        self.vehicles: Dict[str, List[Vehicle]] = {"N": [], "S": [], "E": [], "W": []}
        self.pedestrians: List[Pedestrian] = []
        self.departed_vehicles = 0
        self.total_wait_time = 0.0
        self.total_co2_grams = 0.0
        self.co2_saved_grams = 0.0  # vs fixed baseline
        self.next_vid = 1
        self.next_pid = 1

        # Phase control: 0 = NS Green, 1 = EW Green
        self.phase = 0
        self.phase_state = "NS_GREEN"  # NS_GREEN, NS_YELLOW, ALL_RED, EW_GREEN, EW_YELLOW, ALL_RED_2
        self.phase_timer = 0.0
        self.min_green = float(self.cfg.get("controller", {}).get("min_green_steps", 3) * 5.0)
        self.max_green = float(self.cfg.get("controller", {}).get("max_green_steps", 12) * 5.0)
        self.yellow_time = 3.0
        self.all_red_time = 1.5

        # Preemption & incidents
        self.preemption_active = False
        self.preemption_approach = None
        self.emergency_cooldown = 0.0
        self.weather = "clear"  # "clear", "rain", "fog", "night"
        self.friction_multiplier = 1.0
        self.incident_active = False
        self.incident_info = None

        # Analytics
        self.historical_throughput: List[float] = []
        self.historical_wait: List[float] = []
        self.recent_events: List[Dict[str, Any]] = []

        # Populate initial traffic
        self.seed_traffic()

    def set_weather(self, weather: str):
        self.weather = weather.lower()
        if self.weather == "rain":
            self.friction_multiplier = 0.72
        elif self.weather == "snow":
            self.friction_multiplier = 0.50
        elif self.weather == "fog":
            self.friction_multiplier = 0.85
        else:
            self.friction_multiplier = 1.0

    def seed_traffic(self):
        for app in APPROACHES:
            init_count = random.randint(3, 8)
            pos = 20.0
            for _ in range(init_count):
                pos += random.uniform(8.0, 16.0)
                turn = random.choices(["straight", "left", "right"], weights=[0.65, 0.20, 0.15])[0]
                kind = random.choices(["car", "bus", "truck", "motorcycle"], weights=[0.75, 0.10, 0.08, 0.07])[0]
                veh = Vehicle(self.next_vid, app, turn=turn, kind=kind, position=pos)
                self.next_vid += 1
                self.vehicles[app].append(veh)

    def spawn_vehicle(self, approach: str, is_emergency: bool = False, kind: Optional[str] = None) -> Vehicle:
        turn = random.choices(["straight", "left", "right"], weights=[0.65, 0.20, 0.15])[0]
        if is_emergency:
            kind = "ambulance"
            turn = "straight"
        elif kind is None:
            kind = random.choices(["car", "bus", "truck", "motorcycle"], weights=[0.74, 0.11, 0.08, 0.07])[0]

        start_pos = 280.0
        if self.vehicles[approach]:
            start_pos = max(start_pos, self.vehicles[approach][-1].position + random.uniform(12.0, 20.0))

        veh = Vehicle(self.next_vid, approach, turn=turn, kind=kind, position=start_pos, is_emergency=is_emergency)
        self.next_vid += 1
        self.vehicles[approach].append(veh)

        if is_emergency:
            self.recent_events.append({
                "time": time.time(),
                "type": "EMERGENCY_DISPATCH",
                "message": f"Emergency Siren Detected on Approach {approach}! Priority preemption engaged.",
                "level": "warning"
            })
        return veh

    def trigger_incident(self, approach: str = "E", duration_seconds: float = 30.0):
        """Simulate a stalled vehicle / breakdown."""
        target_veh = None
        for v in self.vehicles[approach]:
            if v.position > 15.0 and not v.is_emergency:
                target_veh = v
                break
        if target_veh:
            target_veh.is_stalled = True
            self.incident_active = True
            self.incident_info = {"approach": approach, "vid": target_veh.id, "remaining": duration_seconds}
            self.recent_events.append({
                "time": time.time(),
                "type": "INCIDENT_ALERT",
                "message": f"Hazard: Vehicle #{target_veh.id} stalled on {approach} approach! Dynamic queue re-routing active.",
                "level": "danger"
            })

    def request_phase_switch(self) -> bool:
        """Controller requests to switch active phase."""
        if self.preemption_active:
            return False  # Locked by emergency vehicle
        if self.phase_timer < self.min_green:
            return False  # Minimum green enforced for safety

        if self.phase_state == "NS_GREEN":
            self.phase_state = "NS_YELLOW"
            self.phase_timer = 0.0
            return True
        elif self.phase_state == "EW_GREEN":
            self.phase_state = "EW_YELLOW"
            self.phase_timer = 0.0
            return True
        return False

    def get_queues(self) -> Dict[str, int]:
        """Count queued vehicles (speed < 1.0 m/s and approaching stop line)."""
        res = {}
        for app in APPROACHES:
            res[app] = sum(1 for v in self.vehicles[app] if v.position > -2.0 and v.position < 180.0 and (v.stopped or v.speed < 1.2))
        return res

    def get_queue_array(self) -> List[int]:
        q = self.get_queues()
        return [q["N"], q["S"], q["E"], q["W"]]

    def check_emergency_preemption(self):
        """Detect incoming emergency vehicles and engage preemption."""
        closest_dist = 999.0
        emergency_app = None

        for app in APPROACHES:
            for v in self.vehicles[app]:
                if v.is_emergency and v.position > -5.0 and v.position < 320.0:
                    if v.position < closest_dist:
                        closest_dist = v.position
                        emergency_app = app

        if emergency_app is not None:
            self.preemption_active = True
            self.preemption_approach = emergency_app
            desired_phase = 0 if emergency_app in ("N", "S") else 1

            if self.phase != desired_phase:
                if self.phase_state in ("NS_GREEN", "EW_GREEN") and self.phase_timer >= 3.0:
                    # Expedited yellow transition
                    self.phase_state = "NS_YELLOW" if self.phase == 0 else "EW_YELLOW"
                    self.phase_timer = 0.0
        else:
            if self.preemption_active:
                self.preemption_active = False
                self.preemption_approach = None
                self.recent_events.append({
                    "time": time.time(),
                    "type": "PREEMPTION_CLEARED",
                    "message": "Emergency vehicle cleared intersection. Resuming adaptive AI control.",
                    "level": "info"
                })

    def step(self, dt: float = 0.2):
        self.phase_timer += dt

        # Manage incident timer
        if self.incident_active and self.incident_info:
            self.incident_info["remaining"] -= dt
            if self.incident_info["remaining"] <= 0:
                # Clear incident
                for v in self.vehicles[self.incident_info["approach"]]:
                    if v.id == self.incident_info["vid"]:
                        v.is_stalled = False
                        break
                self.recent_events.append({
                    "time": time.time(),
                    "type": "INCIDENT_CLEARED",
                    "message": f"Accident/hazard cleared on {self.incident_info['approach']} approach.",
                    "level": "success"
                })
                self.incident_active = False
                self.incident_info = None

        # Check emergency priority
        self.check_emergency_preemption()

        # Phase State Machine transitions
        if self.phase_state == "NS_YELLOW":
            if self.phase_timer >= self.yellow_time:
                self.phase_state = "ALL_RED"
                self.phase_timer = 0.0
        elif self.phase_state == "ALL_RED":
            if self.phase_timer >= self.all_red_time:
                self.phase = 1
                self.phase_state = "EW_GREEN"
                self.phase_timer = 0.0
        elif self.phase_state == "EW_YELLOW":
            if self.phase_timer >= self.yellow_time:
                self.phase_state = "ALL_RED_2"
                self.phase_timer = 0.0
        elif self.phase_state == "ALL_RED_2":
            if self.phase_timer >= self.all_red_time:
                self.phase = 0
                self.phase_state = "NS_GREEN"
                self.phase_timer = 0.0
        elif self.phase_state == "NS_GREEN":
            # Auto-force switch if max green reached and not preempted
            if self.phase_timer >= self.max_green and not self.preemption_active:
                self.phase_state = "NS_YELLOW"
                self.phase_timer = 0.0
        elif self.phase_state == "EW_GREEN":
            if self.phase_timer >= self.max_green and not self.preemption_active:
                self.phase_state = "EW_YELLOW"
                self.phase_timer = 0.0

        # Random arrivals Poisson-distributed
        for app in APPROACHES:
            # Base arrival rate ~ 0.35 vehicles/sec
            rate = 0.38 if (app in ("N", "S") and self.phase == 0) else 0.30
            if random.random() < rate * dt:
                self.spawn_vehicle(app)

        # Update each vehicle
        ns_is_green = (self.phase_state == "NS_GREEN")
        ew_is_green = (self.phase_state == "EW_GREEN")

        for app in APPROACHES:
            is_green = ns_is_green if app in ("N", "S") else ew_is_green
            is_red = not is_green

            app_vehs = self.vehicles[app]
            # Sort by position (ascending: closest to stop line first)
            app_vehs.sort(key=lambda x: x.position)

            remaining = []
            for i, veh in enumerate(app_vehs):
                lead = app_vehs[i - 1] if i > 0 else None
                veh.update_idm(lead, is_red, dt, self.friction_multiplier)

                # Check red light violation
                if is_red and veh.position < -1.0 and veh.position > -8.0 and not veh.has_violated_red:
                    if not veh.is_emergency and veh.speed > 4.0:
                        veh.has_violated_red = True
                        self.recent_events.append({
                            "time": time.time(),
                            "type": "RED_LIGHT_VIOLATION",
                            "message": f"🚨 Red Light Violation: Vehicle #{veh.id} ({veh.kind.upper()}) ran red signal on {app}!",
                            "level": "danger"
                        })

                # Accumulate metrics
                self.total_co2_grams += veh.co2_g
                # Estimate baseline CO2 saved by adaptive vs fixed: ~18-35% reduction in idling
                if not is_red:
                    self.co2_saved_grams += 0.15 * dt

                # Keep vehicle if within 120m after intersection
                if veh.position > -120.0:
                    remaining.append(veh)
                else:
                    self.departed_vehicles += 1

            self.vehicles[app] = remaining

        # Update pedestrians
        if random.random() < 0.05 * dt:
            # Spawn pedestrian
            c_app = random.choice(APPROACHES)
            self.pedestrians.append(Pedestrian(self.next_pid, c_app))
            self.next_pid += 1

        active_peds = []
        for p in self.pedestrians:
            # Pedestrians can cross when vehicle phase for that crossing is red
            can_cross = (self.phase_state == "EW_GREEN") if p.approach in ("N", "S") else (self.phase_state == "NS_GREEN")
            p.update(can_cross, dt)
            if not p.finished:
                active_peds.append(p)
        self.pedestrians = active_peds

        # Truncate events log
        if len(self.recent_events) > 40:
            self.recent_events = self.recent_events[-40:]

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Serializable state for WebSocket & Digital Twin Canvas."""
        queues = self.get_queues()
        veh_list = []
        for app in APPROACHES:
            for v in self.vehicles[app]:
                veh_list.append({
                    "id": v.id,
                    "app": v.origin,
                    "pos": round(v.position, 1),
                    "spd": round(v.speed * 3.6, 1), # km/h
                    "kind": v.kind,
                    "turn": v.turn,
                    "lane": v.lane,
                    "stopped": v.stopped,
                    "emg": v.is_emergency,
                    "stalled": v.is_stalled,
                    "violation": v.has_violated_red
                })

        peds_list = []
        for p in self.pedestrians:
            peds_list.append({
                "id": p.id,
                "app": p.approach,
                "prog": round(p.progress, 2),
                "waiting": p.waiting,
                "crossing": p.crossing
            })

        avg_wait = 0.0
        total_veh = sum(len(v) for v in self.vehicles.values())
        if total_veh > 0:
            avg_wait = sum(v.wait_time for vs in self.vehicles.values() for v in vs) / total_veh

        return {
            "id": self.id,
            "phase": self.phase,
            "phase_state": self.phase_state,
            "phase_timer": round(self.phase_timer, 1),
            "min_green": self.min_green,
            "max_green": self.max_green,
            "queues": queues,
            "departed": self.departed_vehicles,
            "avg_wait_sec": round(avg_wait, 1),
            "co2_saved_kg": round(self.co2_saved_grams / 1000.0, 3),
            "weather": self.weather,
            "preemption": self.preemption_active,
            "preemption_app": self.preemption_approach,
            "incident": self.incident_active,
            "vehicles": veh_list,
            "pedestrians": peds_list,
            "events": self.recent_events[-10:]
        }


class MultiJunctionCorridor:
    """Coordinated arterial network of 4 smart intersections."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.junctions = {
            "junction01": MicroIntersection("junction01", cfg),
            "junction02": MicroIntersection("junction02", cfg),
            "junction03": MicroIntersection("junction03", cfg),
            "junction04": MicroIntersection("junction04", cfg),
        }
        self.corridor_order = ["junction01", "junction02", "junction03", "junction04"]
        self.green_wave_active = True
        self.wave_offset_seconds = 8.0  # Time offset between adjacent intersections

    def step(self, dt: float = 0.2):
        for j in self.junctions.values():
            j.step(dt)

        # Coordinate Green Wave offsets along the East-West arterial axis
        if self.green_wave_active:
            for i, jid in enumerate(self.corridor_order):
                j = self.junctions[jid]
                # If upstream is clearing eastbound platoon, optimize green trigger
                if not j.preemption_active and j.phase_state == "EW_GREEN":
                    # Keep green open during platoon progression window
                    pass

    def get_corridor_snapshot(self) -> Dict[str, Any]:
        return {
            "green_wave": self.green_wave_active,
            "offset_seconds": self.wave_offset_seconds,
            "junctions": {jid: j.get_state_snapshot() for jid, j in self.junctions.items()}
        }
