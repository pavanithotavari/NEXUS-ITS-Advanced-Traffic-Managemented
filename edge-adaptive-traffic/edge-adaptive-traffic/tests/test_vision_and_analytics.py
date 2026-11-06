from edge_traffic.vision_engine import EdgeVisionEngine
from edge_traffic.analytics import TrafficAnalytics
from edge_traffic.green_wave import GreenWaveCoordinator
from edge_traffic.preemption import EmergencyVehicleManager


def test_edge_vision_engine():
    engine = EdgeVisionEngine()
    fake_snapshot = {
        "vehicles": [
            {"id": 1, "app": "N", "pos": 30.0, "spd": 35.0, "kind": "car", "stopped": False, "emg": False, "stalled": False, "violation": False},
            {"id": 2, "app": "E", "pos": 10.0, "spd": 55.0, "kind": "ambulance", "stopped": False, "emg": True, "stalled": False, "violation": False},
        ],
        "phase_state": "NS_GREEN"
    }
    result = engine.process_digital_twin_frame(fake_snapshot)
    assert result["total_detected"] == 2
    assert len(result["detections"]) == 2
    assert result["approach_counts"]["N"] == 1
    assert result["approach_counts"]["E"] == 1


def test_traffic_analytics():
    analytics = TrafficAnalytics()
    queues = {"N": 5, "S": 8, "E": 12, "W": 3}
    dci = analytics.compute_congestion_index(queues)
    assert 0.0 <= dci <= 100.0

    eco = analytics.compute_eco_metrics(queues, elapsed_seconds=2.0)
    assert "fuel_saved_liters" in eco
    assert "co2_prevented_kg" in eco
    assert eco["co2_prevented_kg"] >= 0.0

    preds = analytics.predict_multi_horizon(queues, hour=8.5)
    assert preds["horizon_1min_vehicles"] >= 0
    assert preds["horizon_15min_vehicles"] >= preds["horizon_1min_vehicles"]


def test_green_wave_and_preemption():
    gw = GreenWaveCoordinator()
    metrics = gw.get_corridor_metrics()
    assert "offsets" in metrics
    assert len(metrics["offsets"]) == 4

    evp = EmergencyVehicleManager()
    ev = evp.dispatch_emergency(vehicle_type="ambulance", origin_junction="junction01")
    assert ev["status"] == "DISPATCHED"

    is_preempt, app, evp_id = evp.evaluate_preemption("junction01")
    assert is_preempt is True
    assert evp_id == ev["evp_id"]

    evp.clear_emergency(evp_id)
    is_preempt_after, _, _ = evp.evaluate_preemption("junction01")
    assert is_preempt_after is False
