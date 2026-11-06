from edge_traffic.micro_sim import MicroIntersection, MultiJunctionCorridor, Vehicle, Pedestrian


def test_vehicle_idm_physics():
    v = Vehicle(1, "N", speed=10.0, position=50.0)
    # Update with lead vehicle ahead
    lead = Vehicle(2, "N", speed=0.0, position=10.0)
    v.update_idm(lead, is_red_for_me=False, dt=0.5)
    # Must decelerate to avoid collision
    assert v.speed < 10.0
    assert v.position < 50.0


def test_micro_intersection_lifecycle():
    inter = MicroIntersection("junction01")
    assert inter.phase == 0
    assert inter.phase_state == "NS_GREEN"

    # Step simulation
    for _ in range(20):
        inter.step(dt=0.2)

    queues = inter.get_queues()
    assert "N" in queues
    assert "S" in queues
    assert "E" in queues
    assert "W" in queues

    snap = inter.get_state_snapshot()
    assert "vehicles" in snap
    assert "queues" in snap
    assert snap["id"] == "junction01"


def test_micro_intersection_preemption():
    inter = MicroIntersection("junction01")
    inter.phase = 0
    inter.phase_state = "NS_GREEN"
    inter.phase_timer = 5.0

    # Spawn emergency vehicle on East approach
    inter.spawn_vehicle("E", is_emergency=True)
    inter.step(dt=0.2)

    assert inter.preemption_active is True
    assert inter.preemption_approach == "E"


def test_corridor_coordination():
    corridor = MultiJunctionCorridor()
    assert len(corridor.junctions) == 4
    for _ in range(10):
        corridor.step(dt=0.2)
    snap = corridor.get_corridor_snapshot()
    assert "junctions" in snap
    assert len(snap["junctions"]) == 4
