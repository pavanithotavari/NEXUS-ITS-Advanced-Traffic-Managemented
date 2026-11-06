"""Arterial Green Wave Corridor Coordination & Offset Optimization.

Calculates speed-of-progression offsets between neighboring signalized intersections
to ensure continuous platooning flow, minimizing arterial stops, delays, and emissions.
"""
from typing import Dict, List, Any


class GreenWaveCoordinator:
    def __init__(
        self,
        corridor_junctions: List[str] = None,
        segment_lengths_meters: List[float] = None,
        progression_speed_kmh: float = 45.0,
        cycle_time_seconds: float = 60.0
    ):
        self.junctions = corridor_junctions or ["junction01", "junction02", "junction03", "junction04"]
        # Default: 400m between consecutive intersections
        self.segment_lengths = segment_lengths_meters or [400.0, 420.0, 380.0]
        self.progression_speed = progression_speed_kmh / 3.6  # m/s (~12.5 m/s)
        self.cycle_time = cycle_time_seconds
        self.offsets = self.compute_optimal_offsets()

    def compute_optimal_offsets(self) -> Dict[str, float]:
        """Compute signal offset in seconds relative to the reference junction."""
        offsets = {self.junctions[0]: 0.0}
        accum_dist = 0.0
        for i, jid in enumerate(self.junctions[1:]):
            accum_dist += self.segment_lengths[i]
            travel_time = accum_dist / self.progression_speed
            offset = travel_time % self.cycle_time
            offsets[jid] = round(offset, 1)
        return offsets

    def get_corridor_metrics(self) -> Dict[str, Any]:
        """Progression bandwidth and efficiency metrics."""
        bandwidth_seconds = 24.0
        efficiency_pct = round((bandwidth_seconds / self.cycle_time) * 100.0, 1)
        return {
            "progression_speed_kmh": round(self.progression_speed * 3.6, 1),
            "cycle_time_sec": self.cycle_time,
            "bandwidth_sec": bandwidth_seconds,
            "bandwidth_efficiency_pct": efficiency_pct,
            "offsets": self.offsets,
            "junction_sequence": self.junctions,
            "coordination_axis": "EAST_WEST_ARTERIAL"
        }
