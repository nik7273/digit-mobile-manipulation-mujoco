"""Outcome reports for the hold, pickup, carry, and transfer experiments."""

import numpy as np


class HoldStats:
    """Sample at 100 Hz after a two-second settling period."""

    def __init__(self):
        self.samples = []

    def record(self, sim):
        if sim.data.time >= 2:
            self.samples.append(sim.measure())

    def summary(self):
        if not self.samples:
            return {"passed": False, "reason": "Need samples after the 2s settling period"}
        samples = self.samples
        positions = np.array([s["relative_position"] for s in samples])
        drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
        normal = np.array([s["normal_force"] for s in samples])
        upward = np.array([s["upward_force"] for s in samples])
        bilateral = float(np.mean(np.all(normal > 1, axis=1)))
        min_height = min(s["box_height"] for s in samples)
        min_base = min(s["base_height"] for s in samples)
        max_base = max(s["base_height"] for s in samples)
        tilt = max(s["tilt_degrees"] for s in samples)
        other = sum(s["other_contacts"] for s in samples)
        support = float(np.mean(upward.sum(axis=1)))
        weight = samples[0]["payload_weight"]
        passed = (
            min_height > 0.8
            and min_base > 0.8
            and max_base < 1.3
            and drift < 0.02
            and tilt < 15
            and bilateral > 0.99
            and other == 0
            and abs(support - weight) < 0.2 * weight
        )
        return dict(
            passed=bool(passed),
            measured_seconds=float(samples[-1]["time"] - samples[0]["time"]),
            min_box_height_m=min_height,
            max_relative_drift_m=drift,
            max_box_tilt_degrees=tilt,
            bilateral_contact_fraction=bilateral,
            mean_palm_normal_force_n=normal.mean(axis=0).tolist(),
            mean_palm_upward_force_n=upward.mean(axis=0).tolist(),
            other_box_contacts=other,
        )


class PickupStats:
    def __init__(self):
        self.hold = HoldStats()
        self.initial = None
        self.events = []
        self.min_base_height = float("inf")
        self.min_clearance = float("inf")
        self.max_platform_force = 0.0
        self.max_approach_palm_force = 0.0

    def record(self, sim):
        values = sim.measure()
        if self.initial is None:
            self.initial = values
        self.events = list(sim.events)
        self.min_base_height = min(self.min_base_height, values["base_height"])
        if sim.phase in ("settle", "approach"):
            self.max_approach_palm_force = max(
                self.max_approach_palm_force, float(np.max(values["normal_force"]))
            )
        if sim.phase == "hold" and sim.data.time - sim.phase_start >= 1:
            self.hold.samples.append(values)
            self.min_clearance = min(self.min_clearance, values["platform_clearance_m"])
            self.max_platform_force = max(self.max_platform_force, values["platform_force_n"])

    def summary(self):
        hold = self.hold.summary()
        initial_support = (
            self.initial is not None
            and self.initial["platform_force_n"] > 0.8 * self.initial["payload_weight"]
            and np.all(self.initial["normal_force"] < 1)
        )
        phases = [name for name, _ in self.events]
        complete = phases == ["settle", "approach", "close", "grasp", "lift", "hold"]
        passed = (
            hold["passed"]
            and initial_support
            and complete
            and hold.get("measured_seconds", 0) >= 3
            and self.min_base_height > 0.8
            and self.min_clearance > 0.04
            and self.max_platform_force < 0.01
            and self.max_approach_palm_force < 1
        )
        return dict(
            passed=bool(passed),
            initially_platform_supported=bool(initial_support),
            phase_events=[{"phase": name, "time": round(t, 3)} for name, t in self.events],
            min_platform_clearance_m=self.min_clearance if self.hold.samples else None,
            max_platform_force_during_hold_n=self.max_platform_force,
            max_palm_force_before_closing_n=self.max_approach_palm_force,
            hold=hold,
        )


class CarryStats:
    def __init__(self):
        self.pickup = PickupStats()
        self.samples = []
        self.unloaded_since = [None, None]
        self.footfalls = [0, 0]
        self.walk_origin = None
        self.box_origin = None
        self.grasp_origin = None

    def record(self, sim):
        if sim.phase != "carry":
            self.pickup.record(sim)
            return
        values = sim.measure()
        if self.walk_origin is None:
            self.walk_origin = sim.walk_origin.copy()
            self.box_origin = sim.box_origin.copy()
            self.grasp_origin = sim.grasp_origin.copy()
        self.samples.append(values)
        # Count landings only after at least 30 ms unloaded, suppressing contact flicker.
        for side, load in enumerate(values["foot_loads"]):
            if load < 5 and self.unloaded_since[side] is None:
                self.unloaded_since[side] = values["time"]
            elif load >= 5 and self.unloaded_since[side] is not None:
                if values["time"] - self.unloaded_since[side] >= 0.03:
                    self.footfalls[side] += 1
                self.unloaded_since[side] = None

    def summary(self):
        acquisition = self.pickup.summary()
        if not self.samples:
            return dict(passed=False, reason="No carrying samples", pickup=acquisition)
        samples = self.samples
        normal = np.array([v["normal_force"] for v in samples])
        upward = np.array([v["upward_force"] for v in samples])
        slip = max(float(np.linalg.norm(v["box_in_grasp"] - self.grasp_origin)) for v in samples)
        base_travel = float(self.walk_origin[0] - samples[-1]["base_position"][0])
        box_travel = float(self.box_origin[0] - samples[-1]["box_position"][0])
        tilt = max(v["tilt_degrees"] for v in samples)
        bilateral = float(np.mean(np.all(normal > 1, axis=1)))
        other = sum(v["other_contacts"] for v in samples)
        min_base = min(v["base_height"] for v in samples)
        max_base = max(v["base_height"] for v in samples)
        min_box = min(v["box_height"] for v in samples)
        support = float(np.mean(upward.sum(axis=1)))
        weight = samples[0]["payload_weight"]
        elapsed = float(samples[-1]["time"] - samples[0]["time"])
        # PickupStats expects 3s of settled hold; carrying starts after 3s total hold.
        pickup_ready = (
            acquisition["hold"]["passed"]
            and acquisition["initially_platform_supported"]
            and acquisition["min_platform_clearance_m"] > 0.04
        )
        passed = (
            pickup_ready
            and elapsed >= 8
            and base_travel > 0.5
            and box_travel > 0.5
            and abs(base_travel - box_travel) < 0.1
            and min(self.footfalls) >= 5
            and slip < 0.03
            and tilt < 20
            and bilateral > 0.99
            and other == 0
            and 0.8 < min_base
            and max_base < 1.3
            and min_box > 0.8
            and abs(support - weight) < 0.25 * weight
            and samples[-1]["table_separation_m"] > 0.1
        )
        return dict(
            passed=bool(passed),
            carry_seconds=elapsed,
            base_travel_m=base_travel,
            payload_travel_m=box_travel,
            footfalls=self.footfalls,
            max_grasp_slip_m=slip,
            max_box_tilt_degrees=tilt,
            bilateral_contact_fraction=bilateral,
            other_box_contacts=other,
            mean_palm_upward_force_n=upward.mean(axis=0).tolist(),
            final_table_separation_m=samples[-1]["table_separation_m"],
            pickup_hold=acquisition["hold"],
        )


class TransferStats:
    def __init__(self):
        self.samples = []
        self.events = []
        self.minimum_base = 10.0
        self.carry_samples = []

    def record(self, sim):
        self.events = list(sim.events)
        self.minimum_base = sim.minimum_base
        if sim.phase in ("clear-pile", "walk-to-table"):
            self.carry_samples.append(sim.measure())
        if sim.phase == "done" and sim.data.time - sim.phase_start >= 1:
            v = sim.measure()
            v["position"] = sim.data.qpos[sim.box_adr : sim.box_adr + 3].copy()
            self.samples.append(v)

    def summary(self):
        stable = len(self.samples) >= 200 and all(
            abs(v["platform_force_n"] - v["payload_weight"]) < 0.2 * v["payload_weight"]
            and np.max(v["normal_force"]) < 0.1
            and v["tilt_degrees"] < 10
            and abs(v["position"][0] - 4) < 0.10
            and abs(v["position"][1] + 2) < 0.25
            for v in self.samples
        )
        supported = bool(self.carry_samples) and all(
            v["other_contacts"] == 0 and np.min(v["normal_force"]) > 1 for v in self.carry_samples
        )
        return dict(
            passed=bool(stable and supported and self.minimum_base > 0.8),
            events=self.events,
            completion_seconds=next((t for phase, t in self.events if phase == "done"), None),
            min_base_height_m=self.minimum_base,
            released_samples=len(self.samples),
            carry_seconds=(self.carry_samples[-1]["time"] - self.carry_samples[0]["time"])
            if self.carry_samples
            else 0,
            max_carry_tilt_degrees=max((v["tilt_degrees"] for v in self.carry_samples), default=0),
            palm_only_carry=supported,
            final_table_support_n=self.samples[-1]["platform_force_n"] if self.samples else None,
            final_box_position=self.samples[-1]["position"].tolist() if self.samples else None,
        )
