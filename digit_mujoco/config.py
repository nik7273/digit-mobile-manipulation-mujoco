"""Immutable tuning defaults. Distances are metres, times seconds, forces newtons."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PalmConfig:
    radius: float = 0.035
    friction: float = 0.7
    squeeze: float = 20
    position_gain: float = 800
    velocity_gain: float = 40
    posture_gain: float = 10
    posture_damping: float = 2
    hold_offset: tuple[float, float, float] = (0.35, 0, 0.05)
    release_clearance: float = 0.12
    minimum_contact_force: float = 1
    contact_loss_seconds: float = 0.1


@dataclass(frozen=True)
class PickupConfig:
    center_x: float = 0.42
    open_clearance: float = 0.06
    settle_seconds: float = 2
    approach_seconds: float = 2
    close_seconds: float = 1.5
    contact_seconds: float = 0.25
    grasp_timeout_seconds: float = 3
    friction_margin: float = 1.2
    lift_seconds: float = 2
    lift_height: float = 0.12
    support_ramp_seconds: float = 0.5
    orientation_gain: float = 400
    orientation_damping: float = 2


@dataclass(frozen=True)
class CarryConfig:
    speed: float = 0.2
    max_speed: float = 0.3
    lift_height: float = 0.16
    hold_seconds: float = 3
    ready_timeout_seconds: float = 6
    speed_ramp_seconds: float = 2
    minimum_clearance: float = 0.04
    ready_tilt_degrees: float = 15


@dataclass(frozen=True)
class TransferConfig:
    speed_scale: float = 2
    max_speed_scale: float = 2
    forward_speed: float = 0.18
    lateral_speed: float = 0.12
    velocity_filter_seconds: float = 0.25
    position_gain: float = 0.7
    velocity_damping: float = 0.5
    integral_gain: float = 0.08
    integral_limit: float = 0.12
    arrival_radius: float = 0.045
    arrival_speed: float = 0.10
    heading_gain: float = 200
    heading_damping: float = 20
    heading_moment_limit: float = 40
    turn_gain: float = 0.2
    pitch_gain: float = 80
    pitch_damping: float = 5
    level_gain: float = 30
    level_damping: float = 3
    level_force_limit: float = 12
    pickup_waypoint: tuple[float, float] = (0.19, 0)
    aisle_y: float = -2
    table_waypoint: tuple[float, float] = (3.69, -2)
    place_center: tuple[float, float, float] = (3.96, -2, 0.825)
    lift_height: float = 0.28
    minimum_base_height: float = 0.75
    settle_seconds: float = 2
    hold_seconds: float = 3
    lower_seconds: float = 4
    open_seconds: float = 2
    retract_seconds: float = 2
    grasp_half_span: float = 0.235
    table_penetration: float = 0.004
    retract_delta: tuple[float, float, float] = (-0.15, 0, 0.25)
    table_support_fraction: float = 0.8
    lower_timeout_seconds: float = 8


PALM = PalmConfig()
PICKUP = PickupConfig()
CARRY = CarryConfig()
TRANSFER = TransferConfig()
