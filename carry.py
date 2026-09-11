"""Pick up a box, then walk backward away from the table with the payload."""
from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from pickup import Pickup, PickupStats, blend
from demo import ROOT, positive
from standing_hold import hold_model, run_experiment


class Carry(Pickup):
    follow_camera = True

    def __init__(self, speed=.2, assets='amazon', squeeze=20):
        if not math.isfinite(speed) or not 0 < speed <= .3:
            raise ValueError('Backward speed must be finite and in (0, 0.3] m/s')
        if assets not in ('amazon', 'simple'):
            raise ValueError('Assets must be amazon or simple')
        self.speed = speed
        self.assets = assets
        model = hold_model(scene=ROOT / 'assets' / 'amazon_pickup.xml') if assets == 'amazon' else None
        super().__init__(squeeze=squeeze, model=model)

    def reset(self):
        super().reset()
        # Walking lowers the pelvis; leave extra clearance above the tabletop.
        self.lift_delta[2] = .16
        self.carry_offsets = None
        self.walk_origin = None
        self.box_origin = None
        self.grasp_origin = None
        self.carry_rotation = None
        self.carry_grasp_rotations = None
        self.foot_ids = [self.model.body(f'{side}-toe-roll').id for side in ('left', 'right')]
        self.floor_id = self.model.geom('floor').id

    def yaw_rotation(self):
        w,x,y,z = self.data.qpos[3:7]
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        c,s = np.cos(yaw), np.sin(yaw)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    def begin_carry(self):
        rotation = self.yaw_rotation()
        self.carry_offsets = (self.targets-self.data.qpos[:3]) @ rotation
        self.walk_origin = self.data.qpos[:3].copy()
        self.box_origin = self.data.qpos[self.box_adr:self.box_adr+3].copy()
        self.grasp_origin = self.measure()['box_in_grasp'].copy()
        self.carry_rotation = rotation.copy()
        self.carry_grasp_rotations = self.grasp_rotations.copy()
        self.transition('carry')
        self.controller.set_mode(2)

    def update_task(self):
        if self.phase != 'carry':
            super().update_task()
            if self.phase == 'hold' and self.data.time-self.phase_start >= 3:
                values = self.measure()
                ready = (np.all(values['normal_force'] > 1.2*self.payload_weight/(2*.7))
                         and values['other_contacts'] == 0 and values['platform_clearance_m'] > .04
                         and values['tilt_degrees'] < 15)
                if ready:
                    self.begin_carry()
                elif self.data.time-self.phase_start > 6:
                    raise RuntimeError('Payload is not securely held clear of the table; refusing to walk')
        if self.phase == 'carry':
            elapsed = self.data.time-self.phase_start
            velocity = -self.speed * float(blend(0,1,elapsed,2)[0])
            self.controller.set_velocity(velocity,0,0)
            rotation = self.yaw_rotation()
            offsets_world = self.carry_offsets @ rotation.T
            self.targets = self.data.qpos[:3] + offsets_world
            # Follow translation/yaw, keeping the payload level through torso roll/pitch.
            body_rotation = self.data.xmat[self.base_id].reshape(3,3)
            angular = body_rotation @ self.data.qvel[3:6]
            forward = body_rotation[:,0]
            forward_rate = np.cross(angular, forward)
            yaw_rate = (forward[0]*forward_rate[1]-forward[1]*forward_rate[0]) / max(np.dot(forward[:2],forward[:2]), 1e-6)
            self.target_velocities = self.data.qvel[:3] + np.cross([0,0,yaw_rate], offsets_world)
            self.grasp_rotations = rotation @ self.carry_rotation.T @ self.carry_grasp_rotations
            values = self.measure()
            if values['other_contacts']:
                raise RuntimeError('Payload touched a surface outside the palms while carrying')
            if np.all(values['normal_force'] > 1):
                self.contact_lost_since = None
            elif self.contact_lost_since is None:
                self.contact_lost_since = self.data.time
            elif self.data.time-self.contact_lost_since > .1:
                raise RuntimeError('Grasp lost while carrying')

    def release(self):
        self.controller.set_velocity(0, 0, 0)
        super().release()

    def palm_command(self, side):
        if self.phase != 'carry':
            return super().palm_command(side)
        inward = self.yaw_rotation() @ [0, -self.squeeze if side == 0 else self.squeeze, 0]
        force = inward + [0,0,self.payload_weight/2]
        return self.targets[side], self.target_velocities[side], force

    def camera_target(self):
        target = .5*(self.data.qpos[:3] + self.data.qpos[self.box_adr:self.box_adr+3])
        target[2] = .85
        return target

    def measure(self):
        values = super().measure()
        m, d = self.model, self.data
        box = d.qpos[self.box_adr:self.box_adr+3]
        palms_midpoint = d.site_xpos[self.sites].mean(axis=0)
        foot_loads = np.zeros(2)
        wrench = np.zeros(6)
        for i, contact in enumerate(d.contact):
            if self.floor_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
            body = m.geom_bodyid[other]
            if body in self.foot_ids:
                mujoco.mj_contactForce(m, d, i, wrench)
                foot_loads[self.foot_ids.index(body)] += wrench[0]
        table_rear = m.geom_pos[self.platform,0]-m.geom_size[self.platform,0]
        box_front = box[0] + np.abs(d.xmat[self.box_id].reshape(3,3)[0]) @ self.box_halfsize
        values.update(box_position=box.copy(), base_position=d.qpos[:3].copy(),
                      box_in_grasp=self.yaw_rotation().T @ (box-palms_midpoint),
                      foot_loads=foot_loads, table_separation_m=float(table_rear-box_front))
        return values


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
        if sim.phase != 'carry':
            self.pickup.record(sim)
            return
        values = sim.measure()
        if self.walk_origin is None:
            self.walk_origin = sim.walk_origin.copy()
            self.box_origin = sim.box_origin.copy()
            self.grasp_origin = sim.grasp_origin.copy()
        self.samples.append(values)
        # Count landings only after at least 30 ms unloaded, suppressing contact flicker.
        for side, load in enumerate(values['foot_loads']):
            if load < 5 and self.unloaded_since[side] is None:
                self.unloaded_since[side] = values['time']
            elif load >= 5 and self.unloaded_since[side] is not None:
                if values['time']-self.unloaded_since[side] >= .03:
                    self.footfalls[side] += 1
                self.unloaded_since[side] = None

    def summary(self):
        acquisition = self.pickup.summary()
        if not self.samples:
            return dict(passed=False, reason='No carrying samples', pickup=acquisition)
        samples = self.samples
        normal = np.array([v['normal_force'] for v in samples])
        upward = np.array([v['upward_force'] for v in samples])
        slip = max(float(np.linalg.norm(v['box_in_grasp']-self.grasp_origin)) for v in samples)
        base_travel = float(self.walk_origin[0]-samples[-1]['base_position'][0])
        box_travel = float(self.box_origin[0]-samples[-1]['box_position'][0])
        tilt = max(v['tilt_degrees'] for v in samples)
        bilateral = float(np.mean(np.all(normal > 1, axis=1)))
        other = sum(v['other_contacts'] for v in samples)
        min_base = min(v['base_height'] for v in samples)
        max_base = max(v['base_height'] for v in samples)
        min_box = min(v['box_height'] for v in samples)
        support = float(np.mean(upward.sum(axis=1)))
        weight = samples[0]['payload_weight']
        elapsed = float(samples[-1]['time']-samples[0]['time'])
        # PickupStats expects 3s of settled hold; carrying starts after 3s total hold.
        pickup_ready = (acquisition['hold']['passed'] and acquisition['initially_platform_supported']
                        and acquisition['min_platform_clearance_m'] > .04)
        passed = (pickup_ready and elapsed >= 8 and base_travel > .5 and box_travel > .5
                  and abs(base_travel-box_travel) < .1 and min(self.footfalls) >= 5
                  and slip < .03 and tilt < 20 and bilateral > .99 and other == 0
                  and .8 < min_base and max_base < 1.3 and min_box > .8
                  and abs(support-weight) < .25*weight and samples[-1]['table_separation_m'] > .1)
        return dict(passed=bool(passed), carry_seconds=elapsed, base_travel_m=base_travel,
                    payload_travel_m=box_travel, footfalls=self.footfalls,
                    max_grasp_slip_m=slip, max_box_tilt_degrees=tilt,
                    bilateral_contact_fraction=bilateral, other_box_contacts=other,
                    mean_palm_upward_force_n=upward.mean(axis=0).tolist(),
                    final_table_separation_m=samples[-1]['table_separation_m'],
                    pickup_hold=acquisition['hold'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=positive, default=30, help='simulation seconds, at least 20')
    parser.add_argument('--speed', type=positive, default=.2, help='backward velocity target, at most .3 m/s')
    parser.add_argument('--assets', choices=('amazon', 'simple'), default='amazon')
    parser.add_argument('--squeeze', type=positive, default=20, help='inward force per palm in newtons')
    parser.add_argument('--contacts', action='store_true')
    args = parser.parse_args()
    if args.duration < 20:
        parser.error('--duration must be at least 20 seconds to validate pickup and carrying')
    if args.speed > .3:
        parser.error('--speed must be at most .3 m/s for this experiment')
    run_experiment(Carry(args.speed, args.assets, args.squeeze), CarryStats(), args)


if __name__ == '__main__':
    main()
