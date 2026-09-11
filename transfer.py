"""Move the top Amazon box from the original pile to the original table."""
from __future__ import annotations

import argparse
import numpy as np
import mujoco

from demo import ROOT, positive
from robot_config import INITIAL_QPOS
from pickup import Pickup, blend
from standing_hold import add_palms, run_experiment
from carry import Carry


def transfer_model():
    # Preserve the original horizontal layout and mesh sizes. Recenter box
    # frames for impedance control, removing mass inferred from visual meshes.
    spec = mujoco.MjSpec.from_file(str(ROOT / 'assets/package_scene.xml'))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    for name in ('package', 'package2', 'package3', 'package4', 'package5'):
        old = spec.body(name)
        position = old.pos.copy() + [0, 0, .1]
        spec.delete(old)
        name = 'held-box' if name == 'package5' else name
        body = spec.worldbody.add_body(name=name, pos=position)
        body.add_freejoint(name=name)
        body.add_geom(name=f'{name}-visual', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='amazonbody',
                      material='amazonbody', pos=[0,0,-.1], quat=[.5,.5,.5,.5],
                      mass=0, contype=0, conaffinity=0)
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[.15,.2,.1],
                      mass=1, contype=4, conaffinity=15, condim=4,
                      friction=[.7,.01,.0001], solref=[.01,1], rgba=[0,0,0,0])
    # Ground the original .725 m table and replace its solid collision volume.
    spec.delete(spec.body('table'))
    spec.worldbody.add_geom(name='table-visual', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='tablebody',
                           material='tablebody', pos=[4,-2,.7], quat=[.5,.5,.5,.5],
                           contype=0, conaffinity=0)
    spec.worldbody.add_geom(name='pickup-platform', type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[4,-2,.7125], size=[.25,.5,.0125],
                           contype=1, conaffinity=15, friction=[.7,.005,.0001], rgba=[0,0,0,0])
    for x in (3.85,4.15):
        for y in (-2.35,-1.65):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x,y,.35],
                                   size=[.05,.05,.35], contype=1, conaffinity=15, rgba=[0,0,0,0])
    add_palms(spec)
    return spec.compile()


class Transfer(Pickup):
    follow_camera = True
    yaw_rotation = Carry.yaw_rotation

    def __init__(self, *, overview=False):
        self.overview = overview
        self.follow_camera = not overview
        self.camera_distance = 7 if overview else 2.5
        super().__init__(model=transfer_model())

    def camera_target(self):
        return np.array([1.6,-.8,.75]) if self.overview else Carry.camera_target(self)

    def reset(self):
        super().reset()
        self.data.qpos[:2] = INITIAL_QPOS[:2]
        # Restore the original free-box poses only at reset; all later motion
        # comes from gravity, contacts, and bounded robot motor torques.
        self.data.qpos[61:] = self.model.qpos0[61:]
        mujoco.mj_forward(self.model, self.data)
        self.targets = self.data.site_xpos[self.sites].copy()
        self.start_palms = self.targets.copy()
        self.travel_offsets = (self.targets-self.data.qpos[:3]) @ self.yaw_rotation()
        self.filtered_velocity = np.zeros(2)
        self.integral = np.zeros(2)
        self.destination = np.array([.19,0])
        self.lift_delta = np.array([0,0,.28])
        self.place_center = np.array([3.96,-2,.825])
        self.minimum_base = 10.
        self.foot_ids = [self.model.body(f'{side}-toe-roll').id for side in ('left','right')]
        self.floor_id = self.model.geom('floor').id
        self.grasp_moment = None
        self.grasp_axis = np.array([0.,1.,0.])

    def palm_command(self, side):
        target, velocity, force = super().palm_command(side)
        if self.phase in ('clear-pile','walk-to-table'):
            force = self.yaw_rotation() @ force
        if self.phase in ('lift','hold','clear-pile','walk-to-table','settle-at-table','lower'):
            rotation = self.data.xmat[self.box_id].reshape(3,3)
            error = .5*sum(np.cross(rotation[:,k], self.yaw_rotation()[:,k]) for k in range(3))
            box_dof = self.model.joint('held-box').dofadr[0]
            angular = rotation @ self.data.qvel[box_dof+3:box_dof+6]
            self.grasp_moment = float(80*np.dot(error, self.grasp_axis)-5*np.dot(angular, self.grasp_axis))
            moment = 30*error-3*angular
            correction = np.clip(np.array([-moment[2],0,moment[0]])/.47, -12,12)
            force = force + (1 if side == 0 else -1)*correction
        return target, velocity, force

    def orientation_moment(self, error, angular_velocity):
        if self.grasp_moment is None:
            return super().orientation_moment(error, angular_velocity)
        return self.grasp_axis*self.grasp_moment

    def motor_torques(self):
        torque = super().motor_torques()
        if self.phase in ('walk-to-pile','clear-pile','walk-to-table'):
            rotation = self.yaw_rotation()
            yaw = np.arctan2(rotation[1,0],rotation[0,0])
            angular = self.data.xmat[self.base_id].reshape(3,3) @ self.data.qvel[3:6]
            moment = np.clip(-200*yaw-20*angular[2], -40,40)
            # Steer the feet toward the route heading; upstream fixes hip yaw at zero.
            grounded = set()
            for contact in self.data.contact:
                if self.floor_id in (contact.geom1, contact.geom2) and contact.dist < 0:
                    other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
                    grounded.add(self.model.geom_bodyid[other])
            for side, idx in enumerate((1,7)):
                if self.foot_ids[side] not in grounded:
                    torque[idx] += .5*moment*self.data.xaxis[self.joint_ids[idx],2]
        return torque

    def start_walk(self, phase, destination):
        self.destination = np.array(destination)
        self.integral[:] = 0
        self.travel_offsets = (self.targets-self.data.qpos[:3]) @ self.yaw_rotation()
        self.travel_rotation = self.yaw_rotation().copy()
        self.travel_grasp = None if self.grasp_rotations is None else self.grasp_rotations.copy()
        self.controller.set_mode(2)
        self.transition(phase)

    def navigate(self):
        dt = self.model.opt.timestep
        self.filtered_velocity += dt/.25*(self.data.qvel[:2]-self.filtered_velocity)
        error = self.destination-self.data.qpos[:2]
        self.integral = np.clip(self.integral + .08*dt*error, -.12,.12)
        command = np.clip(.7*error - .5*self.filtered_velocity + self.integral, -.18,.18)
        command[1] = np.clip(command[1], -.12, .12)
        rotation = self.yaw_rotation()
        self.grasp_axis = rotation[:,1].copy()
        local = rotation[:2,:2].T @ command
        yaw = np.arctan2(rotation[1,0],rotation[0,0])
        self.controller.set_velocity(float(local[0]),float(local[1]),float(-.2*yaw))
        if self.travel_grasp is not None:
            self.grasp_rotations = rotation @ self.travel_rotation.T @ self.travel_grasp
        offsets = self.travel_offsets @ rotation.T
        self.targets = self.data.qpos[:3] + offsets
        self.target_velocities = np.tile(self.data.qvel[:3], (2,1))
        return np.linalg.norm(error) < .045 and np.linalg.norm(self.filtered_velocity) < .10

    def stop_walk(self, phase):
        self.controller.set_velocity(0,0,0)
        self.controller.set_mode(0)
        self.target_velocities[:] = 0
        self.transition(phase)

    def update_task(self):
        t = self.data.time
        elapsed = t-self.phase_start
        self.minimum_base = min(self.minimum_base, float(self.data.qpos[2]))
        if self.minimum_base < .75:
            raise RuntimeError(f'Robot fell during {self.phase}')
        if self.phase == 'settle':
            if elapsed >= 2:
                self.start_walk('walk-to-pile', [.19,0])
        elif self.phase == 'walk-to-pile':
            if self.navigate():
                self.stop_walk('settle-at-pile')
        elif self.phase == 'settle-at-pile':
            if elapsed >= 2:
                self.pick_center = self.data.qpos[self.box_adr:self.box_adr+3].copy()
                self.start_palms = self.data.site_xpos[self.sites].copy()
                self.closed_palms = np.array([self.pick_center+[0,s*.235,0] for s in (1,-1)])
                self.open_palms = self.closed_palms + [[0,.06,0],[0,-.06,0]]
                self.transition('approach')
        elif self.phase in ('approach','close','grasp','lift','hold'):
            super().update_task()
            if self.phase == 'hold' and t-self.phase_start >= 3:
                self.start_walk('clear-pile', [self.data.qpos[0],-2])
        elif self.phase in ('clear-pile','walk-to-table'):
            arrived = self.navigate()
            if self.phase == 'clear-pile' and arrived:
                self.destination = np.array([3.69,-2])
                self.integral[:] = 0
                self.transition('walk-to-table')
            elif self.phase == 'walk-to-table' and arrived:
                self.stop_walk('settle-at-table')
        elif self.phase == 'settle-at-table':
            if elapsed >= 2:
                self.lower_start = self.targets.copy()
                self.lower_end = np.array([self.place_center+[0,s*.235,-.004] for s in (1,-1)])
                self.transition('lower')
        elif self.phase == 'lower':
            self.targets,self.target_velocities = blend(self.lower_start,self.lower_end,elapsed,4)
            self.support_scale = 1-float(blend(0,1,elapsed,4)[0])
            if elapsed >= 4 and self.measure()['platform_force_n'] > .8*self.payload_weight:
                self.transition('open')
                self.open_start = self.targets.copy()
            elif elapsed > 8:
                raise RuntimeError('Table did not support the payload')
        elif self.phase == 'open':
            a = float(blend(0,1,elapsed,2)[0])
            self.squeeze_scale = 1-a
            self.targets,self.target_velocities = blend(self.open_start,self.open_start+[[0,.12,0],[0,-.12,0]],elapsed,2)
            self.grasp_rotations = None
            if elapsed >= 2:
                self.transition('done')
        if self.phase in ('clear-pile','walk-to-table','settle-at-table'):
            values = self.measure()
            if values['other_contacts']:
                raise RuntimeError(f'Payload collided during {self.phase}')
            if np.all(values['normal_force'] > 1):
                self.contact_lost_since = None
            elif self.contact_lost_since is None:
                self.contact_lost_since = t
            elif t-self.contact_lost_since > .1:
                raise RuntimeError(f'Grasp lost during {self.phase}')


class TransferStats:
    def __init__(self):
        self.samples = []
        self.events = []
        self.minimum_base = 10.
        self.carry_samples = []

    def record(self, sim):
        self.events = list(sim.events)
        self.minimum_base = sim.minimum_base
        if sim.phase in ('clear-pile','walk-to-table'):
            self.carry_samples.append(sim.measure())
        if sim.phase == 'done' and sim.data.time-sim.phase_start >= 1:
            v = sim.measure()
            v['position'] = sim.data.qpos[sim.box_adr:sim.box_adr+3].copy()
            self.samples.append(v)

    def summary(self):
        stable = len(self.samples) >= 200 and all(
            abs(v['platform_force_n']-v['payload_weight']) < .2*v['payload_weight']
            and np.max(v['normal_force']) < .1 and v['tilt_degrees'] < 10
            and abs(v['position'][0]-4)<.10 and abs(v['position'][1]+2)<.25
            for v in self.samples)
        supported = bool(self.carry_samples) and all(
            v['other_contacts'] == 0 and np.min(v['normal_force']) > 1
            for v in self.carry_samples)
        bounded_tilt = all(v['tilt_degrees'] < 35 for v in self.carry_samples)
        return dict(passed=bool(stable and supported and bounded_tilt and self.minimum_base > .8), events=self.events,
                    min_base_height_m=self.minimum_base, released_samples=len(self.samples),
                    carry_seconds=(self.carry_samples[-1]['time']-self.carry_samples[0]['time']) if self.carry_samples else 0,
                    max_carry_tilt_degrees=max((v['tilt_degrees'] for v in self.carry_samples), default=0),
                    palm_only_carry=supported,
                    final_table_support_n=self.samples[-1]['platform_force_n'] if self.samples else None,
                    final_box_position=self.samples[-1]['position'].tolist() if self.samples else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=positive, default=100)
    parser.add_argument('--contacts', action='store_true')
    parser.add_argument('--overview', action='store_true', help='show the pile, route, and table together')
    args = parser.parse_args()
    run_experiment(Transfer(overview=args.overview), TransferStats(), args)


if __name__ == '__main__':
    main()
