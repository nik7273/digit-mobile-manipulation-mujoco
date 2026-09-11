"""Shared mesh inspection helpers."""


def mesh_bounds(sim, name):
    m, d = sim.model, sim.data
    geom = m.geom(name).id
    mesh = m.geom_dataid[geom]
    start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    points = (
        m.mesh_vert[start : start + count] @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
    )
    return points.min(axis=0), points.max(axis=0)
