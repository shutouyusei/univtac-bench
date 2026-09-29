"""Contact wrench on a gel pad from libuipc's per-vertex contact gradients.

libuipc reports, per contact primitive type, the gradient of the contact energy
with respect to every vertex it touched, indexed by the vertex's global index.
The force the other body exerts on a vertex is minus that gradient. These
helpers select the gel's share of those rows and reduce them to one force and
one torque; they are plain numpy so they can be tested without the simulator.

libuipc minimises an incremental potential in which every potential energy,
the contact barrier included, is multiplied by dt^2 (``kappa * dt * dt`` in its
contact models), and it exports the gradients of that scaled energy unchanged.
A reported gradient is therefore a force in newtons times dt^2.
"""

import numpy as np


def gel_vertex_forces(vertex_ids, gradients, offset: int, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose global vertex index falls in ``[offset, offset + n_verts)``.

    Returns the gel-local vertex indices and the forces ``-gradient`` on them.
    """
    ids = np.asarray(vertex_ids).reshape(-1)
    grads = np.asarray(gradients, dtype=np.float64).reshape(-1, 3)
    if ids.shape[0] != grads.shape[0]:
        raise ValueError(f"{ids.shape[0]} vertex ids but {grads.shape[0]} gradients")
    hit = (ids >= offset) & (ids < offset + n_verts)
    return ids[hit] - offset, -grads[hit]


def accumulate_vertex_forces(n_verts: int, contributions) -> np.ndarray:
    """Sum ``(local_indices, forces)`` pairs into one ``(n_verts, 3)`` force per gel vertex."""
    per_vertex = np.zeros((n_verts, 3), dtype=np.float64)
    for local, forces in contributions:
        local = np.asarray(local).reshape(-1)
        if local.size:
            np.add.at(per_vertex, local, np.asarray(forces, dtype=np.float64).reshape(-1, 3))
    return per_vertex


def wrench_about(point, positions, forces) -> tuple[np.ndarray, np.ndarray]:
    """Net force and torque about ``point`` of point forces ``forces`` applied at ``positions``."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    if p.shape != f.shape:
        raise ValueError(f"positions {p.shape} and forces {f.shape} differ")
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return f.sum(axis=0), np.cross(p - point, f).sum(axis=0)


def from_incremental_potential(value, dt: float) -> np.ndarray:
    """A force or torque as libuipc reports it (scaled by ``dt**2``) in SI units (N, N m)."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return np.asarray(value, dtype=np.float64) / (dt * dt)
