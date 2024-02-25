import dolfinx
from mpi4py import MPI
import numpy as np
import customquad as cq


def test_hex_mesh_topology():

    N = 1
    bbmin = np.array([-0.25, -10.25, -4.4])
    bbmax = np.array([1.25, 17.5, 0.1])

    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh_dx = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        np.array([bbmin, bbmax]),
        np.array([N, N, N]),
        cell_type,
    )

    polynomial_order = 1
    debug = False
    mesh_cq = cq.create_mesh(
        np.array([bbmin, bbmax]), np.array([N, N, N]), polynomial_order, debug
    )

    dim = 3
    meshes = [mesh_dx, mesh_cq]

    for mesh in meshes:
        for d0 in range(dim + 1):
            for d1 in range(dim + 1):
                mesh.topology.create_connectivity(d0, d1)

    for d0 in range(1, dim + 1):
        for d1 in range(dim + 1):
            assert np.all(
                mesh_dx.topology.connectivity(d0, d1)
                == mesh_cq.topology.connectivity(d0, d1)
            )


def test_hex_mesh_geometry():

    N = 1
    bbmin = np.array([-0.25, -10.25, -4.4])
    bbmax = np.array([1.25, 17.5, 0.1])

    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh_dx = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        np.array([bbmin, bbmax]),
        np.array([N, N, N]),
        cell_type,
    )
    x = mesh_dx.geometry.x

    polynomial_order = 1
    debug = False
    mesh_cq = cq.create_mesh(
        np.array([bbmin, bbmax]), np.array([N, N, N]), polynomial_order, debug
    )
    y = mesh_cq.geometry.x

    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-14


def test_hex_mesh_functionspace():

    N = 1
    bbmin = np.array([-0.25, -10.25, -4.4])
    bbmax = np.array([1.25, 17.5, 0.1])

    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh_dx = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        np.array([bbmin, bbmax]),
        np.array([N, N, N]),
        cell_type,
    )

    polynomial_order = 1
    debug = False
    mesh_cq = cq.create_mesh(
        np.array([bbmin, bbmax]), np.array([N, N, N]), polynomial_order, debug
    )

    V_dx = dolfinx.fem.FunctionSpace(mesh_dx, ("Lagrange", 1))
    V_cq = dolfinx.fem.FunctionSpace(mesh_cq, ("Lagrange", 1))
    x = V_dx.tabulate_dof_coordinates()
    y = V_cq.tabulate_dof_coordinates()

    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-14
