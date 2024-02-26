import pytest
import dolfinx
from mpi4py import MPI
import numpy as np
import customquad as cq


@pytest.mark.parametrize(
    ("gdim, cell_type, dx_mesher"),
    [
        (2, dolfinx.mesh.CellType.quadrilateral, dolfinx.mesh.create_rectangle),
        (3, dolfinx.mesh.CellType.hexahedron, dolfinx.mesh.create_box),
    ],
)
def test_cq_mesh_topology(gdim, cell_type, dx_mesher):

    N = [1, 1, 1]
    bbmin = np.array([-0.25, -10.25, -4.4])
    bbmax = np.array([1.25, 17.5, 0.1])

    N = N[:gdim]
    bbmin = bbmin[:gdim]
    bbmax = bbmax[:gdim]

    mesh_dx = dx_mesher(
        MPI.COMM_WORLD,
        np.array([bbmin, bbmax]),
        np.array(N),
        cell_type,
    )

    polynomial_order = 1
    debug = False
    mesh_cq = cq.create_mesh(
        np.array([bbmin, bbmax]), np.array(N), polynomial_order, debug
    )

    meshes = [mesh_dx, mesh_cq]

    for mesh in meshes:
        for d0 in range(gdim + 1):
            for d1 in range(gdim + 1):
                mesh.topology.create_connectivity(d0, d1)

    for d0 in range(1, gdim + 1):
        for d1 in range(gdim + 1):
            assert np.all(
                mesh_dx.topology.connectivity(d0, d1)
                == mesh_cq.topology.connectivity(d0, d1)
            )


@pytest.mark.parametrize(
    ("gdim, cell_type, dx_mesher"),
    [
        (2, dolfinx.mesh.CellType.quadrilateral, dolfinx.mesh.create_rectangle),
        (3, dolfinx.mesh.CellType.hexahedron, dolfinx.mesh.create_box),
    ],
)
def test_cq_mesh_geometry(gdim, cell_type, dx_mesher):

    N = [1, 1, 1]
    bbmin = np.array([-0.25, -10.25, -4.4])
    bbmax = np.array([1.25, 17.5, 0.1])

    N = N[:gdim]
    bbmin = bbmin[:gdim]
    bbmax = bbmax[:gdim]

    mesh_dx = dx_mesher(
        MPI.COMM_WORLD,
        np.array([bbmin, bbmax]),
        np.array(N),
        cell_type,
    )
    x = mesh_dx.geometry.x

    polynomial_order = 1
    debug = False
    mesh_cq = cq.create_mesh(
        np.array([bbmin, bbmax]), np.array(N), polynomial_order, debug
    )
    y = mesh_cq.geometry.x

    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-14


@pytest.mark.parametrize(
    ("gdim, cell_type, dx_mesher"),
    [
        (2, dolfinx.mesh.CellType.quadrilateral, dolfinx.mesh.create_rectangle),
        (3, dolfinx.mesh.CellType.hexahedron, dolfinx.mesh.create_box),
    ],
)
def test_cq_mesh_functionspace_dofs(gdim, cell_type, dx_mesher):

    N = [1, 1, 1]
    bbmin = np.array([-0.25, -10.25, -4.4])
    bbmax = np.array([1.25, 17.5, 0.1])

    N = N[:gdim]
    bbmin = bbmin[:gdim]
    bbmax = bbmax[:gdim]

    mesh_dx = dx_mesher(
        MPI.COMM_WORLD,
        np.array([bbmin, bbmax]),
        np.array(N),
        cell_type,
    )

    polynomial_order = 1
    debug = False
    mesh_cq = cq.create_mesh(
        np.array([bbmin, bbmax]), np.array(N), polynomial_order, debug
    )

    V_dx = dolfinx.fem.FunctionSpace(mesh_dx, ("Lagrange", 1))
    V_cq = dolfinx.fem.FunctionSpace(mesh_cq, ("Lagrange", 1))
    x = V_dx.tabulate_dof_coordinates()
    y = V_cq.tabulate_dof_coordinates()

    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-14
