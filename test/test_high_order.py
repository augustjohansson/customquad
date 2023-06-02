import pytest
import basix
import dolfinx
import libcutfemx
import ufl
from ufl import grad, inner
from mpi4py import MPI
import numpy as np
from numpy import sin, pi, exp
from petsc4py import PETSc
import FIAT

# See test_quadrilateral_mesh test_higher_order_mesh.py

flatten = lambda l: [item for sublist in l for item in sublist]


def assemble_vector_test(mesh, fiat_element, polynomial_order, quadrature_degree, rhs):
    # Setup integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f.interpolate(rhs)
    L_eqn = inner(f, v)

    # Runtime quadrature
    L = L_eqn * ufl.dx(metadata={"quadrature_rule": "runtime"})
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(num_cells)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = np.tile(q.get_points().flatten(), [num_cells, 1])
    qr_w = np.tile(q.get_weights().flatten(), [num_cells, 1])
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_vector(L, [(cells, qr_pts, qr_w, qr_n)])

    # Reference
    L_ref = L_eqn * ufl.dx
    b_ref = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L_ref))

    return b, b_ref


def coord_to_vertex(order, x, y):
    return (order + 1) * y + x


def get_points(order, Nx, Ny):
    points = []
    points += [[i / order, 0] for i in range(order + 1)]
    for j in range(1, order):
        points += [[i / order + 0.1, j / order] for i in range(order + 1)]
    points += [[j / order, 1] for j in range(order + 1)]

    # Combine to several cells (test first w/o unique vertices)
    all_points = []
    pnp = np.array(points)

    ex = np.array([1.0, 0.0])
    for i in range(Nx):
        ptmp = pnp + i * ex
        all_points.append(ptmp.tolist())

    all_points_x = flatten(all_points)

    ey = np.array([0.0, 1.0])
    for j in range(1, Ny):
        for q in all_points_x:
            ptmp = np.array(q) + j * ey
            all_points.append([ptmp.tolist()])

    all_points = flatten(all_points)

    assert len(all_points) == (order + 1) ** 2 * Nx * Ny

    return all_points


def get_cells(order, Nx, Ny):
    cell = [
        coord_to_vertex(order, i, j)
        for i, j in [(0, 0), (order, 0), (0, order), (order, order)]
    ]
    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, order, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i, order))

        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i, j))

    # Combine to several cells as done for the points
    all_cells = []
    cnp = np.array(cell)
    n = len(cell)

    for i in range(Nx):
        ctmp = cnp + n * i
        all_cells.append(ctmp.tolist())

    cells_x = all_cells.copy()
    offset = np.array(cells_x).max() + 1

    for j in range(1, Ny):
        for cc in cells_x:
            ctmp = np.array(cc) + j * offset
            all_cells.append(ctmp.tolist())

    assert len(all_cells) == Nx * Ny

    return all_cells


def rhs1(x):
    return x[0] ** 0


def rhs2(x):
    return x[0]


def rhs3(x):
    return sin(pi * x[0]) * sin(pi * x[1])


def rhs4(x):
    return exp(x[0] * x[1])


@pytest.mark.parametrize(
    ("polynomial_order", "quadrature_degree"), [(1, 2), (2, 4)]  # , (3, 2), (4, 3)]
)
@pytest.mark.parametrize("rhs", [rhs1, rhs2, rhs3, rhs4])
def test_high_order_quads(polynomial_order, quadrature_degree, rhs):
    cell_type = dolfinx.cpp.mesh.CellType.quadrilateral
    fiat_element = FIAT.reference_element.UFCQuadrilateral()
    Nx = 2
    Ny = 3
    points = get_points(polynomial_order, Nx, Ny)
    cells = get_cells(polynomial_order, Nx, Ny)
    domain = ufl.Mesh(
        basix.ufl_wrapper.create_vector_element(
            "Q",
            "quadrilateral",
            polynomial_order,
            gdim=2,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
    )
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)

    b, b_ref = assemble_vector_test(
        mesh, fiat_element, polynomial_order, quadrature_degree, rhs
    )
    assert np.linalg.norm(b.array - b_ref.array) / np.linalg.norm(b_ref.array) < 1e-10
