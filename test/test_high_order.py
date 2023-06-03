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

# Mesh generation copied from test_quadrilateral_mesh from
# test_higher_order_mesh.py in dolfinx. And the same for the hexes.

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
    def coord_to_vertex(x, y):
        return (polynomial_order + 1) * y + x

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
        # Define a cell using DOLFINx ordering
        cell = [
            coord_to_vertex(i, j)
            for i, j in [(0, 0), (order, 0), (0, order), (order, order)]
        ]
        if order > 1:
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order))

            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j))

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


@pytest.mark.parametrize(
    ("polynomial_order", "quadrature_degree"), [(1, 2), (2, 4)]  # , (3, 2), (4, 3)]
)
@pytest.mark.parametrize("rhs", [rhs1, rhs2, rhs3, rhs4])
def test_high_order_hexes(polynomial_order, quadrature_degree, rhs):
    def coord_to_vertex(x, y, z):
        return (polynomial_order + 1) ** 2 * z + (polynomial_order + 1) * y + x

    def get_points(order, Nx, Ny, Nz):
        points = []
        points += [
            [i / order, j / order, 0]
            for j in range(order + 1)
            for i in range(order + 1)
        ]
        for k in range(1, order):
            points += [
                [i / order, j / order + 0.1, k / order]
                for j in range(order + 1)
                for i in range(order + 1)
            ]
        points += [
            [i / order, j / order, 1]
            for j in range(order + 1)
            for i in range(order + 1)
        ]

        # Combine to several cells (vertices doesn't have to be unique)
        all_points = []
        pnp = np.array(points)

        ex = np.array([1, 0, 0])
        for i in range(Nx):
            ptmp = pnp + i * ex
            all_points.append(ptmp.tolist())  # extend?
        all_points_x = flatten(all_points)

        ey = np.array([0, 1, 0])
        for j in range(1, Ny):
            for q in all_points_x:
                ptmp = np.array(q) + j * ey
                all_points.append([ptmp.tolist()])
        all_points_xy = flatten(all_points)

        ez = np.array([0, 0, 1])
        for k in range(1, Nz):
            for q in all_points_xy:
                ptmp = np.array(q) + k * ez
                all_points.append([ptmp.tolist()])
        all_points = flatten(all_points)

        assert len(all_points) == (order + 1) ** 3 * Nx * Ny * Nz

        return all_points

    def get_cells(order, Nx, Ny, Nz):
        # Define a cell using DOLFINx ordering
        cell = [
            coord_to_vertex(x, y, z)
            for x, y, z in [
                (0, 0, 0),
                (order, 0, 0),
                (0, order, 0),
                (order, order, 0),
                (0, 0, order),
                (order, 0, order),
                (0, order, order),
                (order, order, order),
            ]
        ]

        if order > 1:
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, 0, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, 0, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, order, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, order, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0, order))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i, order))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i, order))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order, order))

            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j, 0))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, 0, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(0, i, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(order, i, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, order, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j, order))

            for k in range(1, order):
                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(i, j, k))

        # Combine to several cells as done for the points
        all_cells = []
        cnp = np.array(cell)
        n = len(cell)

        for i in range(Nx):
            ctmp = cnp + n * i
            all_cells.append(ctmp.tolist())

        cells_x = all_cells.copy()
        offset_x = np.array(cells_x).max() + 1

        for j in range(1, Ny):
            for cc in cells_x:
                ctmp = np.array(cc) + j * offset_x
                all_cells.append(ctmp.tolist())

        cells_xy = all_cells.copy()
        offset_xy = np.array(cells_xy).max() + 1

        for k in range(1, Nz):
            for cc in cells_xy:
                ctmp = np.array(cc) + k * offset_xy
                all_cells.append(ctmp.tolist())

        assert len(all_cells) == Nx * Ny * Nz

        return all_cells

    cell_type = dolfinx.cpp.mesh.CellType.hexahedron
    fiat_element = FIAT.reference_element.UFCHexahedron()
    Nx = 2
    Ny = 3
    Nz = 4
    points = get_points(polynomial_order, Nx, Ny, Nz)
    cells = get_cells(polynomial_order, Nx, Ny, Nz)
    domain = ufl.Mesh(
        basix.ufl_wrapper.create_vector_element(
            "Q",
            "hexahedron",
            polynomial_order,
            gdim=3,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
    )

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)

    b, b_ref = assemble_vector_test(
        mesh, fiat_element, polynomial_order, quadrature_degree, rhs
    )
    assert np.linalg.norm(b.array - b_ref.array) / np.linalg.norm(b_ref.array) < 1e-10
