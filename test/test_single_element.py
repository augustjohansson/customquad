import pytest
import dolfinx
import libcutfemx
import ufl
from ufl import grad, inner
from mpi4py import MPI
import numpy as np
from numpy import sin, pi, exp
from petsc4py import PETSc
import FIAT


def assemble_vector_test(mesh, fiat_element, polynomial_order, quadrature_degree, rhs):
    # Setup integral
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f.interpolate(rhs)
    L_eqn = inner(f, v)

    # Runtime quadrature
    L = L_eqn * ufl.dx(metadata={"quadrature_rule": "runtime"})
    cells = np.arange(1)
    q = FIAT.create_quadrature(fiat_element, quadrature_degree)
    qr_pts = q.get_points().flatten()
    qr_w = q.get_weights().flatten()
    qr_n = qr_pts  # dummy
    b = libcutfemx.custom_assemble_vector(L, [(cells, [qr_pts], [qr_w], [qr_n])])

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


# @pytest.mark.parametrize("cell", ["quadrilateral", "hexahedron"])
# @pytest.mark.parametrize(
#     ("polynomial_order", "quadrature_degree"),
#     [(1, 2)],  # , (2, 2), (3, 2), (4, 3), (5, 4)]
# )
# @pytest.mark.parametrize("xmin", [[0, 0, 0], [-0.25, -1.25, 0.25]])
# @pytest.mark.parametrize("xmax", [[1, 1, 1], [1.25, 7.5, 4.4]])
# @pytest.mark.parametrize("rhs", [rhs1, rhs2, rhs3, rhs4])
# def test_tensor_product_element(
#     cell, polynomial_order, quadrature_degree, xmin, xmax, rhs
# ):
#     b, b_ref = single_tensor_product_element(
#         cell, polynomial_order, quadrature_degree, xmin, xmax, rhs
#     )
#     assert np.linalg.norm(b.array - b_ref.array) / np.linalg.norm(b_ref.array) < 1e-10


@pytest.mark.parametrize("xmin", [[0, 0], [-0.25, -10.25]])
@pytest.mark.parametrize("xmax", [[1, 1], [1.25, 17.5]])
@pytest.mark.parametrize("rhs", [rhs1, rhs2, rhs3, rhs4])
def test_single_linear_quad(xmin, xmax, rhs):
    polynomial_order = 1
    quadrature_degree = 2
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array([1, 1]),
        cell_type,
    )
    fiat_element = FIAT.reference_element.UFCQuadrilateral()

    b, b_ref = assemble_vector_test(
        mesh, fiat_element, polynomial_order, quadrature_degree, rhs
    )
    assert np.linalg.norm(b.array - b_ref.array) / np.linalg.norm(b_ref.array) < 1e-10


@pytest.mark.parametrize("xmin", [[0, 0, 0], [-0.25, -10.25, 0.25]])
@pytest.mark.parametrize("xmax", [[1, 1, 1], [1.25, 17.5, 4.4]])
@pytest.mark.parametrize("rhs", [rhs1, rhs2, rhs3, rhs4])
def test_single_linear_hex(xmin, xmax, rhs):
    polynomial_order = 1
    quadrature_degree = 2
    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array([1, 1, 1]),
        cell_type,
    )
    fiat_element = FIAT.reference_element.UFCHexahedron()

    b, b_ref = assemble_vector_test(
        mesh, fiat_element, polynomial_order, quadrature_degree, rhs
    )
    assert np.linalg.norm(b.array - b_ref.array) / np.linalg.norm(b_ref.array) < 1e-10
