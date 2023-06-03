import pytest
import dolfinx
from mpi4py import MPI
import numpy as np
import FIAT
from common import assemble_vector_test, rhs1, rhs2, rhs3, rhs4

# FIXME move common routines somewhere

@pytest.mark.parametrize("N", [[1, 1], [3, 1], [1, 3], [3, 4]])
@pytest.mark.parametrize("xmin", [[0, 0], [-0.25, -10.25]])
@pytest.mark.parametrize("xmax", [[1, 1], [1.25, 17.5]])
@pytest.mark.parametrize("rhs", [rhs1, rhs2, rhs3, rhs4])
def test_linear_quads(N, xmin, xmax, rhs):
    polynomial_order = 1
    quadrature_degree = 2
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array(N),
        cell_type,
    )
    fiat_element = FIAT.reference_element.UFCQuadrilateral()

    b, b_ref = assemble_vector_test(
        mesh, fiat_element, polynomial_order, quadrature_degree, rhs
    )
    assert np.linalg.norm(b.array - b_ref.array) / np.linalg.norm(b_ref.array) < 1e-10


@pytest.mark.parametrize("N", [[1, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3], [3, 4, 5]])
@pytest.mark.parametrize("xmin", [[0, 0, 0], [-0.25, -10.25, 0.25]])
@pytest.mark.parametrize("xmax", [[1, 1, 1], [1.25, 17.5, 4.4]])
@pytest.mark.parametrize("rhs", [rhs1, rhs2, rhs3, rhs4])
def test_linear_hexes(N, xmin, xmax, rhs):
    polynomial_order = 1
    quadrature_degree = 2
    cell_type = dolfinx.mesh.CellType.hexahedron
    fiat_element = FIAT.reference_element.UFCHexahedron()
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array(N),
        cell_type,
    )

    b, b_ref = assemble_vector_test(
        mesh, fiat_element, polynomial_order, quadrature_degree, rhs
    )
    assert np.linalg.norm(b.array - b_ref.array) / np.linalg.norm(b_ref.array) < 1e-10


def test_normals():
    pass
