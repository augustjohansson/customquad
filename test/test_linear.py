import pytest
import dolfinx
from mpi4py import MPI
import numpy as np
import FIAT
from common import (
    assemble_scalar_test,
    assemble_vector_test,
    assemble_matrix_test,
    fcn1,
    fcn2,
    fcn3,
    fcn4,
    scalar_norm,
    vector_norm,
    matrix_norm,
)


@pytest.mark.parametrize(
    ("assembler, norm"),
    [
        (assemble_scalar_test, scalar_norm),
        (assemble_vector_test, vector_norm),
        (assemble_matrix_test, matrix_norm),
    ],
)
@pytest.mark.parametrize("N", [[1, 1], [3, 1], [1, 3], [3, 4]])
@pytest.mark.parametrize("xmin", [[0, 0], [-0.25, -10.25]])
@pytest.mark.parametrize("xmax", [[1, 1], [1.25, 17.5]])
@pytest.mark.parametrize("fcn", [fcn1, fcn2, fcn3, fcn4])
def test_quads_assembly(assembler, norm, N, xmin, xmax, fcn):
    polynomial_order = 1
    quadrature_degree = 2
    fiat_element = FIAT.reference_element.UFCQuadrilateral()
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array(N),
        cell_type,
    )

    b, b_ref = assembler(mesh, fiat_element, polynomial_order, quadrature_degree, fcn)
    assert norm(b - b_ref) / norm(b_ref) < 1e-10


@pytest.mark.parametrize(
    ("assembler, norm"),
    [
        (assemble_scalar_test, scalar_norm),
        (assemble_vector_test, vector_norm),
        (assemble_matrix_test, matrix_norm),
    ],
)
@pytest.mark.parametrize("N", [[1, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3], [3, 4, 5]])
@pytest.mark.parametrize("xmin", [[0, 0, 0], [-0.25, -10.25, 0.25]])
@pytest.mark.parametrize("xmax", [[1, 1, 1], [1.25, 17.5, 4.4]])
@pytest.mark.parametrize("fcn", [fcn1, fcn2, fcn3, fcn4])
def test_hexes_assembly(assembler, norm, N, xmin, xmax, fcn):
    polynomial_order = 1
    quadrature_degree = 2
    fiat_element = FIAT.reference_element.UFCHexahedron()
    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        np.array([xmin, xmax]),
        np.array(N),
        cell_type,
    )

    b, b_ref = assembler(mesh, fiat_element, polynomial_order, quadrature_degree, fcn)
    assert norm(b - b_ref) / norm(b_ref) < 1e-10
