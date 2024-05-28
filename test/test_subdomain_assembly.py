from petsc4py import PETSc
import numpy as np
import pytest
import dolfinx
import customquad as cq
import ufl
import common

mesh, cell_vol, dx_sub, dx_cut, qr_data, cut_cell_tag, celltags = (
    common.setup_midpoint_qr()
)


@pytest.mark.parametrize(
    "integrand", [1.0, dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0))]
)
@pytest.mark.parametrize(
    "measure",
    [
        # Use runtime qr with subdomain id set:
        dx_sub(cut_cell_tag),
        # For scalar assembly of non-fem integrands, the
        # subdomain_id=cut_cell_tag does not need to be set
        dx_sub,
        # Since the qr_data are over the desired domain, we can use
        # dx_cut, which just uses the qr_data and no subdomains (here
        # this is the same as dx_sub):
        dx_cut,
    ],
)
def test_assemble_scalar_constant(integrand, measure):

    # Calculate volume using dolfinx
    dx = ufl.dx(subdomain_data=celltags, metadata={"quadrature_degree": 1}, domain=mesh)
    form = dolfinx.fem.form(integrand * dx(cut_cell_tag))
    exact_val = dolfinx.fem.assemble_scalar(form)

    # # Calculate volume geometrically
    # exact_val2 = len(qr_data[0][0]) * cell_vol
    # assert abs(exact_val - exact_val2) / abs(exact_val) < 1e-15

    # Test
    form = dolfinx.fem.form(integrand * measure)
    val = cq.assemble_scalar(form, qr_data)
    assert abs(val - exact_val) / abs(exact_val) < 1e-15


@pytest.mark.parametrize(
    "measure",
    [
        # Use runtime qr with subdomain id set:
        dx_sub(cut_cell_tag),
        # For scalar assembly of non-fem integrands, the
        # subdomain_id=cut_cell_tag does not need to be set
        dx_sub,
        # Since the qr_data are over the desired domain, we can use
        # dx_cut, which just uses the qr_data and no subdomains (here
        # this is the same as dx_sub):
        dx_cut,
    ],
)
def test_assemble_scalar_function(measure):

    # Function integrand
    x = ufl.SpatialCoordinate(mesh)
    f = 2 * x[0] + x[1]
    f_lambda = lambda x: 2 * x[0] + x[1]

    # Exact value using manual integration
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            f_lambda(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in qr_data[0][0]
        ]
    )

    # # Exact value using dolfinx assembly
    # dx = ufl.dx(subdomain_data=celltags, metadata={"quadrature_degree": 1})
    # form = dolfinx.fem.form(f * dx(cut_cell_tag))
    # exact_val2 = dolfinx.fem.assemble_scalar(form)
    # assert abs(exact_val - exact_val2) / abs(exact_val) < 1e-15

    # Test
    form = dolfinx.fem.form(f * measure)
    val = cq.assemble_scalar(form, qr_data)
    assert abs(val - exact_val) / abs(exact_val) < 1e-15


@pytest.mark.parametrize(
    ("measure, debug_message"),
    [
        # Use runtime qr with subdomain id set:
        (dx_sub(cut_cell_tag), "FE coeffs are mapped"),
        # For scalar assembly of non-fem integrands, the
        # subdomain_id=cut_cell_tag does not need to be set
        (dx_sub, "No mapping -- no message"),
        # Since the qr_data are over the desired domain, we can use
        # dx_cut, which just uses the qr_data and no subdomains (here
        # this is the same as dx_sub):
        (dx_cut, "No mapping -- no message"),
    ],
)
def test_assemble_scalar_fem_function(measure, debug_message):

    # FEM function integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    f = dolfinx.fem.Function(V)
    f_lambda = lambda x: 2 * x[0] + x[1]
    f.interpolate(f_lambda)
    integrand = f

    # Exact value using manual integration
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            f_lambda(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in qr_data[0][0]
        ]
    )

    # # Exact value using dolfinx assembly
    # dx = ufl.dx(subdomain_data=celltags, metadata={"quadrature_degree": 1})
    # form = dolfinx.fem.form(integrand * dx(cut_cell_tag))
    # exact_val2 = dolfinx.fem.assemble_scalar(form)
    # assert abs(exact_val - exact_val2) / abs(exact_val) < 1e-15

    # If we assemble FE functions over subdomains, then the fem_coeffs
    # are mapped. This should print the debug_message when running
    # something like pytest -s test_subdomain_assembly.py -k scalar_fem_function
    form = dolfinx.fem.form(integrand * measure)
    val = cq.assemble_scalar(form, qr_data, debug=True, debug_message=debug_message)
    assert abs(val - exact_val) / abs(exact_val) < 1e-15


@pytest.mark.parametrize(
    "measure",
    [
        dx_sub(cut_cell_tag),
        dx_cut,
    ],
)
def test_assemble_vector_fem_function(measure):

    # FEM function integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f_lambda = lambda x: 2 * x[0] + x[1]
    f.interpolate(f_lambda)
    integrand = f * v

    # Exact value using manual integration
    x = mesh.geometry.x
    exact_val = np.zeros(x.shape[0])
    basis_val = 0.25  # Evaluation of basis in midpoint
    conn = mesh.topology.connectivity(2, 0)
    for c in qr_data[0][0]:
        exact_val[conn.links(c)] += (
            f_lambda(np.mean(x[conn.links(c)], axis=0)) * basis_val * cell_vol
        )

    # # Exact value using dolfinx assembly
    # dx = ufl.dx(subdomain_data=celltags, metadata={"quadrature_degree": 1})
    # form = dolfinx.fem.form(integrand * dx(cut_cell_tag))
    # exact_val2 = dolfinx.fem.petsc.assemble_vector(form)
    # exact_val2.assemble()
    # assert (
    #     np.linalg.norm(exact_val - exact_val2.array) / np.linalg.norm(exact_val) < 1e-15
    # )

    form = dolfinx.fem.form(integrand * measure)
    val = cq.assemble_vector(form, qr_data)
    val.assemble()
    assert np.linalg.norm(val.array - exact_val) / np.linalg.norm(exact_val) < 1e-15


@pytest.mark.parametrize(
    "measure",
    [
        dx_sub(cut_cell_tag),  # Maps coeffs
        dx_cut,  # No mapping
    ],
)
def test_assemble_matrix_fem_function(measure):

    # FEM function integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Function(V)
    f_lambda = lambda x: 2 * x[0] + x[1]
    f.interpolate(f_lambda)
    integrand = f * u * v

    # Exact value using dolfinx assembly
    dx = ufl.dx(subdomain_data=celltags, metadata={"quadrature_degree": 1})
    form = dolfinx.fem.form(integrand * dx(cut_cell_tag))
    exact_val = dolfinx.fem.petsc.assemble_matrix(form)
    exact_val.assemble()

    form = dolfinx.fem.form(integrand * measure)
    val = cq.assemble_matrix(form, qr_data)
    val.assemble()
    assert (val - exact_val).norm() / (exact_val).norm() < 1e-15
