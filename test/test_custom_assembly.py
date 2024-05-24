import dolfinx
import numpy as np
import customquad as cq
import ufl
import common

mesh, cell_vol, ds_cut, dx_cut, qr_data, cut_cells, cut_cell_tag = (
    common.assemble_scalar_setup()
)


def test_assemble_scalar_constant():

    # Constant integrand
    integrand = 1.0

    form = dolfinx.fem.form(integrand * ds_cut(cut_cell_tag))
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    exact_val = len(cut_cells) * cell_vol
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_cut)
    val = cq.assemble_scalar(form, qr_data)
    exact_val = len(cut_cells) * cell_vol
    assert abs(val - exact_val) / abs(val) < 1e-15


def test_assemble_scalar_function():

    # Function integrand
    x = ufl.SpatialCoordinate(mesh)
    integrand = 2 * x[0] + x[1]
    integrand2 = lambda x: 2 * x[0] + x[1]

    form = dolfinx.fem.form(integrand * ds_cut(cut_cell_tag))
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in cut_cells
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_cut)
    val = cq.assemble_scalar(form, qr_data)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in cut_cells
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15


def test_assemble_scalar_fem_function():

    # FEM function integrand
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    f = dolfinx.fem.Function(V)
    integrand2 = lambda x: 2 * x[0] + x[1]
    f.interpolate(integrand2)
    integrand = f

    form = dolfinx.fem.form(integrand * ds_cut(cut_cell_tag))
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in cut_cells
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_cut)
    val = cq.assemble_scalar(form, qr_data)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in cut_cells
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15
