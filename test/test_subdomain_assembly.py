import dolfinx
import numpy as np
import customquad as cq
import ufl
import common

mesh, cell_vol, dx_sub, dx_cut, qr_data, cut_cell_tag = common.setup_midpoint_qr()


def test_assemble_scalar_constant():

    # Constant integrand
    integrand = 1.0

    form = dolfinx.fem.form(integrand * dx_sub(cut_cell_tag))
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    exact_val = len(qr_data[0][0]) * cell_vol
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_sub)
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    exact_val = len(qr_data[0][0]) * cell_vol
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_cut)
    val = cq.assemble_scalar(form, qr_data)
    exact_val = len(qr_data[0][0]) * cell_vol
    assert abs(val - exact_val) / abs(val) < 1e-15


def test_assemble_scalar_function():

    # Function integrand
    x = ufl.SpatialCoordinate(mesh)
    integrand = 2 * x[0] + x[1]
    integrand2 = lambda x: 2 * x[0] + x[1]

    form = dolfinx.fem.form(integrand * dx_sub(cut_cell_tag))
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in qr_data[0][0]
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_sub)
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in qr_data[0][0]
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_cut)
    val = cq.assemble_scalar(form, qr_data)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in qr_data[0][0]
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

    # Assemble FE function over subdomain requires passing the
    # subdomain_id to cq.assemble_scalar
    form = dolfinx.fem.form(integrand * dx_sub(cut_cell_tag))
    val = cq.assemble_scalar(form, qr_data, cut_cell_tag)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in qr_data[0][0]
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15

    form = dolfinx.fem.form(integrand * dx_cut)
    val = cq.assemble_scalar(form, qr_data)
    conn = mesh.topology.connectivity(2, 0)
    exact_val = np.sum(
        [
            integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
            for c in qr_data[0][0]
        ]
    )
    assert abs(val - exact_val) / abs(val) < 1e-15


# def test_assemble_vector_fem_function():

#     # FEM function integrand
#     V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
#     v = ufl.TestFunction(V)
#     f = dolfinx.fem.Function(V)
#     integrand2 = lambda x: 2 * x[0] + x[1]
#     f.interpolate(integrand2)
#     integrand = f * v

#     form = dolfinx.fem.form(integrand * dx_sub(cut_cell_tag))
#     val = cq.assemble_vector(form, qr_data, cut_cell_tag)
#     conn = mesh.topology.connectivity(2, 0)
#     x = V.tabulate_dof_coordinates()
#     exact_val = np.zeros(x.shape[0])
#     basis_val = 0.25
#     for c in qr_data[0][0]:
#         exact_val[conn.links(c)] += (
#             integrand2(np.mean(x[conn.links(c)], axis=0)) * basis_val * cell_vol
#         )

#     assert np.linalg.norm(val.array - exact_val) / np.linalg.norm(val.array) < 1e-15

#     form = dolfinx.fem.form(integrand * dx_cut)
#     val = cq.assemble_vector(form, qr_data)
#     conn = mesh.topology.connectivity(2, 0)
#     exact_val = [
#         integrand2(np.mean(mesh.geometry.x[conn.links(c)], axis=0)) * cell_vol
#         for c in qr_data[0][0]
#     ]
#     breakpoint()
#     assert abs(val - exact_val) / abs(val) < 1e-15
#     breakpoint()
