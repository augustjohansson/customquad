import os
from contextlib import ExitStack
import argparse
import numpy as np
import dolfinx
import ufl
from ufl import nabla_grad, inner, dot, avg
from mpi4py import MPI
from petsc4py import PETSc
import customquad as cq
import algoim_utils

# Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=16)
parser.add_argument("-algoim", action="store_true")
parser.add_argument("-betaN", type=float, default=10.0)
parser.add_argument("-betas", type=float, default=1.0)
parser.add_argument("-domain", type=str, default="circle")
parser.add_argument("-p", type=int, default=1)
parser.add_argument("-order", type=int, default=2)
parser.add_argument("-verbose", action="store_true")
parser.add_argument("-output", type=str, default="output")
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print("\t", arg, getattr(args, arg))

os.makedirs(args.output, exist_ok=True)


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [dolfinx.la.create_petsc_vector(index_map, bs) for i in range(6)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]

        # Build the three translational rigid body modes
        for i in range(3):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        basis[3][dofs[0]] = -x1
        basis[3][dofs[1]] = x0
        basis[4][dofs[0]] = x2
        basis[4][dofs[2]] = -x0
        basis[5][dofs[2]] = x1
        basis[5][dofs[1]] = -x2

    # Orthonormalise the six vectors
    dolfinx.la.orthonormalize(ns)
    assert dolfinx.la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)


# Domain
if args.domain == "circle":
    xmin = np.array([-1.11, -1.51])
    xmax = np.array([1.55, 1.22])
    volume_exact = np.pi
    area_exact = 2 * np.pi

elif args.domain == "sphere":
    xmin = np.array([-1.11, -1.51, -1.23])
    xmax = np.array([1.55, 1.22, 1.11])
    volume_exact = 4 * np.pi / 3
    area_exact = 4 * np.pi

else:
    raise RuntimeError("Unknown domain", args.domain)

gdim = len(xmin)

# Mesh
NN = np.array([args.N] * gdim, dtype=np.int32)
mesh = cq.create_mesh(np.array([xmin, xmax]), NN, args.p, args.verbose)

# Generate qr
algoim_opts = {"verbose": args.verbose}
t = dolfinx.common.Timer()
[
    cut_cells,
    uncut_cells,
    outside_cells,
    qr_pts,
    qr_w,
    qr_pts_bdry,
    qr_w_bdry,
    qr_n,
    xyz,
    xyz_bdry,
] = algoim_utils.generate_qr(mesh, NN, args.order, args.domain, algoim_opts)

print("Generating qr took", t.elapsed()[0])
print("num cells", cq.utils.get_num_cells(mesh))
print("num cut_cells", len(cut_cells))
print("num uncut_cells", len(uncut_cells))
print("num outside_cells", len(outside_cells))

# Set up cell tags and face tags
uncut_cell_tag = 1
cut_cell_tag = 2
outside_cell_tag = 3
ghost_penalty_tag = 4
celltags = cq.utils.get_celltags(
    mesh,
    cut_cells,
    uncut_cells,
    outside_cells,
    uncut_cell_tag=uncut_cell_tag,
    cut_cell_tag=cut_cell_tag,
    outside_cell_tag=outside_cell_tag,
)
facetags = cq.utils.get_facetags(
    mesh, cut_cells, outside_cells, ghost_penalty_tag=ghost_penalty_tag
)

# Write mesh with tags
with dolfinx.io.XDMFFile(mesh.comm, args.output + f"/msh{args.N}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(celltags)
    xdmf.write_meshtags(facetags)


def u_exact(m):
    def u_fcn(x):
        if mesh.geometry.dim == 2:
            return [
                m.cos((m.pi * x[1])) * m.sin((m.pi * x[0])),
                m.sin((m.pi * x[0])) * m.sin((m.pi * x[1])),
            ]
        else:
            return [
                m.cos(m.pi * x[1]) * m.sin(m.pi * x[0]) * m.sin(m.pi * x[2]),
                m.sin(m.pi * x[0]) * m.sin(m.pi * x[1]) * m.sin(m.pi * x[2]),
                m.sin(m.pi * x[0]) * m.sin(m.pi * x[1]) * m.cos(m.pi * x[2]),
            ]

    if m == ufl:
        return ufl.as_vector(u_fcn(ufl.SpatialCoordinate(mesh)))
    elif m == np:
        return lambda x: np.stack(u_fcn(x))
    else:
        raise RuntimeError("Unknown mule for backend", m)


E = 1
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(v):
    return ufl.sym(nabla_grad(v))


def sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v))


# FEM
V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", args.p))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
g = dolfinx.fem.Function(V)
n = ufl.FacetNormal(mesh)
if args.p == 1:
    h = ufl.CellDiameter(mesh)
else:
    h = max((xmax - xmin) / args.N)

# Data
g = u_exact(ufl)
f = -ufl.nabla_div(sigma(u_exact(ufl)))

# PDE
betaN = args.betaN
betas = args.betas
a_bulk = inner(sigma(u), epsilon(v))
L_bulk = inner(f, v)
a_bdry = (
    -inner(dot(n, sigma(u)), v)
    - inner(u, dot(n, sigma(v)))
    + betaN / h * inner((2 * mu + lmbda) * u, v)
)
L_bdry = -inner(g, dot(n, sigma(v))) + betaN / h * inner((2 * mu + lmbda) * g, v)


def tensor_jump(v, n):
    return ufl.outer(v("+"), n("+")) + ufl.outer(v("-"), n("-"))


a_stab = (
    betas * avg(h) * inner(tensor_jump(nabla_grad(u), n), tensor_jump(nabla_grad(v), n))
)
if args.p == 2:
    a_stab += (
        betas
        * avg(h) ** 3
        * inner(
            tensor_jump(nabla_grad(nabla_grad(u)), n),
            tensor_jump(nabla_grad(nabla_grad(v)), n),
        )
    )


# Standard measures
dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
dS = ufl.dS(subdomain_data=facetags, domain=mesh)

# Integration using standard assembler (uncut cells, ghost penalty
# faces)
ax = dolfinx.fem.form(
    a_bulk * dx_uncut(uncut_cell_tag) + a_stab * dS(ghost_penalty_tag)
)
t = dolfinx.common.Timer()
Ax = dolfinx.fem.petsc.assemble_matrix(ax)
Ax.assemble()
print("Assemble interior took", t.elapsed()[0])
Lx = dolfinx.fem.form(L_bulk * dx_uncut(uncut_cell_tag))
bx = dolfinx.fem.petsc.assemble_vector(Lx)

# Integration using custom assembler (i.e. integrals over cut cells,
# both cut bulk part and bdry part)
dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
ds_cut = ufl.ds(
    subdomain_data=celltags, metadata={"quadrature_rule": "runtime"}, domain=mesh
)

qr_bulk = [(cut_cells, qr_pts, qr_w)]
qr_bdry = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]

form1 = dolfinx.fem.form(a_bulk * dx_cut)
form2 = dolfinx.fem.form(a_bdry * ds_cut(cut_cell_tag))

t = dolfinx.common.Timer()
Ac1 = cq.assemble_matrix(form1, qr_bulk)
Ac1.assemble()
print("Runtime assemble bulk took", t.elapsed()[0])

t = dolfinx.common.Timer()
Ac2 = cq.assemble_matrix(form2, qr_bdry)
Ac2.assemble()
print("Runtime assemble bdry took", t.elapsed()[0])

t = dolfinx.common.Timer()
A = Ax
A += Ac1
A += Ac2
print("Matrix += took", t.elapsed()[0])

L1 = dolfinx.fem.form(L_bulk * dx_cut)
bc1 = cq.assemble_vector(L1, qr_bulk)
b = bx
b += bc1

L2 = dolfinx.fem.form(L_bdry * ds_cut)
bc2 = cq.assemble_vector(L2, qr_bdry)
b += bc2

assert np.isfinite(b.array).all()
assert np.isfinite(A.norm())

# Lock inactive dofs
t = dolfinx.common.Timer()
inactive_dofs = cq.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
print("Get inactive_dofs took", t.elapsed()[0])
t = dolfinx.common.Timer()
A = cq.utils.lock_inactive_dofs(inactive_dofs, A)
print("Lock inactive dofs took", t.elapsed()[0])
assert np.isfinite(A.norm()).all()

uh = dolfinx.fem.Function(V)
uh.name = "uh"

if gdim == 2:
    # Direct solver using mumps
    print("Start solve using mumps")
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()  # type: ignore
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = (
        1  # Option to support solving a singular matrix (pressure nullspace)
    )
    opts["mat_mumps_icntl_25"] = (
        0  # Option to support solving a singular matrix (pressure nullspace)
    )
    opts["ksp_error_if_not_converged"] = 1
    ksp.setFromOptions()
    ksp.setOperators(A)
    vec = A.createVecRight()
    ksp.solve(b, vec)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    uh.vector.setArray(vec.array)

else:
    # Solve as in demo_elasticity.py in dolfinx
    null_space = build_nullspace(V)
    A.setNearNullSpace(null_space)

    # Set solver options
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-10
    opts["pc_type"] = "gamg"

    # Use Chebyshev smoothing for multigrid
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"

    # Improve estimate of eigenvalues for Chebyshev smoothing
    opts["mg_levels_esteig_ksp_type"] = "cg"
    opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

    # Create PETSc Krylov solver and turn convergence monitoring on
    solver = PETSc.KSP().create(mesh.comm)
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(A)

    # Set a monitor, solve linear system, and display the solver
    # configuration
    solver.setMonitor(
        lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
    )
    solver.solve(b, uh.vector)
    solver.view()

    # Scatter forward the solution vector to update ghost values
    uh.x.scatter_forward()


# Evaluate solution in qr to see that there aren't any spikes
pts = np.reshape(cq.utils.flatten(xyz), (-1, gdim))
pts_bdry = np.reshape(cq.utils.flatten(xyz_bdry), (-1, gdim))
pts_bulk = dolfinx.mesh.compute_midpoints(mesh, gdim, uncut_cells)
pts = np.append(pts, pts_bdry, axis=0)
pts = np.append(pts, pts_bulk[:, 0:gdim], axis=0)
if gdim == 2:
    # Pad with zero column
    pts = np.hstack((pts, np.zeros((pts.shape[0], 1))))

bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, gdim)
cell_candidates = dolfinx.cpp.geometry.compute_collisions(bb_tree, pts)
cells = dolfinx.cpp.geometry.compute_colliding_cells(mesh, cell_candidates, pts)
uh_vals = uh.eval(pts, cells.array)
for d in range(gdim):
    print(f"uh {d} in range", min(uh_vals[:, d]), max(uh_vals[:, d]))

if gdim == 2:
    # Save coordinates and solution for plotting
    filename = args.output + "/uu" + str(args.N) + ".txt"
    uu = np.empty((uh_vals.shape[0], 4))
    uu[:, 0:2] = pts[:, 0:2]
    uu[:, 2] = uh_vals[:, 0]
    uu[:, 3] = uh_vals[:, 1]
    np.savetxt(filename, uu)

    filename = args.output + "/err" + str(args.N) + ".txt"
    err = np.empty((uh_vals.shape[0], 4))
    err[:, 0:2] = pts[:, 0:2]
    xy = [pts[:, 0], pts[:, 1]]
    uxy = u_exact(np)(xy)
    err[:, 2] = abs(uh_vals[:, 0] - uxy[0])
    err[:, 3] = abs(uh_vals[:, 1] - uxy[1])
    np.savetxt(filename, err)
    for d in range(gdim):
        print(f"err {d} in range", min(err[:, d + 2]), max(err[:, d + 2]))

with dolfinx.io.XDMFFile(mesh.comm, args.output + "/displacements.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)


# L2 errors: beware of cancellation
t = dolfinx.common.Timer()
L2_integrand = (uh - u_exact(ufl)) ** 2
L2_err = np.sqrt(
    cq.utils.assemble_cut_uncut(L2_integrand, dx_cut, qr_bulk, dx_uncut, uncut_cell_tag)
)
# L2_err = cq.utils.error_L2(uh, u_exact(ufl), dx_cut, qr_bulk, dx_uncut, uncut_cell_tag)
print("Computing L2 errors took", t.elapsed()[0])

# H10 errors
t = dolfinx.common.Timer()
H10_integrand = (nabla_grad(uh) - nabla_grad(u_exact(ufl))) ** 2
H10_err = np.sqrt(
    cq.utils.assemble_cut_uncut(
        H10_integrand, dx_cut, qr_bulk, dx_uncut, uncut_cell_tag
    )
)
print("Computing H10 errors took", t.elapsed()[0])

# Dump
h = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, cut_cells)
conv = np.array([max(h), L2_err, H10_err])
print(conv)
np.savetxt(args.output + "/conv" + str(args.N) + ".txt", conv.reshape(1, conv.shape[0]))
