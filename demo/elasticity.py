from contextlib import ExitStack
import argparse
import dolfinx
import customquad
import ufl
from ufl import grad, inner, dot, jump, avg
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import algoim_utils

# Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=16)
parser.add_argument("-algoim", action="store_true")
parser.add_argument("-betaN", type=float, default=10.0)
parser.add_argument("-betas", type=float, default=1.0)
parser.add_argument("-domain", type=str, default="sphere")
parser.add_argument("-verbose", action="store_true")
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print("\t", arg, getattr(args, arg))


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


def write(filename, mesh, data):
    with dolfinx.io.XDMFFile(
        mesh.comm,
        filename,
        "w",
    ) as xdmffile:
        xdmffile.write_mesh(mesh)
        if isinstance(data, dolfinx.mesh.MeshTagsMetaClass):
            xdmffile.write_meshtags(data)
        elif isinstance(data, dolfinx.fem.Function):
            xdmffile.write_function(data)
        else:
            breakpoint()


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
    RuntimeError("Unknown domain", args.domain)

gdim = len(xmin)


# Mesh
NN = np.array([args.N] * gdim, dtype=np.int32)
if gdim == 2:
    cell_type = dolfinx.mesh.CellType.quadrilateral
    mesh_generator = dolfinx.mesh.create_rectangle
else:
    cell_type = dolfinx.mesh.CellType.hexahedron
    mesh_generator = dolfinx.mesh.create_box
print(f"{NN=}")
print(f"{xmin=}")
print(f"{xmax=}")
mesh = mesh_generator(MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type)
assert mesh.geometry.dim == gdim

# Generate qr
degree = 1
algoim_opts = {"verbose": args.verbose}
t = dolfinx.common.Timer()
[
    cut_cells,
    uncut_cells,
    outside_cells,
    qr_pts0,
    qr_w0,
    qr_pts_bdry0,
    qr_w_bdry0,
    qr_n0,
    xyz,
    xyz_bdry,
] = algoim_utils.generate_qr(mesh, NN, degree, args.domain, algoim_opts)

print("Generating qr took", t.elapsed()[0])
print("num cells", customquad.utils.get_num_cells(mesh))
print("num cut_cells", len(cut_cells))
print("num uncut_cells", len(uncut_cells))
print("num outside_cells", len(outside_cells))

# Algoim creates (at the moment) quadrature rules for _all_ cells, not
# only the cut cells. Remove these empty entries
qr_pts = [qr_pts0[k] for k in cut_cells]
qr_w = [qr_w0[k] for k in cut_cells]
qr_pts_bdry = [qr_pts_bdry0[k] for k in cut_cells]
qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]
qr_n = [qr_n0[k] for k in cut_cells]

# Set up cell tags and face tags
uncut_cell_tag = 1
cut_cell_tag = 2
outside_cell_tag = 3
ghost_penalty_tag = 4
celltags = customquad.utils.get_celltags(
    mesh,
    cut_cells,
    uncut_cells,
    outside_cells,
    uncut_cell_tag=uncut_cell_tag,
    cut_cell_tag=cut_cell_tag,
    outside_cell_tag=outside_cell_tag,
)
facetags = customquad.utils.get_facetags(
    mesh, cut_cells, outside_cells, ghost_penalty_tag=ghost_penalty_tag
)

# Write cell tags
write("output/celltags" + str(args.N) + ".xdmf", mesh, celltags)

# Data
if gdim == 2:
    f = ufl.as_vector([1.0, 0.0])
elif gdim == 3:
    f = ufl.as_vector([0.0, 0.0, 1.0])
else:
    raise RuntimeError("Unknown dim")

E = 1e9
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(v):
    return ufl.sym(grad(v))  # ~1


def sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v))


# FEM
V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
g = dolfinx.fem.Function(V)
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)


# PDE
betaN = args.betaN
betas = args.betas
a_bulk = inner(sigma(u), grad(v))
L_bulk = inner(f, v)
a_bdry = (
    -inner(dot(n, sigma(u)), v)
    - inner(u, dot(n, sigma(v)))
    + inner(betaN / h * (2 * mu + lmbda) * u, v)
)
a_stab = betas * avg(h) * inner(jump(sigma(u)), jump(sigma(v)))

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
ds_cut = ufl.dx(
    subdomain_data=celltags, metadata={"quadrature_rule": "runtime"}, domain=mesh
)

qr_bulk = [(cut_cells, qr_pts, qr_w)]
qr_bdry = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]

# FIXME make sure we can assemble over many forms
form1 = dolfinx.fem.form(a_bulk * dx_cut)
form2 = dolfinx.fem.form(a_bdry * ds_cut(cut_cell_tag))

t = dolfinx.common.Timer()
Ac1 = customquad.assemble_matrix(form1, qr_bulk)
Ac1.assemble()
print("Runtime assemble bulk took", t.elapsed()[0])

t = dolfinx.common.Timer()
Ac2 = customquad.assemble_matrix(form2, qr_bdry)
Ac2.assemble()
print("Runtime assemble bdry took", t.elapsed()[0])

t = dolfinx.common.Timer()
A = Ax
A += Ac1
A += Ac2
print("Matrix += took", t.elapsed()[0])

L1 = dolfinx.fem.form(L_bulk * dx_cut)
bc1 = customquad.assemble_vector(L1, qr_bulk)
b = bx
b += bc1

# L2 = dolfinx.fem.form(L_bdry * ds_cut)
# bc2 = customquad.assemble_vector(L2, qr_bdry)
# b += bc2

assert np.isfinite(b.array).all()
assert np.isfinite(A.norm())

# Lock inactive dofs
t = dolfinx.common.Timer()
inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
print("Get inactive_dofs took", t.elapsed()[0])
t = dolfinx.common.Timer()
A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)
print("Lock inactive dofs took", t.elapsed()[0])
if args.verbose:
    customquad.utils.dump("output/A_locked.txt", A)
assert np.isfinite(A.norm()).all()

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

uh = dolfinx.fem.Function(V)

# Set a monitor, solve linear system, and display the solver
# configuration
solver.setMonitor(
    lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
)
solver.solve(b, uh.vector)
solver.view()

# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()

with dolfinx.io.XDMFFile(mesh.comm, "output/displacements.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)
