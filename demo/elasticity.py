import os
from contextlib import ExitStack
import argparse
import dolfinx
import customquad as cq
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
parser.add_argument("-p", type=int, default=1)
parser.add_argument("-order", type=int, default=1)
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
            raise RuntimeError("Unsupported data when writing file", filename)


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
# if gdim == 2:
#     cell_type = dolfinx.mesh.CellType.quadrilateral
#     mesh_generator = dolfinx.mesh.create_rectangle
# else:
#     cell_type = dolfinx.mesh.CellType.hexahedron
#     mesh_generator = dolfinx.mesh.create_box
# print(f"{NN=}")
# print(f"{xmin=}")
# print(f"{xmax=}")
# mesh = mesh_generator(MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type)
# assert mesh.geometry.dim == gdim

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
os.makedirs("output", exist_ok=True)
with dolfinx.io.XDMFFile(mesh.comm, f"output/msh{args.N}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(celltags)
    xdmf.write_meshtags(facetags)

# Data
if gdim == 2:
    f = ufl.as_vector([1.0, 0.0])
elif gdim == 3:
    f = ufl.as_vector([0.0, 0.0, 1.0])
else:
    raise RuntimeError("Unknown dim")

E = 1
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(v):
    return ufl.sym(grad(v))


def sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v))


# FEM
V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", args.p))
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

# L2 = dolfinx.fem.form(L_bdry * ds_cut)
# bc2 = cq.assemble_vector(L2, qr_bdry)
# b += bc2

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
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()
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


def flatten(lst):
    return [item for sublist in lst for item in sublist]


# Evaluate solution in qr to see that there aren't any spikes
pts = np.reshape(flatten(xyz), (-1, gdim))
pts_bdry = np.reshape(flatten(xyz_bdry), (-1, gdim))
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
    filename = "output/uu" + str(args.N) + ".txt"
    uu = np.empty((uh_vals.shape[0], 4))
    uu[:, 0:2] = pts[:, 0:2]
    uu[:, 2] = uh_vals[:, 0]
    uu[:, 3] = uh_vals[:, 1]
    np.savetxt(filename, uu)


os.makedirs("output", exist_ok=True)
with dolfinx.io.XDMFFile(mesh.comm, "output/displacements.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)
