import argparse
import os
import numpy as np
import dolfinx
import ufl
from ufl import grad, inner, dot, jump, avg
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
parser.add_argument("-order", type=int, default=1)
parser.add_argument("-verbose", action="store_true")
parser.add_argument("-solver", type=str, default="mumps")
# parser.add_argument("-gamma", type=float, default=0.5)
parser.add_argument("-output", type=str, default="output")
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print("\t", arg, getattr(args, arg))

os.makedirs(args.output, exist_ok=True)

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


def u_exact(m):
    if gdim == 2:
        return lambda x: m.sin(m.pi * x[0]) * m.sin(m.pi * x[1])
    return lambda x: m.sin(m.pi * x[0]) * m.sin(m.pi * x[1]) * m.sin(m.pi * x[2])


# Mesh
NN = np.array([args.N] * gdim, dtype=np.int32)
t = dolfinx.common.Timer()
mesh = cq.create_mesh(np.array([xmin, xmax]), NN, args.p, args.verbose)
print("Generating mesh took", t.elapsed()[0])

if args.verbose:
    print(f"{NN=}")
    print(f"{xmin=}")
    print(f"{xmax=}")

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
t = dolfinx.common.Timer()
celltags = cq.utils.get_celltags(
    mesh,
    cut_cells,
    uncut_cells,
    outside_cells,
    uncut_cell_tag=uncut_cell_tag,
    cut_cell_tag=cut_cell_tag,
    outside_cell_tag=outside_cell_tag,
)
print("Generating cell tags took", t.elapsed()[0])
t = dolfinx.common.Timer()
facetags = cq.utils.get_facetags(
    mesh, cut_cells, outside_cells, ghost_penalty_tag=ghost_penalty_tag
)
print("Generating face tags took", t.elapsed()[0])

# Write mesh with tags
with dolfinx.io.XDMFFile(mesh.comm, args.output + f"/msh{args.N}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(celltags)
    xdmf.write_meshtags(facetags)

# Check functional assembly
ds_cut = ufl.dx(
    subdomain_data=celltags, metadata={"quadrature_rule": "runtime"}, domain=mesh
)
dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
qr_bulk = [(cut_cells, qr_pts, qr_w)]
qr_bdry = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]
area_func = cq.assemble_scalar(dolfinx.fem.form(1.0 * ds_cut(cut_cell_tag)), qr_bdry)
volume_func = cq.utils.assemble_cut_uncut(
    1.0, dx_cut, qr_bulk, dx_uncut, uncut_cell_tag
)
ve = abs(volume_exact - volume_func) / volume_exact
ae = abs(area_exact - area_func) / area_exact
print("functional volume error", ve)
print("functional area error", ae)

# Geometry errors
volume = cq.utils.volume(xmin, xmax, NN, uncut_cells, qr_w)
area = cq.utils.area(xmin, xmax, NN, qr_w_bdry)
volume_err = abs(volume_exact - volume) / volume_exact
area_err = abs(area_exact - area) / area_exact
print("qr volume error", volume_err)
print("qr area error", area_err)

# FEM
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", args.p))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
g = dolfinx.fem.Function(V)
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
if args.p == 1:
    h = ufl.CellDiameter(mesh)
else:
    h = max((xmax - xmin) / args.N)

# Setup boundary traction and rhs
g = u_exact(ufl)(x)
f = -ufl.div(ufl.grad(u_exact(ufl)(x)))

# PDE
betaN = args.betaN * args.p**2
betas = args.betas
a_bulk = inner(grad(u), grad(v))
L_bulk = inner(f, v)
a_bdry = (
    -inner(dot(n, grad(u)), v) - inner(u, dot(n, grad(v))) + inner(betaN / h * u, v)
)
L_bdry = -inner(g, dot(n, grad(v))) + inner(betaN / h * g, v)

# Stabilization
a_stab = betas * avg(h) * inner(jump(n, grad(u)), jump(n, grad(v)))
if args.p == 2:
    a_stab += (
        betas * avg(h) ** 3 * inner(jump(n, grad(grad(u))), jump(n, grad(grad(v))))
    )
elif args.p > 2:
    raise RuntimeError("No stab yet for elements higher than quadratic")

# Standard measures
dS = ufl.dS(subdomain_data=facetags, domain=mesh)

# Integration using standard assembler (uncut cells, ghost penalty
# faces)
ax = dolfinx.fem.form(
    a_bulk * dx_uncut(uncut_cell_tag) + a_stab * dS(ghost_penalty_tag)
)
t = dolfinx.common.Timer()
A = dolfinx.fem.petsc.assemble_matrix(ax)
A.assemble()
print("Assemble interior took", t.elapsed()[0])
Lx = dolfinx.fem.form(L_bulk * dx_uncut(uncut_cell_tag))
bx = dolfinx.fem.petsc.assemble_vector(Lx)

# Integration using custom assembler (i.e. integrals over cut cells,
# both cut bulk part and bdry part)
form_cut_bulk = dolfinx.fem.form(a_bulk * dx_cut)
form_cut_bdry = dolfinx.fem.form(a_bdry * ds_cut(cut_cell_tag))

t = dolfinx.common.Timer()
Ac1 = cq.assemble_matrix(form_cut_bulk, qr_bulk)
Ac1.assemble()
print("Runtime assemble bulk took", t.elapsed()[0])

t = dolfinx.common.Timer()
Ac2 = cq.assemble_matrix(form_cut_bdry, qr_bdry)
Ac2.assemble()
print("Runtime assemble bdry took", t.elapsed()[0])

t = dolfinx.common.Timer()
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

if args.verbose:
    cq.debug_utils.dump(args.output + "/A.txt", A)
    cq.debug_utils.dump(args.output + "/b.txt", b)
    cq.debug_utils.dump(args.output + "/bx.txt", bx)
    cq.debug_utils.dump(args.output + "/bc1.txt", bc1)
    cq.debug_utils.dump(args.output + "/bc2.txt", bc2)

assert np.isfinite(b.array).all()
assert np.isfinite(A.norm())

# Lock inactive dofs
t = dolfinx.common.Timer()
inactive_dofs = cq.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
print("Get inactive_dofs took", t.elapsed()[0])
t = dolfinx.common.Timer()
A = cq.utils.lock_inactive_dofs(inactive_dofs, A)
print("Lock inactive dofs took", t.elapsed()[0])
if args.verbose:
    cq.debug_utils.dump(args.output + "/A_locked.txt", A)
assert np.isfinite(A.norm()).all()


def mumps(A, b):
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
    return vec


def cg(A, b):
    # Iterative solver using eg cg/gamg (or jacobi to save memory)
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-4
    opts["ksp_max_it"] = A.size[0]
    # opts["pc_type"] = "gamg"
    opts["pc_type"] = "jacobi"
    print("Start solve using", opts["ksp_type"], opts["pc_type"])
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setFromOptions()
    ksp.setOperators(A)
    vec = A.createVecRight()
    ksp.setMonitor(
        lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
    )
    ksp.solve(b, vec)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return vec


# Solve
t = dolfinx.common.Timer()
if args.solver == "mumps":
    vec = mumps(A, b)
elif args.solver == "cg":
    vec = cg(A, b)
else:
    raise RuntimeError("Unknown solver", args.solver)
print(f"Solver {args.solver} solve took", t.elapsed()[0])
print("Matrix size", len(vec.array))

if args.verbose:
    cg.debug_utils.dump(args.output + "/vec.txt", vec)

uh = dolfinx.fem.Function(V)
uh.vector.setArray(vec.array)
uh.name = "uh"
cq.utils.writeXDMF(args.output + "/poisson" + str(args.N) + ".xdmf", mesh, uh)
assert np.isfinite(uh.vector.array).all()

# L2 errors: beware of cancellation
t = dolfinx.common.Timer()
L2_integrand = (uh - u_exact(ufl)(x)) ** 2
L2_err = np.sqrt(
    cq.utils.assemble_cut_uncut(L2_integrand, dx_cut, qr_bulk, dx_uncut, uncut_cell_tag)
)
print("Computing L2 errors took", t.elapsed()[0])

# H10 errors
t = dolfinx.common.Timer()
H10_integrand = (grad(uh) - grad(u_exact(ufl)(x))) ** 2
H10_err = np.sqrt(
    cq.utils.assemble_cut_uncut(
        H10_integrand, dx_cut, qr_bulk, dx_uncut, uncut_cell_tag
    )
)
print("Computing H10 errors took", t.elapsed()[0])


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
uh_vals = uh.eval(pts, cells.array).flatten()
print("uh in range", uh_vals.min(), uh_vals.max())

if gdim == 2:
    # Save coordinates and solution for plotting
    filename = args.output + "/uu" + str(args.N) + ".txt"
    uu = pts
    uu[:, 2] = uh_vals
    np.savetxt(filename, uu)

    # Save xy and error for plotting
    err = pts
    xy = [pts[:, 0], pts[:, 1]]
    err[:, 2] = abs(u_exact(np)(xy) - uh_vals)
    filename = args.output + "/err" + str(args.N) + ".txt"
    np.savetxt(filename, err)

    if args.verbose:
        filename = args.output + "/xyz" + str(args.N) + ".txt"
        np.savetxt(filename, np.reshape(cq.utils.flatten(xyz), (-1, gdim)))
        filename = args.output + "/xyz_bdry" + str(args.N) + ".txt"
        np.savetxt(filename, np.reshape(cq.utils.flatten(xyz_bdry), (-1, gdim)))

# Print
h = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, cut_cells)
conv = np.array(
    [
        max(h),
        L2_err,
        H10_err,
        volume,
        volume_err,
        area,
        area_err,
        args.N,
    ],
)

print(conv)

np.savetxt(args.output + "/conv" + str(args.N) + ".txt", conv.reshape(1, conv.shape[0]))
