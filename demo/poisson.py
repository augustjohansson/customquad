import dolfinx
import customquad as cq
import ufl
from ufl import grad, inner, dot, jump, avg
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import argparse
import algoim_utils
import os


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
parser.add_argument("-gamma", type=float, default=0.5)
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print("\t", arg, getattr(args, arg))


# tag = (
#     "domain" + args.domain + "_"
#     "p" + str(args.p) + "_"
#     "order" + str(args.order) + "_"
#     "betaN" + str(args.betaN) + "_"
#     "betas" + str(args.betas) + "_"
# )
# outputdir = "output_" + tag
outputdir = "output"
os.makedirs(outputdir, exist_ok=True)


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
            raise RuntimeError("Trying to write unsupported data")


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


def u_exact(backend):
    if gdim == 2:
        return lambda x: backend.sin(backend.pi * x[0]) * backend.sin(backend.pi * x[1])
    else:
        return (
            lambda x: backend.sin(backend.pi * x[0])
            * backend.sin(backend.pi * x[1])
            * backend.sin(backend.pi * x[2])
        )


# Mesh
NN = np.array([args.N] * gdim, dtype=np.int32)
# if args.p == 1:
#     if gdim == 2:
#         cell_type = dolfinx.mesh.CellType.quadrilateral
#         mesh_generator = dolfinx.mesh.create_rectangle
#     else:
#         cell_type = dolfinx.mesh.CellType.hexahedron
#         mesh_generator = dolfinx.mesh.create_box
#     mesh = mesh_generator(MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type)
#     assert mesh.geometry.dim == gdim

# else:
#     if gdim == 2:
#         mesh = cq.create_high_order_quad_mesh(np.array([xmin, xmax]), NN, args.p)
#     else:
#         mesh = cq.create_high_order_hex_mesh(np.array([xmin, xmax]), NN, args.p)
#     assert mesh.geometry.dim == gdim

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
with dolfinx.io.XDMFFile(mesh.comm, f"output/msh{args.N}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(celltags)
    xdmf.write_meshtags(facetags)

# Check functional assembly
ds_cut = ufl.dx(
    subdomain_data=celltags, metadata={"quadrature_rule": "runtime"}, domain=mesh
)
dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)


def assemble(integrand):
    m_cut = cq.assemble_scalar(dolfinx.fem.form(integrand * dx_cut), qr_bulk)
    m_uncut = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(integrand * dx_uncut(uncut_cell_tag))
    )
    return m_cut + m_uncut


qr_bulk = [(cut_cells, qr_pts, qr_w)]
qr_bdry = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]
area_func = cq.assemble_scalar(dolfinx.fem.form(1.0 * ds_cut(cut_cell_tag)), qr_bdry)
volume_func = assemble(1.0)
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
g.interpolate(u_exact(np))
f = -ufl.div(ufl.grad(u_exact(ufl)(x)))
# g.interpolate(lambda x: 0.0 + 1e-14 * x[0])
# f.interpolate(lambda x: 1.0 + 1e-14 * x[0])

# PDE
betaN = args.betaN * args.p**2
betas = args.betas
a_bulk = inner(grad(u), grad(v))
L_bulk = inner(f, v)
a_bdry = (
    -inner(dot(n, grad(u)), v) - inner(u, dot(n, grad(v))) + inner(betaN / h * u, v)
)
L_bdry = -inner(g, dot(n, grad(v))) + inner(betaN / h * g, v)
a_stab = betas * avg(h) ** (2 * args.gamma) * inner(jump(n, grad(u)), jump(n, grad(v)))
if args.p == 2:
    a_stab += (
        betas
        * avg(h) ** (2 * args.gamma)
        * inner(jump(n, grad(grad(u))), jump(n, grad(grad(v))))
    )

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
    cq.utils.dump("output/A.txt", A)
    cq.utils.dump("output/b.txt", b)
    cq.utils.dump("output/bx.txt", bx)
    cq.utils.dump("output/bc1.txt", bc1)
    cq.utils.dump("output/bc2.txt", bc2)

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
    cq.utils.dump("output/A_locked.txt", A)
assert np.isfinite(A.norm()).all()


def mumps(A, b):
    # Direct solver using mumps
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()
    ksp.solve(b, vec)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return vec


def cg(A, b):
    # Iterative solver using cg/gamg
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-10
    opts["pc_type"] = "gamg"
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setFromOptions()
    ksp.setOperators(A)
    vec = b.copy()
    # ksp.setMonitor(
    #     lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
    # )
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
    cg.utils.dump("output/vec.txt", vec)

uh = dolfinx.fem.Function(V)
uh.vector.setArray(vec.array)
uh.name = "uh"
write(outputdir + "/poisson" + str(args.N) + ".xdmf", mesh, uh)
assert np.isfinite(vec.array).all()
assert np.isfinite(uh.vector.array).all()


# L2 errors: beware of cancellation
t = dolfinx.common.Timer()
L2_integrand = (uh - u_exact(ufl)(x)) ** 2
L2_err = np.sqrt(assemble(L2_integrand))
print("Computing L2 errors took", t.elapsed()[0])

# H10 errors
t = dolfinx.common.Timer()
H10_integrand = (grad(uh) - grad(u_exact(ufl)(x))) ** 2
H10_err = np.sqrt(assemble(H10_integrand))
print("Computing H10 errors took", t.elapsed()[0])


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
uh_vals = uh.eval(pts, cells.array).flatten()
print("uh in range", uh_vals.min(), uh_vals.max())

if gdim == 2:
    # Save coordinates and solution for plotting
    filename = outputdir + "/uu" + str(args.N) + ".txt"
    uu = pts
    uu[:, 2] = uh_vals
    np.savetxt(filename, uu)

    # Save xy and error for plotting
    err = pts
    xy = [pts[:, 0], pts[:, 1]]
    err[:, 2] = abs(u_exact(np)(xy) - uh_vals)
    filename = outputdir + "/err" + str(args.N) + ".txt"
    np.savetxt(filename, err)

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

np.savetxt(outputdir + "/conv" + str(args.N) + ".txt", conv.reshape(1, conv.shape[0]))
