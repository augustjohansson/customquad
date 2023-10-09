import argparse
import dolfinx
import customquad
import ufl
from ufl import grad, inner, dot, jump, avg, curl
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import algoim_utils

# Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=16)
parser.add_argument("-algoim", action="store_true")
parser.add_argument("-betaN", type=float, default=20.0)  # NB!
parser.add_argument("-betas", type=float, default=1.0)
parser.add_argument("-domain", type=str, default="circle")
parser.add_argument("-degree", type=int, default=1)
parser.add_argument("-verbose", action="store_true")
args = parser.parse_args()
print("arguments:")
for arg in vars(args):
    print("\t", arg, getattr(args, arg))


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
            raise RuntimeError("Cannot write this datatype")


# Mesh
gdim = 2
NN = np.array([args.N] * gdim, dtype=np.int32)
xmin = np.array([-1.11, -1.51])
xmax = np.array([1.55, 1.22])
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, np.array([xmin, xmax]), NN, dolfinx.mesh.CellType.quadrilateral
)

# QR
algoim_opts = {"verbose": args.verbose}
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
] = algoim_utils.generate_qr(mesh, NN, args.degree, args.domain, algoim_opts)

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

# FEM
V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)
g = dolfinx.fem.Function(V)
f = dolfinx.fem.Function(V)

if gdim == 2:
    # fmt:off
    # u_expr = lambda x: np.stack((
    #     x[0]*x[0],
    #     x[1]*x[1]
    # ))
    # u_ufl = ufl.as_vector([
    #     x[0]*x[0],
    #     x[1]*x[1]
    # ])
    # f_expr = lambda x: np.stack((
    #     -2*np.ones(x.shape[1]),
    #     -2*np.ones(x.shape[1])
    # ))
    # f_ufl = ufl.as_vector([
    #     -2,
    #     -2
    # ])

    u_ufl  = ufl.as_vector([x[0], x[1]])
    u_expr = lambda x: np.stack((x[0], x[1]))
    f_ufl = ufl.as_vector([1e-15, 1e-15])

    # fmt:on

# g.interpolate(u_expr)
# f.interpolate(f_expr)
g = u_ufl
f = f_ufl


# PDE
betaN = args.betaN
betas = args.betas

a_bulk = inner(grad(u), grad(v))
L_bulk = inner(f, v)

# Standard
a_bdry = (
    -inner(dot(n, grad(u)), v) - inner(u, dot(n, grad(v))) + inner(betaN / h * u, v)
)
L_bdry = -inner(g, dot(n, grad(v))) + inner(betaN / h * g, v)

# L_bdry = (
#     -inner(dot(n, grad(g)), v) - inner(g, dot(n, grad(v))) + inner(betaN / h * g, v)
# )

# Flip (a,b) to (b,a)
# a_bdry = (
#     -inner(dot(grad(u), n), v) - inner(u, dot(grad(v), n)) + inner(betaN / h * u, v)
# )
# L_bdry = -inner(g, dot(grad(v), n)) + inner(betaN / h * g, v)

# # Nonsymm should not require stabilization (check Burman paper)
# a_bdry = -inner(dot(grad(u), n), v) + inner(u, dot(grad(v), n))
# L_bdry = inner(g, dot(grad(v), n))

# Stabilization
# TODO: check tensor product
# a_stab = betas * avg(h) * inner(jump(n, grad(u)), jump(n, grad(v)))
a_stab = betas * avg(h) * inner(jump(grad(u), n), jump(grad(v), n))
# a_stab = inner(betas / avg(h) * jump(n, u), jump(n, v))
# a_stab = inner(betas / avg(h) * jump(dot(n, u)), jump(dot(n, v)))
# a_stab = inner(betas / avg(h) * jump(u), jump(v))

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

L2 = dolfinx.fem.form(L_bdry * ds_cut)
bc2 = customquad.assemble_vector(L2, qr_bdry)
b += bc2

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


def ksp_solve(A, b):
    # Direct solver using mumps
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()
    t = dolfinx.common.Timer()
    ksp.solve(b, vec)
    print("Solve took", t.elapsed()[0])
    print("Matrix size", len(b.array))
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return vec


# Solve
vec = ksp_solve(A, b)
uh = dolfinx.fem.Function(V)
uh.vector.setArray(vec.array)
uh.name = "uh"

write("output/vector_poisson" + str(args.N) + ".xdmf", mesh, uh)
assert np.isfinite(vec.array).all()
assert np.isfinite(uh.vector.array).all()


def assemble(integrand):
    m_cut = customquad.assemble_scalar(dolfinx.fem.form(integrand * dx_cut), qr_bulk)
    m_uncut = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(integrand * dx_uncut(uncut_cell_tag))
    )
    return m_cut + m_uncut


# L2 errors: beware of cancellation
L2_integrand = (uh - u_ufl) ** 2
L2_err = np.sqrt(assemble(L2_integrand))

# H10 errors
H10_integrand = (grad(uh) - grad(u_ufl)) ** 2
H10_err = np.sqrt(assemble(H10_integrand))


def project(f, mesh):
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    pf = problem.solve()
    return pf


B = project(curl(uh), mesh)
write("output/B" + str(args.N) + ".xdmf", mesh, B)

# Evaluate solution in qr to see that there aren't any spikes
bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, gdim)
flatten = lambda l: [item for sublist in l for item in sublist]
pts = np.reshape(flatten(xyz), (-1, gdim))
pts_bdry = np.reshape(flatten(xyz_bdry), (-1, gdim))
pts_midpt = dolfinx.mesh.compute_midpoints(mesh, gdim, uncut_cells)
pts = np.append(pts, pts_bdry, axis=0)
pts = np.append(pts, pts_midpt[:, 0:gdim], axis=0)
if gdim == 2:
    # Pad with zero column
    pts = np.hstack((pts, np.zeros((pts.shape[0], 1))))

cell_candidates = dolfinx.cpp.geometry.compute_collisions(bb_tree, pts)
cells = dolfinx.cpp.geometry.compute_colliding_cells(mesh, cell_candidates, pts)
uh_vals = uh.eval(pts, cells.array)
uh_x = uh_vals[:, 0]
uh_y = uh_vals[:, 1]
print("uh_x in range", min(uh_x), max(uh_x))
print("uh_y in range", min(uh_y), max(uh_y))

if gdim == 2:
    # Save coordinates and solution for plotting
    filename = "output/uu" + str(args.N) + ".txt"
    uu = np.empty((len(uh_x), 4))
    uu[:, 0:2] = pts[:, 0:2]
    uu[:, 2] = uh_x
    uu[:, 3] = uh_y
    np.savetxt(filename, uu)

    # Save xy and error for plotting
    err = np.empty((len(uh_x), 4))
    err[:, 0:2] = pts[:, 0:2]
    xy = [pts[:, 0], pts[:, 1]]
    zz = u_expr(xy)
    err[:, 2] = abs(zz[0] - uh_x)
    err[:, 3] = abs(zz[1] - uh_y)
    filename = "output/err" + str(args.N) + ".txt"
    np.savetxt(filename, err)

    uuzz = np.empty((len(uh_x), 4))
    uuzz[:, 0:2] = pts[:, 0:2]
    uuzz[:, 2] = zz[0]
    uuzz[:, 3] = zz[1]
    filename = "output/uuzz" + str(args.N) + ".txt"
    np.savetxt(filename, uuzz)

    max_err_x = max(err[:, 2])
    max_err_y = max(err[:, 3])
    print("max err x", 1.0 / args.N, max_err_x)
    print("max err y", 1.0 / args.N, max_err_y)
    print(1.0 / args.N, max_err_x)

h = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, cut_cells)
conv = np.array(
    [
        max(h),
        L2_err,
        H10_err,
        max_err_x,
        max_err_y,
        args.N,
    ],
)

# breakpoint()
print(conv)
np.savetxt("output/conv" + str(args.N) + ".txt", conv.reshape(1, conv.shape[0]))
