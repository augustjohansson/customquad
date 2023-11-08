import argparse
import dolfinx
import customquad
import ufl
from ufl import nabla_grad, inner, dot, jump, avg
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
parser.add_argument("-domain", type=str, default="circle")
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
            breakpoint()


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
degree = 1
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

u_ufl = ufl.as_vector([
    ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    x[0]*0
])
u_expr = lambda x: np.stack((
    np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
    x[0]*0
))
f_ufl = ufl.as_vector([
    2 * ufl.pi * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    0
])

g = u_ufl
f = f_ufl

# PDE
betaN = args.betaN
betas = args.betas
a_bulk = inner(nabla_grad(u), nabla_grad(v))
L_bulk = inner(f, v)
a_bdry = (
    -inner(dot(n, nabla_grad(u)), v) - inner(u, dot(n, nabla_grad(v))) + inner(betaN / h * u, v)
)
L_bdry = -inner(g, dot(n, nabla_grad(v))) + inner(betaN / h * g, v)
a_stab = betas * avg(h) * inner(jump(n, nabla_grad(u)), jump(n, nabla_grad(v)))

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
H10_integrand = (nabla_grad(uh) - nabla_grad(u_ufl)) ** 2
H10_err = np.sqrt(assemble(H10_integrand))

write("output/std_vector_poisson" + str(args.N) + ".xdmf", mesh, uh)
h = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, cut_cells)
print(max(h), L2_err, H10_err)
