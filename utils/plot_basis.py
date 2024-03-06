import dolfinx
import ufl
import numpy as np
import customquad as cq

gdim = 2
N = [1] * gdim
bbmin = np.array([0.0, 0.0, 0.0])[0:gdim]
bbmax = np.array([1.0, 1.0, 1.0])[0:gdim]
polynomial_order = 2
debug = False
mesh = cq.create_mesh(np.array([bbmin, bbmax]), np.array(N), polynomial_order, debug)
tdim = gdim

xmin = np.min(mesh.geometry.x, axis=0)[0:tdim]
xmax = np.max(mesh.geometry.x, axis=0)[0:tdim]
xdiff = xmax - xmin

num_cells = cq.utils.get_num_cells(mesh)
cells = np.arange(num_cells)

V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", polynomial_order))
v = ufl.TestFunction(V)
integrand = 1 * v
dx = ufl.dx(metadata={"quadrature_rule": "runtime"})

x0 = 0
y0 = 0
Nq = 100
qr_pts = np.empty((1, 2))
qr_w = np.array([[1.0]])

num_basis = (polynomial_order + 1) ** 2

z = np.ones((num_basis, Nq * Nq, 3))

for iy in range(Nq):
    for ix in range(Nq):
        qr_pts[0][0] = x = float(ix) / Nq
        qr_pts[0][1] = y = float(iy) / Nq

        b = cq.assemble_vector(
            dolfinx.fem.form(integrand * dx), [(cells, qr_pts, qr_w)]
        )

        idx = iy * Nq + ix

        for k in range(num_basis):
            z[k, idx, 0] = x
            z[k, idx, 1] = y
            z[k, idx, 2] = b.array[k]

for k in range(num_basis):
    np.savetxt(f"basis{k}.txt", z[k])
