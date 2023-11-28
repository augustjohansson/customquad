import dolfinx
import ufl
import basix
import numpy as np
from mpi4py import MPI
import gmsh

flatten = lambda l: [item for sublist in l for item in sublist]


def create_mesh(xrange, N, degree, debug=False):
    # 1. Create gmsh grid
    gmsh.initialize()
    factory = gmsh.model.occ
    model_name = "create_high_order_mesh"
    gmsh.model.add(model_name)
    factory = gmsh.model.occ

    xmin = xrange[0]
    xmax = xrange[1]
    L = xmax - xmin
    gdim = len(xmin)
    assert len(N) == gdim

    if gdim == 2:
        obj = factory.addRectangle(xmin[0], xmin[1], 0, L[0], L[1])
        dx_cell_type = dolfinx.mesh.CellType.quadrilateral
        gmsh_cell_type = "quadrangle"
    elif gdim == 3:
        obj = factory.addBox(xmin[0], xmin[1], xmin[2], L[0], L[1], L[2])
        dx_cell_type = dolfinx.mesh.CellType.hexahedron
        gmsh_cell_type = "hexahedron"
    else:
        RuntimeError("Unknown dimension")

    factory.synchronize()

    # Lines
    for c in gmsh.model.getEntities(1):
        bbox = gmsh.model.getBoundingBox(c[0], c[1])
        dx = np.array(bbox[3:]) - np.array(bbox[:3])
        dim = np.argmax(dx)
        assert dim < gdim
        gmsh.model.mesh.setTransfiniteCurve(c[1], N[dim] + 1)

    # Surfaces
    for s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s[1])
        gmsh.model.mesh.setRecombine(s[0], s[1])

    if gdim == 3:
        # Volumes
        gmsh.model.mesh.setTransfiniteVolume(obj)
        gmsh.model.mesh.setRecombine(3, obj)

    gmsh.model.mesh.generate(gdim)

    if debug:
        Nstr = str(N).replace(" ", "_").replace("[", "").replace("]", "")
        gmsh.write("output/" + model_name + "_" + Nstr + ".msh")
        gmsh.write("output/" + model_name + "_" + Nstr + ".mesh")

    gmsh.model.mesh.setOrder(degree)

    # 2. Convert to dolfinx (from test_quadrilateral_mesh from
    # test_higher_order_mesh.py in dolfinx)
    idx, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    assert coords.shape[0] == degree**gdim * np.prod(N + 1)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = coords[srt, :gdim]
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=gdim)
    (
        name,
        dim,
        degree,
        num_nodes,
        local_coords,
        num_first_order_nodes,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    cells = cells[:, dolfinx.io.gmshio.cell_perm_array(dx_cell_type, cells.shape[1])]

    gmsh_cell_id = gmsh.model.mesh.getElementType(gmsh_cell_type, degree)
    domain = dolfinx.io.gmshio.ufl_mesh(gmsh_cell_id, x.shape[1])

    # if MPIpx.COMM_WORLD.rank == 0:
    #     tri_points = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    #     triangles = np.array([[0,1,3], [0,2,3]], dtype=np.int64)
    # else:
    #     tri_points = np.empty((0,2), dtype=np.float64)
    #     triangles = np.empty((0,3), dtype=np.int64)
    # ufl_tri = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.triangle, 1))
    # tri_mesh = dolfinx.mesh.create_mesh(
    #     MPIpx.COMM_WORLD, triangles, tri_points, ufl_tri)
    # cell_index_map = tri_mesh.topology.index_map(tri_mesh.topology.dim)
    # print(f"Num cells local: {cell_index_map.size_local}\n Num cells global: {cell_index_map.size_global}")

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

    assert mesh.topology.index_map(0).size_local == np.prod(N + 1)
    assert mesh.topology.index_map(gdim).size_local == np.prod(N)
    assert mesh.geometry.x.shape[0] == degree**gdim * np.prod(N + 1)

    gmsh.finalize()

    return mesh


# def create_high_order_quad_mesh(xrange, N, degree, debug=False):
#     gdim = 2
#     gmsh.initialize()
#     model_name = "create_high_order_quad_mesh"
#     gmsh.model.add(model_name)
#     factory = gmsh.model.occ

#     xmin = xrange[0]
#     xmax = xrange[1]
#     dx = xmax - xmin

#     rect = factory.addRectangle(xmin[0], xmin[1], 0, dx[0], dx[1])
#     factory.synchronize()

#     p0 = factory.addPoint(xmin[0], xmin[1], 0)
#     p1 = factory.addPoint(xmax[0], xmin[1], 0)
#     p2 = factory.addPoint(xmin[0], xmax[1], 0)
#     p3 = factory.addPoint(xmax[0], xmax[1], 0)
#     l0 = factory.addLine(p0, p1)
#     l1 = factory.addLine(p0, p2)
#     l2 = factory.addLine(p1, p3)
#     l3 = factory.addLine(p2, p3)
#     cl = factory.addCurveLoop([l0, l2, -l3, -l1])
#     surf = factory.addPlaneSurface([cl])
#     factory.synchronize()

#     nx = N[0] + 1
#     ny = N[1] + 1
#     gmsh.model.mesh.setTransfiniteCurve(l0, nx)
#     gmsh.model.mesh.setTransfiniteCurve(l3, nx)
#     gmsh.model.mesh.setTransfiniteCurve(l1, ny)
#     gmsh.model.mesh.setTransfiniteCurve(l2, ny)
#     gmsh.model.mesh.setTransfiniteSurface(surf)

#     gmsh.model.mesh.setRecombine(2, surf)
#     factory.synchronize()

#     # Create mesh
#     mesh = gmsh_to_dolfinx(gmsh, gdim, N, degree, debug)

#     return mesh


# def create_high_order_hex_mesh(xrange, N, degree, debug=False):
#     gdim = 3
#     gmsh.initialize()
#     model_name = "create_high_order_hex_mesh"
#     gmsh.model.add(model_name)
#     factory = gmsh.model.geo

#     xmin = xrange[0]
#     xmax = xrange[1]

#     p0 = factory.addPoint(xmin[0], xmin[1], xmin[2])
#     p1 = factory.addPoint(xmax[0], xmin[1], xmin[2])
#     p2 = factory.addPoint(xmin[0], xmax[1], xmin[2])
#     p3 = factory.addPoint(xmax[0], xmax[1], xmin[2])
#     p4 = factory.addPoint(xmin[0], xmin[1], xmax[2])
#     p5 = factory.addPoint(xmax[0], xmin[1], xmax[2])
#     p6 = factory.addPoint(xmin[0], xmax[1], xmax[2])
#     p7 = factory.addPoint(xmax[0], xmax[1], xmax[2])

#     l0 = factory.addLine(p0, p1)
#     l1 = factory.addLine(p0, p2)
#     l2 = factory.addLine(p1, p3)
#     l3 = factory.addLine(p2, p3)

#     l4 = factory.addLine(p4, p5)
#     l5 = factory.addLine(p4, p6)
#     l6 = factory.addLine(p5, p7)
#     l7 = factory.addLine(p6, p7)

#     l8 = factory.addLine(p0, p4)
#     l9 = factory.addLine(p1, p5)
#     l10 = factory.addLine(p2, p6)
#     l11 = factory.addLine(p3, p7)

#     cl0 = factory.addCurveLoop([l0, l2, -l3, -l1])  # interior normal
#     cl1 = factory.addCurveLoop([l5, l7, -l6, -l4])
#     cl2 = factory.addCurveLoop([-l0, l8, l4, -l9])
#     cl3 = factory.addCurveLoop([l3, l11, -l7, -l10])
#     cl4 = factory.addCurveLoop([l1, l10, -l5, -l8])
#     cl5 = factory.addCurveLoop([l9, l6, -l11, -l2])  # l2, l11, -l6, -l9])

#     s0 = factory.addPlaneSurface([cl0])
#     s1 = factory.addPlaneSurface([cl1])
#     s2 = factory.addPlaneSurface([cl2])
#     s3 = factory.addPlaneSurface([cl3])
#     s4 = factory.addPlaneSurface([cl4])
#     s5 = factory.addPlaneSurface([cl5])
#     surfs = [s0, s1, s2, s3, s4, s5]

#     sl = factory.addSurfaceLoop(surfs)

#     vol = factory.addVolume([sl])

#     nx = N[0] + 1
#     ny = N[1] + 1
#     nz = N[2] + 1
#     gmsh.model.mesh.setTransfiniteCurve(l0, nx)
#     gmsh.model.mesh.setTransfiniteCurve(l3, nx)
#     gmsh.model.mesh.setTransfiniteCurve(l4, nx)
#     gmsh.model.mesh.setTransfiniteCurve(l7, nx)
#     gmsh.model.mesh.setTransfiniteCurve(l1, ny)
#     gmsh.model.mesh.setTransfiniteCurve(l2, ny)
#     gmsh.model.mesh.setTransfiniteCurve(l5, ny)
#     gmsh.model.mesh.setTransfiniteCurve(l6, ny)
#     gmsh.model.mesh.setTransfiniteCurve(l7, nz)
#     gmsh.model.mesh.setTransfiniteCurve(l9, nz)
#     gmsh.model.mesh.setTransfiniteCurve(l10, nz)
#     gmsh.model.mesh.setTransfiniteCurve(l11, nz)

#     for s in surfs:
#         gmsh.model.mesh.setTransfiniteSurface(s)

#     gmsh.model.mesh.setTransfiniteVolume(vol)

#     for s in surfs:
#         gmsh.model.mesh.setRecombine(2, s)

#     gmsh.model.mesh.setRecombine(3, vol)

#     factory.synchronize()

#     # Create mesh
#     mesh = gmsh_to_dolfinx(gmsh, gdim, N, degree, debug)

#     return mesh


# def gmsh_to_dolfinx(gmsh, gdim, N, degree, debug=False):
#     gmsh.model.mesh.generate(gdim)
#     if debug:
#         model_name = gmsh.model.getCurrent()
#         Nstr = str(N).replace(" ", "_").replace("[", "").replace("]", "")
#         gmsh.write(model_name + "_" + Nstr + ".mesh")
#     gmsh.model.mesh.setOrder(degree)

#     # Mesh conversion
#     idx, coords, _ = gmsh.model.mesh.getNodes()
#     coords = coords.reshape(-1, 3)
#     # assert coords.shape[0] == (degree * nx) * (degree * ny)
#     assert coords.shape[0] == degree**gdim * np.prod(N + 1)
#     idx -= 1
#     srt = np.argsort(idx)
#     assert np.all(idx[srt] == np.arange(len(idx)))
#     x = coords[srt, :gdim]
#     element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=gdim)
#     (
#         name,
#         dim,
#         degree,
#         num_nodes,
#         local_coords,
#         num_first_order_nodes,
#     ) = gmsh.model.mesh.getElementProperties(element_types[0])

#     if gdim == 2:
#         dx_cell_type = dolfinx.mesh.CellType.quadrilateral
#         gmsh_cell_type = "quadrangle"
#     elif gdim == 3:
#         dx_cell_type = dolfinx.mesh.CellType.hexahedron
#         gmsh_cell_type = "hexahedron"
#     else:
#         RuntimeError("Unknown gdim")

#     cells = node_tags[0].reshape(-1, num_nodes) - 1
#     cells = cells[:, dolfinx.io.gmshio.cell_perm_array(dx_cell_type, cells.shape[1])]

#     gmsh_cell_id = gmsh.model.mesh.getElementType(gmsh_cell_type, degree)
#     mesh = dolfinx.mesh.create_mesh(
#         MPI.COMM_WORLD, cells, x, dolfinx.io.gmshio.ufl_mesh(gmsh_cell_id, x.shape[1])
#     )

#     assert mesh.topology.index_map(0).size_local == np.prod(N + 1)
#     assert mesh.geometry.x.shape[0] == degree**gdim * np.prod(N + 1)
#     assert mesh.topology.index_map(gdim).size_local == np.prod(N)

#     gmsh.finalize()

#     return mesh
