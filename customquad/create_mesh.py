import dolfinx
import numpy as np
from mpi4py import MPI
import gmsh


def create_mesh(xrange, N, degree, debug=False):
    # 1. Create gmsh grid
    gmsh.initialize()
    if not debug:
        gmsh.option.setNumber("General.Terminal", 0)
    factory = gmsh.model.occ
    model_name = "cq_create_mesh"
    gmsh.model.add(model_name)
    factory = gmsh.model.occ

    xmin = xrange[0]
    xmax = xrange[1]
    L = xmax - xmin
    gdim = len(xmin)
    assert len(N) == gdim

    if gdim == 2:
        obj = factory.addRectangle(xmin[0], xmin[1], 0, L[0], L[1])
        gmsh_cell_type = "quadrangle"
        perm = [0, 3, 1, 2]
    elif gdim == 3:
        obj = factory.addBox(xmin[0], xmin[1], xmin[2], L[0], L[1], L[2])
        gmsh_cell_type = "hexahedron"
        perm = [2, 6, 3, 7, 1, 5, 0, 4]
    else:
        raise RuntimeError("Unknown dimension")

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
        from os import makedirs

        N_str = str(N).replace(" ", "_").replace("[", "").replace("]", "")
        makedirs("output", exist_ok=True)
        gmsh.write("output/" + model_name + "_" + N_str + ".msh")
        gmsh.write("output/" + model_name + "_" + N_str + ".mesh")

    gmsh.model.mesh.setOrder(degree)

    # 2. Convert to dolfinx (from test_quadrilateral_mesh from
    # test_higher_order_mesh.py in dolfinx)
    idx, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    assert coords.shape[0] == np.prod((degree * N + 1))
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = coords[srt, :gdim]
    element_types, _, node_tags = gmsh.model.mesh.getElements(dim=gdim)
    (
        _,
        dim,
        degree,
        num_nodes,
        _,
        _,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    cells = cells[:, perm]

    gmsh_cell_id = gmsh.model.mesh.getElementType(gmsh_cell_type, degree)
    domain = dolfinx.io.gmshio.ufl_mesh(gmsh_cell_id, x.shape[1])
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

    assert mesh.topology.index_map(0).size_local == np.prod(N + 1)
    assert mesh.topology.index_map(gdim).size_local == np.prod(N)
    assert mesh.geometry.x.shape[0] == np.prod((degree * N + 1))

    gmsh.finalize()

    return mesh
