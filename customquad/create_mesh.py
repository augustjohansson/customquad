import dolfinx
import numpy as np
from mpi4py import MPI
import gmsh
import ufl
import basix


def create_mesh(xrange, N, degree, debug=False):
    # 1. Create gmsh grid using transfinite interpolation
    gmsh.initialize()
    if not debug:
        gmsh.option.setNumber("General.Terminal", 0)
    factory = gmsh.model.occ
    model_name = "cq_create_mesh"
    gmsh.model.add(model_name)
    factory = gmsh.model.occ

    xmin = xrange[0]
    xmax = xrange[1]
    xdiff = xmax - xmin
    gdim = len(xmin)
    assert len(N) == gdim

    if gdim == 2:
        obj = factory.addRectangle(xmin[0], xmin[1], 0, xdiff[0], xdiff[1])
        gmsh_cell_type = "quadrangle"
        shape = "quadrilateral"

        if degree == 1:
            # Match dolfinx mesh
            perm = [0, 3, 1, 2]
        elif degree == 2:
            perm = [0, 1, 3, 2, 4, 7, 5, 6, 8]
        else:
            raise RuntimeError("Not implemented")

    elif gdim == 3:
        obj = factory.addBox(xmin[0], xmin[1], xmin[2], xdiff[0], xdiff[1], xdiff[2])
        gmsh_cell_type = "hexahedron"
        shape = "hexahedron"

        if degree == 1:
            # Match dolfinx mesh
            perm = [2, 6, 3, 7, 1, 5, 0, 4]
        elif degree == 2:
            # fmt: off
            perm = [2, 6, 3, 7,
                    1, 5, 0, 4,
                    14, 13, 11, 19,
                    18, 15, 9, 17,
                    12, 8, 16, 10,
                    24, 23, 20, 25,
                    22, 21, 26 ]
            # fmt: on
        else:
            raise RuntimeError("Not implemented")
    else:
        raise RuntimeError("Unknown dimension")

    factory.synchronize()

    # Lines
    for line in gmsh.model.getEntities(1):
        bbox = gmsh.model.getBoundingBox(line[0], line[1])
        dx = np.array(bbox[3:]) - np.array(bbox[:3])
        dim = np.argmax(dx)
        assert dim < gdim
        gmsh.model.mesh.setTransfiniteCurve(line[1], N[dim] + 1)

    # Surfaces
    for surf in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(surf[1])
        gmsh.model.mesh.setRecombine(surf[0], surf[1])

    if gdim == 3:
        # Volumes
        gmsh.model.mesh.setTransfiniteVolume(obj)
        gmsh.model.mesh.setRecombine(3, obj)

    gmsh.model.addPhysicalGroup(gdim, [obj])

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
    x = coords[srt, 0:gdim]

    element_types, _, node_tags = gmsh.model.mesh.getElements(dim=gdim)
    (
        _,
        dim,
        degree,
        num_nodes,
        _,
        _,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    assert len(element_types) == 1

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    cells = cells[:, perm]

    cell = ufl.Cell(shape, geometric_dimension=gdim)
    element = basix.ufl_wrapper.create_vector_element(
        basix.ElementFamily.P,
        cell.cellname(),
        degree,
        basix.LagrangeVariant.equispaced,
        dim=gdim,
        gdim=gdim,
    )
    domain = ufl.Mesh(element)
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

    assert mesh.topology.index_map(0).size_local == np.prod(N + 1)
    assert mesh.topology.index_map(gdim).size_local == np.prod(N)
    assert mesh.geometry.x.shape[0] == np.prod((degree * N + 1))

    gmsh.finalize()

    return mesh
