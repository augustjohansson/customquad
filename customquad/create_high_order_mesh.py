# High order mesh generation copied from test_quadrilateral_mesh from
# test_higher_order_mesh.py in dolfinx. And the same for the hexes.

import dolfinx
import ufl
import basix
import numpy as np
from mpi4py import MPI
import gmsh

flatten = lambda l: [item for sublist in l for item in sublist]


def create_high_order_quad_mesh(xrange, N, order, debug=False):
    gdim = 2
    gmsh.initialize()
    model_name = "create_high_order_quad_mesh"
    gmsh.model.add(model_name)
    factory = gmsh.model.geo

    xmin = xrange[0]
    xmax = xrange[1]

    p0 = factory.addPoint(xmin[0], xmin[1], 0)
    p1 = factory.addPoint(xmax[0], xmin[1], 0)
    p2 = factory.addPoint(xmin[0], xmax[1], 0)
    p3 = factory.addPoint(xmax[0], xmax[1], 0)
    l0 = factory.addLine(p0, p1)
    l1 = factory.addLine(p0, p2)
    l2 = factory.addLine(p1, p3)
    l3 = factory.addLine(p2, p3)
    cl = factory.addCurveLoop([l0, l2, -l3, -l1])
    surf = factory.addPlaneSurface([cl])

    gmsh.model.geo.mesh.setTransfiniteCurve(l0, N[0] + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, N[0] + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, N[1] + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, N[1] + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surf)

    gmsh.model.geo.mesh.setRecombine(2, surf)
    factory.synchronize()

    # Create mesh
    gmsh.model.mesh.generate(gdim)
    if debug:
        gmsh.write(model_name + "_" + str(N[0]) + "_" + str(N[1]) + ".mesh")
    gmsh.model.mesh.setOrder(order)

    # Mesh conversion
    idx, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    assert coords.shape[0] == (N[0] + 1) * (N[1] + 1)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = coords[srt, :gdim]
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=gdim)
    (
        name,
        dim,
        order,
        num_nodes,
        local_coords,
        num_first_order_nodes,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    cell_type = dolfinx.mesh.CellType.quadrilateral
    cells = cells[:, dolfinx.io.gmshio.cell_perm_array(cell_type, cells.shape[1])]

    gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    mesh = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, cells, x, dolfinx.io.gmshio.ufl_mesh(gmsh_cell_id, x.shape[1])
    )

    assert mesh.topology.index_map(0).size_local == (N[0] + 1) * (N[1] + 1)
    assert mesh.topology.index_map(gdim).size_local == N[0] * N[1]

    gmsh.finalize()

    return mesh


def create_high_order_hex_mesh(Nx, Ny, Nz, polynomial_order):
    def coord_to_vertex(x, y, z):
        return (polynomial_order + 1) ** 2 * z + (polynomial_order + 1) * y + x

    def get_points(order, Nx, Ny, Nz):
        points = []
        points += [
            [i / order, j / order, 0]
            for j in range(order + 1)
            for i in range(order + 1)
        ]
        for k in range(1, order):
            points += [
                [i / order, j / order + 0.1, k / order]
                for j in range(order + 1)
                for i in range(order + 1)
            ]
        points += [
            [i / order, j / order, 1]
            for j in range(order + 1)
            for i in range(order + 1)
        ]

        # Combine to several cells (vertices doesn't have to be unique)
        all_points = []
        pnp = np.array(points)

        ex = np.array([1, 0, 0])
        for i in range(Nx):
            ptmp = pnp + i * ex
            all_points.append(ptmp.tolist())  # extend?
        all_points_x = flatten(all_points)

        ey = np.array([0, 1, 0])
        for j in range(1, Ny):
            for q in all_points_x:
                ptmp = np.array(q) + j * ey
                all_points.append([ptmp.tolist()])
        all_points_xy = flatten(all_points)

        ez = np.array([0, 0, 1])
        for k in range(1, Nz):
            for q in all_points_xy:
                ptmp = np.array(q) + k * ez
                all_points.append([ptmp.tolist()])
        all_points = flatten(all_points)

        assert len(all_points) == (order + 1) ** 3 * Nx * Ny * Nz

        return all_points

    def get_cells(order, Nx, Ny, Nz):
        # Define a cell using DOLFINx ordering
        cell = [
            coord_to_vertex(x, y, z)
            for x, y, z in [
                (0, 0, 0),
                (order, 0, 0),
                (0, order, 0),
                (order, order, 0),
                (0, 0, order),
                (order, 0, order),
                (0, order, order),
                (order, order, order),
            ]
        ]

        if order > 1:
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, 0, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, 0, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, order, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, order, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0, order))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i, order))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i, order))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order, order))

            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j, 0))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, 0, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(0, i, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(order, i, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, order, j))
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j, order))

            for k in range(1, order):
                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(i, j, k))

        # Combine to several cells as done for the points
        all_cells = []
        cnp = np.array(cell)
        n = len(cell)

        for i in range(Nx):
            ctmp = cnp + n * i
            all_cells.append(ctmp.tolist())

        cells_x = all_cells.copy()
        offset_x = np.array(cells_x).max() + 1

        for j in range(1, Ny):
            for cc in cells_x:
                ctmp = np.array(cc) + j * offset_x
                all_cells.append(ctmp.tolist())

        cells_xy = all_cells.copy()
        offset_xy = np.array(cells_xy).max() + 1

        for k in range(1, Nz):
            for cc in cells_xy:
                ctmp = np.array(cc) + k * offset_xy
                all_cells.append(ctmp.tolist())

        assert len(all_cells) == Nx * Ny * Nz

        return all_cells

    points = get_points(polynomial_order, Nx, Ny, Nz)
    cells = get_cells(polynomial_order, Nx, Ny, Nz)
    domain = ufl.Mesh(
        basix.ufl_wrapper.create_vector_element(
            "Q",
            "hexahedron",
            polynomial_order,
            gdim=3,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
    )

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)

    return mesh
