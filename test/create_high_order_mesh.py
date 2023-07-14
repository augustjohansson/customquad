# High order mesh generation copied from test_quadrilateral_mesh from
# test_higher_order_mesh.py in dolfinx. And the same for the hexes.

import dolfinx
import ufl
import basix
import numpy as np
from mpi4py import MPI


flatten = lambda l: [item for sublist in l for item in sublist]


def create_high_order_quad_mesh(Nx, Ny, polynomial_order):
    def coord_to_vertex(x, y):
        return (polynomial_order + 1) * y + x

    def get_points(order, Nx, Ny):
        points = []
        points += [[i / order, 0] for i in range(order + 1)]
        for j in range(1, order):
            points += [[i / order, j / order] for i in range(order + 1)]
        points += [[j / order, 1] for j in range(order + 1)]

        # Combine to several cells (test first w/o unique vertices)
        all_points = []
        pnp = np.array(points)

        ex = np.array([1.0, 0.0])
        for i in range(Nx):
            ptmp = pnp + i * ex
            all_points.append(ptmp.tolist())
        all_points_x = flatten(all_points)

        ey = np.array([0.0, 1.0])
        for j in range(1, Ny):
            for q in all_points_x:
                ptmp = np.array(q) + j * ey
                all_points.append([ptmp.tolist()])
        all_points = flatten(all_points)

        assert len(all_points) == (order + 1) ** 2 * Nx * Ny

        return all_points

    def get_cells(order, Nx, Ny):
        # Define a cell using DOLFINx ordering
        cell = [
            coord_to_vertex(i, j)
            for i, j in [(0, 0), (order, 0), (0, order), (order, order)]
        ]
        if order > 1:
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0))
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i))
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order))

            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j))

        # Combine to several cells as done for the points
        all_cells = []
        cnp = np.array(cell)
        n = len(cell)

        for i in range(Nx):
            ctmp = cnp + n * i
            all_cells.append(ctmp.tolist())

        cells_x = all_cells.copy()
        offset = np.array(cells_x).max() + 1

        for j in range(1, Ny):
            for cc in cells_x:
                ctmp = np.array(cc) + j * offset
                all_cells.append(ctmp.tolist())

        assert len(all_cells) == Nx * Ny

        return all_cells

    points = get_points(polynomial_order, Nx, Ny)
    cells = get_cells(polynomial_order, Nx, Ny)
    domain = ufl.Mesh(
        basix.ufl_wrapper.create_vector_element(
            "Q",
            "quadrilateral",
            polynomial_order,
            gdim=2,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
    )
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)
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
