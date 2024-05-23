from petsc4py import PETSc
import numpy as np

def print_for_header(
    b_local,
    coeffs,
    constants,
    cell_coords,
    entity_local_index,
    quadrature_permutation,
    num_quadrature_points,
    qr_pts,
    qr_w,
    qr_n,
):
    def print_flat(x):
        print("{", end="")
        for xi in x:
            print(xi, end=",")
        print("};")

    print("printing function params:")
    print("double A[] = ", end="")
    print_flat(b_local)
    print("const double w[] = ", end="")
    print_flat(coeffs)
    print("const double c[] = ", end="")
    print_flat(constants)
    print("const double coordinate_dofs[] = ", end="")
    print_flat(cell_coords.flatten())
    print("const int entity_local_index[] = ", end="")
    print_flat(entity_local_index.flatten())
    print("const uint8_t quadrature_permutation[] = ", end="")
    print_flat(quadrature_permutation.flatten())
    print("const int num_quadrature_points = ", num_quadrature_points, ";")
    print("const double quadrature_points[] = ", end="")
    print_flat(qr_pts)
    print("const double quadrature_weights[] = ", end="")
    print_flat(qr_w)
    print("const double facet_normals[] = ", end="")
    print_flat(qr_n)
    print(
        "tabulate_tensor_integral_custom_otherwise(A,w,c,coordinate_dofs,entity_local_index,quadrature_permutation,num_quadrature_points,quadrature_points,quadrature_weights,facet_normals);"
    )


def dump(filename, A, do_print=False):
    print(f"dump to {filename}")

    if isinstance(A, PETSc.Mat):
        assert A.assembled
        with open(filename, "w") as f:
            for r in range(A.size[0]):
                cols, vals = A.getRow(r)
                for col, val in zip(cols, vals):
                    s = str(r + 1) + " " + str(col + 1) + " " + str(val) + "\n"
                    f.write(s)
                    if do_print:
                        print(s, end="")
    else:
        np.savetxt(filename, A.array)
