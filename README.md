# Customquad

The Customquad library allows for custom quadrature rules to be used
in FEniCSx (https://fenicsproject.org). By custom quadrature we mean
**user-specified quadrature rules in different elements specified at
runtime**. These can be used for performing surface and volume integrals
over cut elements in methods such as CutFEM, TraceFEM and
\phi-FEM. The user can also provide normals in the quadrature
points.

See the demo `poisson.py`, the tests and read the description below to
see how to use the library.

## Dependencies

In addition to dolfinx (https://github.com/FEniCS/dolfinx/) and basix
(https://github.com/FEniCS/basix/), the library depends to a large
extent on a fork of ffcx at
- https://github.com/augustjohansson/ffcx-custom

A small change is made to ufl to allow for normals in cell
integrals. To this end, this fork of ufl is needed
- https://github.com/augustjohansson/ufl-custom

Some of the demos use the Algoim library for obtaining quadrature
rules. It is found at
- https://algoim.github.io

## Installation (non-dev)

Please use the provided docker file based on the dolfinx docker
image. The docker file may be built and run from the main directory as
```
docker build -f docker/Dockerfile -t customquad .
docker run -it -v `pwd`:/root customquad bash -i
```
Then install the customquad module using pip, e.g.,
```
pip3 install . -U
```
Compiling the ffcx forms with runtime quadrature requires a C++
compiler, whereas standard ffcx forms is compiled using a C
compiler. For now we simply overwrite the C compiler with a C++
compiler. In addition, since C++ forbids pointer and integer
comparison, the -fpermissive flag must be set.
```
export CC="/usr/lib/ccache/g++ -fpermissive"
```
A bashrc file with useful aliases is provided in the utils directory.

## Installation (dev)

For the development of this library, the development of ffcx is the
most challenging part. I have the following setup:
```
git clone git@github.com:augustjohansson/customquad.git
cd customquad
git clone git@github.com:augustjohansson/ffcx-custom.git
git clone git@github.com:augustjohansson/ufl-custom.git
git config --global --add safe.directory /root/ffcx-custom
```
Then I start the container and use the `install-all` alias in the
provided bashrc.sh to install ffcx, ufl and customquad, as well as
overriding the C compiler with a C++ compiler as described above.

## How to contribute

If you find a bug or have suggestions for improvements, please place
an issue using the GitHub issue tracker. To contribute to the code,
please file a pull request or start an issue as a basis for
discussions. Please follow the code of conduct of
https://www.contributor-covenant.org.

## Description

The idea behind the library is to modify ffcx to change the generated
forms such that they evaluate basis functions in provided quadrature
points.

For example, we may be interested in evaluating the integral
```
a_bulk = \int_\Omega \nabla u \cdot \nabla v
````
in quadrature points that are different in each cell. Say we have a
list of these cells with quadrature points and weights:

```
cut_cells = [0, 1, 5]
qr_pts = [qr_pts_cell_0, qr_pts_cell_1, qr_pts_cell_5]
qr_w = [qr_w_cell_0, qr_w_cell_1, qr_w_cell_5]
```

The local quadrature points and weights are flat numpy arrays. Then
the runtime assembly is done by first constructing a custom measure
```
dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"})
```
Then, the customquad matrix assembly routines may be called on a
dolfinx form as
```
form = dolfinx.fem.form(a_bulk * dx_cut)
A = customquad.assemble_matrix(form, [(cut_cells, qr_pts, qr_w)])
A.assemble()
```
The reason for having a list of tuples in the second argument is to
allow for multiple forms with their own quadrature data in the future.

To understand a bit how the modifications to ffcx is done, we can look
in ffcx's cache directory (e.g. ~/.cache/fenics). Here there are files
such as `libffcx_forms_...c` which contaian standard tabulate tensor
functions that may look like
```cpp
void tabulate_tensor_integral_a0f3282139356df733c38db2e5d422f3272a1d5c(double*  A,
				    const double*  w,
				    const double*  c,
				    const double*  coordinate_dofs,
				    const int*  entity_local_index,
				    const uint8_t*  quadrature_permutation)
{
  // Quadrature rules
  static const double weights_8c4[16] = { 0.03025074832140047, 0.05671296296296294, 0.05671296296296292, 0.03025074832140047, 0.05671296296296294, 0.1063233257526736, 0.1063233257526736, 0.05671296296296294, 0.05671296296296292, 0.1063233257526736, 0.1063233257526735, 0.05671296296296292, 0.03025074832140047, 0.05671296296296294, 0.05671296296296292, 0.03025074832140047 };
  // Precomputed values of basis functions and precomputations
  // FE* dimensions: [permutation][entities][points][dofs]
  static const double FE8_C0_F_Q8c4[1][6][16][8] = {{{{ ... }}}};
  static const double FE9_C1_D001_F_Q8c4[1][6][16][8] = {{{{ ... }}}};
  static const double FE9_C1_D010_F_Q8c4[1][6][16][8] = {{{{ ... }}}};
  static const double FE9_C1_D100_F_Q8c4[1][6][16][8] = {{{{ ... }}}};
  ...
}
```
The function above is generated without runtime quadrature: the
weights and the basis functions are fixed. What the custom ffcx
implementation does is to generate code such that basix
(https://github.com/FEniCS/basix/) is used to evaluate the basis using
quadrature points given at runtime. Corresponding weights must also be
provided. If the form involves normals, these must be provided in the
quadrature points.

A tabulate tensor function with runtime quadrature may look like this:
```cpp
void tabulate_tensor_integral_3edb7c068402923a697d72e1b03e0957554f29c3(double* A,
				    const double* w,
				    const double* c,
				    const double* coordinate_dofs,
				    const int* entity_local_index,
				    const uint8_t* quadrature_permutation,
				    int num_quadrature_points,
				    const double* quadrature_points,
				    const double* quadrature_weights,
				    const double* quadrature_normals)
{
  // Quadrature rules
  const double* weights_8eb = quadrature_weights;
  // Precomputed values of basis functions and precomputations
  // FE* dimensions: [permutation][entities][points][dofs]
  double**** FE8_C0_Q8eb;
  double**** FE9_C0_D100_Q8eb;
  double**** FE9_C1_D010_Q8eb;
  double**** FE9_C2_D001_Q8eb;
  // Compute basis and/or derivatives using basix
  call_basix(&FE8_C0_Q8eb, num_quadrature_points, quadrature_points, 0, 1, 5, 1, 0, 3);
  call_basix(&FE9_C0_D100_Q8eb, num_quadrature_points, quadrature_points, 1, 1, 5, 1, 0, 3);
  call_basix(&FE9_C1_D010_Q8eb, num_quadrature_points, quadrature_points, 2, 1, 5, 1, 0, 3);
  call_basix(&FE9_C2_D001_Q8eb, num_quadrature_points, quadrature_points, 3, 1, 5, 1, 0, 3);
  ...
}
```
The `call_basix` function is a C++ function that evaluates the basis
(and derivatives) using basix (see `call_basix.hpp`). There's room for
improvement here: one call to `call_basix` should be sufficient. Note
that evaluating the Jacobian needs derivatives of the basis. The fixed
arguments to `call_basix` include type of basis function, which
derivative to compute, quadrature degree, and more.

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
