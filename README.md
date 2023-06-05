# Customquad

## Description

The Customquad library allows for custom quadrature to be used in
FEniCS-X. By custom quadrature we mean user-specified quadrature rules
in different elements. These can be used for performing surface and
volume integrals over cut elements in methods such as CutFEM, TraceFEM
and \phi-FEM. The user can also provide normals in the surface
quadrature points.

## Dependencies

The library depends to a large extent on a fork of ffcx

- https://github.com/augustjohansson/ffcx

Some of the demos use the Algoim library for obtaining quadrature
rules. It is found at

- https://algoim.github.io

## Installation

Please use the provided docker file based on the dolfinx docker
image. It will install all fenics-related dependencies (but not
Algoim) and build the library.

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
