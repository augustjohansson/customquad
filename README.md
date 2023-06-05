# Customquad

## Description

The Customquad library allows for custom quadrature to be used in FEniCS-X. By custom quadrature we mean user-specified quadrature rules in different elements. These can be used for performing surface and volume integrals over cut elements in methods such as CutFEM, TraceFEM and \phi-FEM. The user can also provide normals in the surface quadrature points.

## Dependencies

The library depend on a fork of ffcx

- https://github.com/augustjohansson/ffcx

Some of the demos use the Algoim library for obtaining quadrature rules. It is found at

- https://algoim.github.io

## Installation

Please use the provided docker file based on the dolfinx docker image. It will install all fenics-related dependencies (but not Algoim) and build the library.

## License

The license is the same as the FEniCS-X project license.
