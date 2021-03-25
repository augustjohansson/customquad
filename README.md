# customquad

## Description

The customquad library allows for custom quadrature to be used in FEniCS-X. By custom quadrature we mean user-specified quadrature rules in different elements. These can be used for performing surface and volume integrals over cut elements in methods such as CutFEM, TraceFEM and \phi-FEM. The user can also provide normals in the surface quadrature points.

## Dependencies

This library is not yet a fully external library, but depend on the following forks of FFC-X and DOLFIN-X.

- https://github.com/augustjohansson/dolfinx
- https://github.com/augustjohansson/ffcx

To run the demos, the library needs the Algoim library.

- https://algoim.github.io

## Installation

Please use the provided docker file based on the dolfinx docker image. It will install all dependencies and build the library.

## License

The license is the intersection of the licenses for Algoim and the FEniCS-X project license. If this is the empty set.
