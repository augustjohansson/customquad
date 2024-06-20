#ifndef CALL_BASIX_H
#define CALL_BASIX_H

#ifdef __cplusplus
extern "C"
{
#endif

  void call_basix(double***** FE,
		  int num_quadrature_points,
		  const double* quadrature_points,
		  int basix_derivative,
		  int family,
		  int cell_type,
		  int degree,
		  int lattice_type,
		  int gdim);

#ifdef __cplusplus
}
#endif

#endif
