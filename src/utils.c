// utils.c
// Utility functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

void gen_gaussian(gsl_rng* rng, int dim, double* mu, double* sigma, double* result) {
  int i, j;
  gsl_matrix* v = gsl_matrix_alloc(dim, dim);
  gsl_vector* r = gsl_vector_alloc(dim);

  for (i = 0; i < dim; i++) {
    gsl_vector_set(r, i, gsl_ran_ugaussian(rng));
    for (j = 0; j < dim; j++) {
      gsl_matrix_set(v, i, j, sigma[i * dim + j]);
    }
  }

  gsl_linalg_cholesky_decomp(v);
  gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, v, r);

  for (i = 0; i < dim; i++) {
    result[i] = mu[i] + gsl_vector_get(r, i);
  }

  gsl_matrix_free(v);
  gsl_vector_free(r);
}

int gen_discrete(gsl_rng *rng, double* dist, int n) {
  int i;
  double s = 0.0;
  double v = gsl_rng_uniform(rng);

  for(i = 0; i < n - 1; i++) {
    s += dist[i];
    if (s >= v) {
      break;
    }
  }
  return i;
}

