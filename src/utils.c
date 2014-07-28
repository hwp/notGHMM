// utils.c
// Utility functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "utils.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

gaussian_t* gaussian_alloc(int dim) {
  gaussian_t* r = malloc(sizeof(gaussian_t)); 
  if (r) {
    r->dim = dim;

    r->mean = gsl_vector_alloc(dim);
    r->cov = gsl_matrix_alloc(dim, dim);
    
    if (!(r->mean && r->cov)) {
      free(r);
      r = NULL;
    }
  }
  return r;
}

void gaussian_free(gaussian_t* dist) {
  if (dist) {
    if (dist->mean) {
      gsl_vector_free(dist->mean);
    }
    if (dist->cov) {
      gsl_matrix_free(dist->cov);
    }
    free(dist);
  }
}

gmm_t* gmm_alloc(int dim, int k) {
  int i;
  int suc = 0;
  
  gmm_t* r = malloc(sizeof(gmm_t));
  if (r) {
    r->dim = dim;
    r->k = k;

    r->comp = calloc(k, sizeof(gaussian_t*));
    if (r->comp) {
      for (i = 0; i < k; i++) {
        r->comp[i] = gaussian_alloc(dim);
        if (!r->comp[i]) {
          break;
        }
      }

      if (i == k) {
        suc = 1;
      }
    }

    if (suc) {
      r->weight = gsl_vector_alloc(k);
      if (!r->weight) {
        suc = 0;
      }
    }
  }

  if (suc) {
    return r;
  }
  else {
    free(r);
    return NULL;
  }
}

void gmm_free(gmm_t* gmm) {
  int i;
  if (gmm) {
    if (gmm->weight) {
      gsl_vector_free(gmm->weight);
    }

    if (gmm->comp) {
      for (i = 0; i < gmm->k; i++) {
        gaussian_free(gmm->comp[i]);
      }
    }

    free(gmm);
  }
}

int discrete_gen(gsl_rng* rng, gsl_vector* dist) {
  int i;
  double s = 0.0;
  double v = gsl_rng_uniform(rng);

  for(i = 0; i < dist->size - 1; i++) {
    s += gsl_vector_get(dist, i);
    if (s >= v) {
      break;
    }
  }
  return i;
}

void gaussian_gen(gsl_rng* rng, gaussian_t* dist,
    gsl_vector* result) {
  int i;
  for (i = 0; i < result->size; i++) {
    gsl_vector_set(result, i, gsl_ran_ugaussian(rng));
  }

  gsl_matrix* v = gsl_matrix_alloc(dist->dim, dist->dim);
  gsl_matrix_memcpy(v, dist->cov);

  gsl_linalg_cholesky_decomp(v);
  gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, v, result);

  gsl_vector_add(result, dist->mean);

  gsl_matrix_free(v);
}

void gmm_gen(gsl_rng* rng, gmm_t* gmm, gsl_vector* result) {
  int i = discrete_gen(rng, gmm->weight);
  gaussian_gen(rng, gmm->comp[i], result);
}

