// utils.c
// Utility functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "utils.h"

#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

gaussian_t* gaussian_alloc(size_t dim) {
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

gmm_t* gmm_alloc(size_t dim, size_t k) {
  size_t i;
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
  size_t i;
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

size_t discrete_gen(const gsl_rng* rng, const gsl_vector* dist) {
  size_t i;
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

double gaussian_pdf(const gaussian_t* dist,
    const gsl_vector* x) {
  double r = 0.0;
  int signum;

  gsl_vector* w1 = gsl_vector_alloc(dist->dim);
  gsl_vector* w2 = gsl_vector_alloc(dist->dim);
  gsl_vector_memcpy(w1, x);
  gsl_vector_sub(w1, dist->mean);

  gsl_matrix* v = gsl_matrix_alloc(dist->dim, dist->dim);
  gsl_matrix_memcpy(v, dist->cov);
  gsl_permutation* p = gsl_permutation_alloc(dist->dim);

  gsl_linalg_LU_decomp(v, p, &signum);
  gsl_linalg_LU_solve(v, p, w1, w2);
  gsl_blas_ddot(w1, w2, &r);
  double det = gsl_linalg_LU_det(v, signum);

  r = exp(-.5 * r) / sqrt(pow(2 * M_PI, dist->dim) * det);

  gsl_vector_free(w1);
  gsl_vector_free(w2);
  gsl_matrix_free(v);
  gsl_permutation_free(p);

  return r;
}

void gaussian_gen(const gsl_rng* rng, const gaussian_t* dist,
    gsl_vector* result) {
  size_t i;
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

double gmm_pdf(const gmm_t* gmm, const gsl_vector* x) {
  gsl_vector* p = gsl_vector_alloc(gmm->k);
  size_t i;
  for (i = 0; i < p->size; i++) {
    gsl_vector_set(p, i, gaussian_pdf(gmm->comp[i], x));
  }

  double result;
  gsl_blas_ddot(p, gmm->weight, &result);

  gsl_vector_free(p);
  return result;
}

void gmm_gen(const gsl_rng* rng, const gmm_t* gmm,
    gsl_vector* result) {
  size_t i = discrete_gen(rng, gmm->weight);
  gaussian_gen(rng, gmm->comp[i], result);
}

double log_sum_exp(const gsl_vector* v) {
  double m = -gsl_vector_max(v);
  if (m == HUGE_VAL) {
    return -HUGE_VAL;
  }

  gsl_vector* w = gsl_vector_alloc(v->size);
  gsl_vector_memcpy(w, v);
  gsl_vector_add_constant(w, m);


  double s = 0.0;
  int i;
  for (i = 0; i < w->size; i++) {
    s += exp(gsl_vector_get(w, i));
  }

  gsl_vector_free(w);

  return -m + log(s);
}

double max_index(gsl_vector* v, size_t* index) {
  size_t i;
  size_t id = -1;
  double m = -HUGE_VAL;
  for (i = 0; i < v->size; i++) {
    if (gsl_vector_get(v, i) > m) {
      id = i;
      m = gsl_vector_get(v, i);
    }
  }

  if (index) {
    *index = id;
  }

  return m;
}

