/* NotGHMM
 * Copyright (c) 2014 Weipeng He <heweipeng@gmail.com>
 * utils.c : Utility functions.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#include "utils.h"

#include <stdio.h>
#include <math.h>
#include <fenv.h>
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include <flann/flann.h>

#define MORE_THAN_ZERO 1e-10

gaussian_t* gaussian_alloc(size_t dim, int cov_diag) {
  gaussian_t* r = malloc(sizeof(gaussian_t));
  if (r) {
    r->dim = dim;

    r->mean = gsl_vector_alloc(dim);
    if (cov_diag) {
      r->cov = NULL;
      r->diag = gsl_vector_alloc(dim);
    }
    else {
      r->cov = gsl_matrix_alloc(dim, dim);
      r->diag = NULL;
    }

    if (!r->mean || !(r->cov || r->diag)) {
      free(r->mean);
      free(r->cov);
      free(r->diag);
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
    if (dist->diag) {
      gsl_vector_free(dist->diag);
    }
    free(dist);
  }
}

int gaussian_isdiagonal(const gaussian_t* dist) {
  if (dist->cov) {
    assert(!dist->diag);
    return 0;
  }
  else {
    assert(dist->diag);
    return 1;
  }
}

int gaussian_valid(const gaussian_t* dist) {
  if (!(dist && dist->dim > 0 && dist->mean->size == dist->dim)) {
    return 0;
  }

  // check positive definite 
  if (gaussian_isdiagonal(dist)) {
    size_t i = 0;
    for (i = 0; i < dist->dim; i++) {
      double x = gsl_vector_get(dist->diag, i);
      if (!isnormal(x) || x <= 0) {
        return 0;
      }
    }
  }
  else {
    gsl_matrix* v = gsl_matrix_alloc(dist->dim, dist->dim);
    gsl_matrix_memcpy(v, dist->cov);

    gsl_set_error_handler_off();
    int status = gsl_linalg_cholesky_decomp(v);
    gsl_set_error_handler(NULL);

    gsl_matrix_free(v);

    if (status == GSL_EDOM) {
      return 0;
    }
  }

  return 1;
}

void gaussian_memcpy(gaussian_t* dest, const gaussian_t* src) {
  assert(dest->dim == src->dim);

  gsl_vector_memcpy(dest->mean, src->mean);
  if (!gaussian_isdiagonal(src)) {
    if (gaussian_isdiagonal(dest)) {
      gsl_vector_free(dest->diag);
      dest->cov = gsl_matrix_alloc(src->dim, src->dim);
    }
    gsl_matrix_memcpy(dest->cov, src->cov);
  }
  else {
    if (!gaussian_isdiagonal(dest)) {
      gsl_matrix_free(dest->cov);
      dest->diag = gsl_vector_alloc(src->dim);
    }
    
    gsl_vector_memcpy(dest->diag, src->diag);
  }
}

gmm_t* gmm_alloc(size_t dim, size_t k, int cov_diag) {
  size_t i;
  int suc = 0;

  gmm_t* r = malloc(sizeof(gmm_t));
  if (r) {
    r->dim = dim;
    r->k = k;

    r->comp = calloc(k, sizeof(gaussian_t*));
    if (r->comp) {
      for (i = 0; i < k; i++) {
        r->comp[i] = gaussian_alloc(dim, cov_diag);
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
      free(gmm->comp);
    }

    free(gmm);
  }
}

int gmm_valid(const gmm_t* gmm) {
  if (!(gmm && gmm->k > 0 && gmm->dim > 0
        && discrete_valid(gmm->weight))) {
    return 0;
  }

  size_t i;
  for (i = 0; i < gmm->k; i++) {
    if (!gaussian_valid(gmm->comp[i])) {
      return 0;
    }
  }

  return 1;
}

void gmm_memcpy(gmm_t* dest, const gmm_t* src) {
  assert(dest->k == src->k && dest->dim == src->dim);

  gsl_vector_memcpy(dest->weight, src->weight);
  size_t i;
  for (i = 0; i < src->k; i++) {
    gaussian_memcpy(dest->comp[i], src->comp[i]);
  }
}

int discrete_valid(const gsl_vector* dist) {
  if (dist == NULL) {
    return 0;
  }
  size_t i;
  double x;
  double s = 0;
  for (i = 0; i < dist->size; i++) {
    x = gsl_vector_get(dist, i);
    if (x < 0.0) {
      return 0.0;
    }
    s += x;
  }

  if (fabs(s - 1.0) > MORE_THAN_ZERO) {
    return 0;
  }

  return 1;
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

double gaussian_pdf_log(const gaussian_t* dist,
    const gsl_vector* x) {
  double r = 0.0;
  double logdet = 0.0;

  if (gaussian_isdiagonal(dist)) {
    size_t i;
    double dx, dd;
    for (i = 0; i < dist->dim; i++) {
      dx = gsl_vector_get(x, i) - gsl_vector_get(dist->mean, i);
      dd = gsl_vector_get(dist->diag, i);
      r += dx * dx / dd;
      logdet += DEBUG_LOG(dd);
    }
  }
  else {
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
    logdet = gsl_linalg_LU_lndet(v);
    assert(gsl_linalg_LU_sgndet(v, signum) == 1.0);

    gsl_vector_free(w1);
    gsl_vector_free(w2);
    gsl_matrix_free(v);
    gsl_permutation_free(p);
  }

  /* Use log to avoid underflow !
     here
     r = (x - mean)^T * cov^-1 * (x - mean)
     logdet = log(det(cov))
     then
     logpdf = -.5 * (k * log(2*pi) + logdet + r);
   */
  r = r + dist->dim * DEBUG_LOG(2 * M_PI) + logdet;
  r = -0.5 * r;

  assert(!isnan(r));

  return r;
}

void gaussian_gen(const gsl_rng* rng, const gaussian_t* dist,
    gsl_vector* result) {
  assert(result->size == dist->dim);

  size_t i;
  for (i = 0; i < result->size; i++) {
    gsl_vector_set(result, i, gsl_ran_ugaussian(rng));
  }

  if (gaussian_isdiagonal(dist)) {
    for (i = 0; i < result->size; i++) {
      double* p = gsl_vector_ptr(result, i);
      *p *= DEBUG_SQRT(gsl_vector_get(dist->diag, i));
    }
  }
  else {
    gsl_matrix* v = gsl_matrix_alloc(dist->dim, dist->dim);
    gsl_matrix_memcpy(v, dist->cov);

    gsl_linalg_cholesky_decomp(v);
    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, v, result);

    gsl_matrix_free(v);
  }

  gsl_vector_add(result, dist->mean);
}

double gmm_pdf_log(const gmm_t* gmm, const gsl_vector* x) {
  gsl_vector* p = gsl_vector_alloc(gmm->k);
  size_t i;
  for (i = 0; i < p->size; i++) {
    gsl_vector_set(p, i, DEBUG_LOG(gsl_vector_get(gmm->weight, i))
        + gaussian_pdf_log(gmm->comp[i], x));
  }

  double result = log_sum_exp(p);

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
  assert(!isnan(m));
  if (isinf(m)) {
    // m = +inf OR -inf
    // both cases the result should be equal to m
    return m;
  }

  gsl_vector* w = gsl_vector_alloc(v->size);
  gsl_vector_memcpy(w, v);
  gsl_vector_add_constant(w, m);

  double s = 0.0;
  size_t i;
  for (i = 0; i < w->size; i++) {
    s += DEBUG_EXP(gsl_vector_get(w, i));
  }

  gsl_vector_free(w);

  return -m + DEBUG_LOG(s);
}

double math_func_fe_except(double (*func)(double),
    double x, const char* func_name, const char* file,
    unsigned int line) {
  errno = 0;
  feclearexcept(FE_ALL_EXCEPT);
  double r = func(x);
  if (fetestexcept(FE_INVALID | FE_DIVBYZERO 
        | FE_OVERFLOW /* | FE_UNDERFLOW */)) {
    fprintf(stderr, "Warning (%s:%d): FE Exception catched, "
        "%s(%g) = %g, %s\n" , file, line, func_name,
        x, r, strerror(errno));
  }
  return r;
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

void vector_fprint(FILE* stream, const gsl_vector* v) {
  size_t i;
  for (i = 0; i < v->size - 1; i++) {
    fprintf(stream, "%g ", gsl_vector_get(v, i));
  }
  fprintf(stream, "%g\n", gsl_vector_get(v, i));
}

void vector_fscan(FILE* stream, gsl_vector* v) {
  size_t i;
  for (i = 0; i < v->size - 1; i++) {
    fscanf(stream, "%lg ", gsl_vector_ptr(v, i));
  }
  fscanf(stream, "%lg\n", gsl_vector_ptr(v, i));
}

void matrix_fprint(FILE* stream, const gsl_matrix* m) {
  size_t i, j;
  for (i = 0; i < m->size1; i++) {
    for (j = 0; j < m->size2 - 1; j++) {
      fprintf(stream, "%g ", gsl_matrix_get(m, i, j));
    }
    fprintf(stream, "%g\n", gsl_matrix_get(m, i, j));
  }
}

void matrix_fscan(FILE* stream, gsl_matrix* m) {
  size_t i, j;
  for (i = 0; i < m->size1; i++) {
    for (j = 0; j < m->size2 - 1; j++) {
      fscanf(stream, "%lg ", gsl_matrix_ptr(m, i, j));
    }
    fscanf(stream, "%lg\n", gsl_matrix_ptr(m, i, j));
  }
}

static void cal_center(float* result, float* data, int* indices, int ncenter, int ndata, int cols) {
  int i, j, k;
  int t = 0;
  for (i = 0; i < ncenter; i++) {
    int c = 0;
    for (k = 0; k < cols; k++) {
      result[i * cols + k] = 0.0f;
    }
    for (j = 0; j < ndata; j++) {
      if (indices[j] == i) {
        for (k = 0; k < cols; k++) {
          result[i * cols + k] += data[j * cols + k];
        }
        c++;
      }
    }
    for (k = 0; k < cols; k++) {
      result[i * cols + k] /= (float) c;
    }
    t += c;
  }

  assert(t == ndata);
}

void kmeans_cluster(gsl_vector** data, size_t size,
    size_t k, size_t* index, gsl_vector** center) {
  assert(k > 0 && k <= size && k <= INT_MAX);
  if (!index && !center) {
    return;
  }

  size_t i, j;

  struct FLANNParameters flann_param = DEFAULT_FLANN_PARAMETERS;

  int cols = data[0]->size;
  int* cid = malloc(size * sizeof(int));
  float* dataset = malloc(k * cols * sizeof(float));
  float* dataold = malloc(k * cols * sizeof(float));
  float* testset = malloc(size * cols * sizeof(float));
  float* distance = malloc(size * sizeof(float));

  for (i = 0; i < size; i++) {
    for (j = 0; j < cols; j++) {
      testset[i * cols + j] = (float) gsl_vector_get(data[i], j);
    }
  }

  // random init
  for (i = 0; i < size; i++) {
    cid[i] = i % k;
  }

  cal_center(dataset, testset, cid, k, size, cols);

  float dist = 1.0;
  while (dist > .001) {
    memcpy(dataold, dataset, k * cols * sizeof(float));

    // new cid
    flann_find_nearest_neighbors(dataset, k, cols,
        testset, size, cid, distance, 1, &flann_param);

    // calculate new center
    cal_center(dataset, testset, cid, k, size, cols);

    // dist
    dist = 0.0;
    for (i = 0; i < k * cols; i++) {
      dist += fabsf(dataset[i] - dataold[i]);
    }
  }

  if (index) {
    for (i = 0; i < size; i++) {
      index[i] = cid[i];
    }
  }

  if (center) {
    for (i = 0; i < k; k++) {
      assert(center[i]->size == cols);
      for (j = 0; j < cols; j++) {
        gsl_vector_set(center[i], j, dataset[i * cols + j]);
      }
    }
  }

  free(cid);
  free(dataset);
  free(dataold);
  free(testset);
  free(distance);
}


