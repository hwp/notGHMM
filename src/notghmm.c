// notghmm.c
// Implementation.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>

#include <gsl/gsl_rng.h>

seq_t* seq_alloc(int size, int dim) {
  seq_t* r = malloc(sizeof(seq_t));
  if (r) {
    r->size = size;
    r->dim = dim;
    
    r->data = malloc(sizeof(double) * size * dim);
    if (!r->data) {
      free(r);
      r == NULL;
    }
  }
  return r;
}

void seq_free(seq_t* seq) {
  free(seq->data);
  free(seq);
}

hmmgmm_t* hmmgmm_alloc(int n, int m, int dim) {
  hmmgmm_t* r = malloc(sizeof(hmmgmm_t));
  if (r) {
    r->n = n;
    r->m = m;
    r->dim = dim;

    r->pi = malloc(sizeof(double) * n);
    r->a = malloc(sizeof(double) * n * n);
    r->c = malloc(sizeof(double) * n * m);
    r->mu = malloc(sizeof(double) * n * m * dim);
    r->sigma = malloc(sizeof(double) * n * m * dim * dim);
    
    if (!(r->a && r->c && r->mu && r->sigma)) {
      free(r->pi);
      free(r->a);
      free(r->c);
      free(r->mu);
      free(r->sigma);
      free(r);
      r = NULL;
    }
  }
  return r;
}

void hmmgmm_free(hmmgmm_t* model) {
  free(model->pi);
  free(model->a);
  free(model->c);
  free(model->mu);
  free(model->sigma);
  free(model);
}



seq_t* gen_sequence(hmmgmm_t* model, int size) {
  seq_t* seq = seq_alloc(size, model->dim);
  assert(seq);

  const gsl_rng_type* rngt = gsl_rng_default;
  gsl_rng* rng = gsl_rng_alloc(rngt);

  int q, k, t;
  double r;
  double* mu;
  double* sigma;
  for (t = 0; t < size; t++) {
    if (t == 0) {
      q = gen_discrete(rng, model->pi, model->n);
    }
    else {
      q = gen_discrete(rng, HMMGMM_A_ROW(model, q), model->n);
    }

    k = gen_discrete(rng, HMMGMM_C_ROW(model, q), model->m);
    
    mu = HMMGMM_MU(model, q, k);
    sigma = HMMGMM_SIGMA(model, q, k);

    gen_gaussian(rng, model->dim, mu, sigma, SEQ_DATA_AT(seq, t));
  }
  return seq;
}

