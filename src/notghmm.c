// notghmm.c
// Implementation.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"
#include "utils.h"

#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <gsl/gsl_rng.h>

seq_t* seq_alloc(size_t size, size_t dim) {
  size_t i;
  int suc = 0;
  seq_t* r = malloc(sizeof(seq_t));
  if (r) {
    r->size = size;
    r->dim = dim;

    r->data = calloc(size, sizeof(gsl_vector*));
    if (r->data) {
      for (i = 0; i < size; i++) {
        r->data[i] = gsl_vector_alloc(dim);
        if (!r->data[i]) {
          break;
        }
      }

      if (i == size) {
        suc = 1;
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

void seq_free(seq_t* seq) {
  size_t i;
  if (seq) {
    if (seq->data) {
      for (i = 0; i < seq->size; i++) {
        gsl_vector_free(seq->data[i]);
      }
      free(seq->data);
    }
    free(seq);
  }
}

hmmgmm_t* hmmgmm_alloc(size_t n, size_t k, size_t dim) {
  size_t i;
  int suc = 0;

  hmmgmm_t* r = malloc(sizeof(hmmgmm_t));
  if (r) {
    r->n = n;
    r->k = k;
    r->dim = dim;

    r->states = calloc(n, sizeof(gmm_t*));
    if (r->states) {
      for (i = 0; i < n; i++) {
        r->states[i] = gmm_alloc(dim, k);
        if (!r->states[i]) {
          break;
        }
      }

      if (i == n) {
        suc = 1;
      }
    }

    if (suc) {
      r->pi = gsl_vector_alloc(n);
      r->a = gsl_matrix_alloc(n, n);

      if (!(r->pi && r->a)) {
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

void hmmgmm_free(hmmgmm_t* model) {
  size_t i;
  if (model) {
    if (model->pi) {
      gsl_vector_free(model->pi);
    }

    if (model->a) {
      gsl_matrix_free(model->a);
    }

    if (model->states) {
      for (i = 0; i < model->n; i++) {
        gmm_free(model->states[i]);
      }
      free(model->states);
    }

    free(model);
  }
}

seq_t* seq_gen(hmmgmm_t* model, size_t size) {
  seq_t* seq = seq_alloc(size, model->dim);
  assert(seq);

  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, time(NULL));

  size_t q, t;
  for (t = 0; t < size; t++) {
    if (t == 0) {
      q = discrete_gen(rng, model->pi);
    }
    else {
      gsl_vector_view view = gsl_matrix_row(model->a, q);
      q = discrete_gen(rng, &view.vector);
    }
    
    gmm_gen(rng, model->states[q], seq->data[t]);
  }
  return seq;
}

/*
   void forward_proc(hmmgmm_t* model, seq_t* seq, double* alpha) {
   size_t i, j, t;

   for (i = 0; i < model->n; i++) {
   alpha[i] = model->pi[i] * pdf_gmm( // TODO
   }
   }
 */
