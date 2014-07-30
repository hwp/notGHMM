// notghmm.c
// Implementation.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"
#include "utils.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>

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

void hmmgmm_memcpy(hmmgmm_t* dest, const hmmgmm_t* src) {
  assert(dest->n == src->n && dest->k == src->k 
      && dest->dim == src->dim);
  gsl_vector_memcpy(dest->pi, src->pi);
  gsl_matrix_memcpy(dest->a, src->a);

  int i;
  for (i = 0; i < src->n; i++) {
    gmm_memcpy(dest->states[i], src->states[i]);
  }
}

seq_t* seq_gen(const hmmgmm_t* model, size_t size) {
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

void forward_proc(const hmmgmm_t* model, const seq_t* seq,
    gsl_matrix* alpha) {
  size_t i, t;

  for (i = 0; i < model->n; i++) {
    gsl_matrix_set(alpha, 0, i, gsl_vector_get(model->pi, i)
        * gmm_pdf(model->states[i], seq->data[0]));
  }

  gsl_vector* b = gsl_vector_alloc(model->n);
  for (t = 1; t < seq->size; t++) {
    for (i = 0; i < b->size; i++) {
      gsl_vector_set(b, i,
          gmm_pdf(model->states[i], seq->data[t]));
    }

    gsl_vector_view p = gsl_matrix_row(alpha, t - 1);
    gsl_vector_view n = gsl_matrix_row(alpha, t);
    gsl_blas_dgemv(CblasTrans, 1.0, model->a, &p.vector,
        0.0, &n.vector);
    gsl_vector_mul(&n.vector, b);
  }
  gsl_vector_free(b);
}

void forward_proc_log(const hmmgmm_t* model,
    const seq_t* seq, gsl_matrix* logalpha) {
  int i, j, t;

  gsl_matrix* loga = gsl_matrix_alloc(model->n, model->n);
  for (i = 0; i < model->n; i++) {
    for (j = 0; j < model->n; j++) {
      gsl_matrix_set(loga, i, j,
          log(gsl_matrix_get(model->a, i, j)));
    }
  }

  for (i = 0; i < model->n; i++) {
    gsl_matrix_set(logalpha, 0, i,
        log(gsl_vector_get(model->pi, i))
        + log(gmm_pdf(model->states[i], seq->data[0])));
  }

  gsl_vector* v = gsl_vector_alloc(model->n);

  for (t = 1; t < seq->size; t++) {
    gsl_vector_view p = gsl_matrix_row(logalpha, t - 1);
    gsl_vector_view n = gsl_matrix_row(logalpha, t);

    for (i = 0; i < model->n; i++) {
      gsl_vector_memcpy(v, &p.vector);
      gsl_vector_view a = gsl_matrix_column(loga, i);
      gsl_vector_add(v, &a.vector);
      gsl_vector_set(&n.vector, i, log_sum_exp(v)
          + log(gmm_pdf(model->states[i], seq->data[t])));
    }
  }

  gsl_matrix_free(loga);
  gsl_vector_free(v);
}

double hmm_log_likelihood(const gsl_matrix* logalpha) {
  gsl_vector_const_view v = gsl_matrix_const_row(logalpha,
      logalpha->size1 - 1);
  return log_sum_exp(&v.vector);
}

void backward_proc(const hmmgmm_t* model, const seq_t* seq,
    gsl_matrix* beta) {
  size_t i, t;

  for (i = 0; i < model->n; i++) {
    gsl_matrix_set(beta, seq->size - 1, i, 1.0);
  }

  gsl_vector* b = gsl_vector_alloc(model->n);
  for (t = seq->size - 1; t > 0; t--) {
    for (i = 0; i < model->n; i++) {
      gsl_vector_set(b, i,
          gmm_pdf(model->states[i], seq->data[t]));
    }

    gsl_vector_view p = gsl_matrix_row(beta, t);
    gsl_vector_view n = gsl_matrix_row(beta, t - 1);
    gsl_vector_mul(b, &p.vector);
    gsl_blas_dgemv(CblasNoTrans, 1.0, model->a, b, 0.0,
        &n.vector);
  }
  gsl_vector_free(b);
}

void backward_proc_log(const hmmgmm_t* model,
    const seq_t* seq, gsl_matrix* logbeta) {
  size_t i, j, t;

  gsl_matrix* loga = gsl_matrix_alloc(model->n, model->n);
  for (i = 0; i < model->n; i++) {
    for (j = 0; j < model->n; j++) {
      gsl_matrix_set(loga, i, j,
          log(gsl_matrix_get(model->a, i, j)));
    }
  }

  for (i = 0; i < model->n; i++) {
    gsl_matrix_set(logbeta, seq->size - 1, i, 0.0);
  }

  gsl_vector* b = gsl_vector_alloc(model->n);
  gsl_vector* v = gsl_vector_alloc(model->n);
  for (t = seq->size - 1; t > 0 ; t--) {
    for (i = 0; i < model->n; i++) {
      gsl_vector_set(b, i, log(gmm_pdf(model->states[i],
              seq->data[t])));
    }

    gsl_vector_view p = gsl_matrix_row(logbeta, t);
    gsl_vector_view n = gsl_matrix_row(logbeta, t - 1);
    gsl_vector_add(b, &p.vector);

    for (i = 0; i < model->n; i++) {
      gsl_vector_view a = gsl_matrix_row(loga, i);
      gsl_vector_memcpy(v, &a.vector);
      gsl_vector_add(v, b);
      gsl_vector_set(&n.vector, i, log_sum_exp(v));
    }
  }

  gsl_matrix_free(loga);
  gsl_vector_free(b);
  gsl_vector_free(v);
}

double viterbi_log(const hmmgmm_t* model, const seq_t* seq,
    size_t* hidden) {
  size_t i, j, t;
  double m;

  gsl_matrix* loga = gsl_matrix_alloc(model->n, model->n);
  for (i = 0; i < model->n; i++) {
    for (j = 0; j < model->n; j++) {
      gsl_matrix_set(loga, i, j,
          log(gsl_matrix_get(model->a, i, j)));
    }
  }

  gsl_vector* logp = gsl_vector_alloc(model->n);
  for (i = 0; i < model->n; i++) {
    gsl_vector_set(logp, i,
        log(gsl_vector_get(model->pi, i))
        + log(gmm_pdf(model->states[i], seq->data[0])));
  }

  size_t* track = calloc((seq->size - 1) * model->n,
      sizeof(size_t));

  gsl_vector* v = gsl_vector_alloc(model->n);
  gsl_vector* w = gsl_vector_alloc(model->n);
  for (t = 1; t < seq->size; t++) {
    gsl_vector_memcpy(v, logp);
    for (i = 0; i < model->n; i++) {
      gsl_vector_memcpy(w, v);
      gsl_vector_view a = gsl_matrix_column(loga, i);
      gsl_vector_add(w, &a.vector);
      m = max_index(w, track + (t - 1) * model->n + i);
      gsl_vector_set(logp, i, m + log(gmm_pdf(model->states[i],
              seq->data[t])));
    }
  }

  m = max_index(logp, &j);

  hidden[seq->size - 1] = j;
  for (t = seq->size - 1; t > 0; t--) {
    j = track[(t - 1) * model->n + j];
    hidden[t - 1] = j;
  }

  gsl_matrix_free(loga);
  gsl_vector_free(logp);
  gsl_vector_free(v);
  gsl_vector_free(w);
  free(track);

  return m;
}

void baum_welch(hmmgmm_t* model, seq_t** data, size_t nos) {
  hmmgmm_t* nmodel = hmmgmm_alloc(model->n, model->k,
      model->dim);
  gsl_vector* gamma = gsl_vector_alloc(model->n);
  gsl_vector* cgamma = gsl_vector_alloc(model->k);
  gsl_matrix* xi = gsl_matrix_alloc(model->n, model->n);
  gsl_matrix* loga = gsl_matrix_alloc(model->n, model->n);
  gsl_matrix* scgamma = gsl_matrix_alloc(model->n, model->k);
  gsl_vector* mean = gsl_vector_alloc(model->dim);
  gsl_matrix* cov = gsl_matrix_alloc(model->dim, model->dim);

  size_t i, j, s, t;
  double slogpo = -HUGE_VAL;
  double plogpo = -HUGE_VAL;
  int iter = 0;

  do {
    plogpo = slogpo;
    slogpo = 0.0;

    // Initialize
    gsl_vector_set_zero(nmodel->pi);
    gsl_matrix_set_zero(nmodel->a);
    for (i = 0; i < nmodel->n; i++) {
      gmm_t* state = nmodel->states[i];
      gsl_vector_set_zero(state->weight);
      for (j = 0; j < nmodel->k; j++) {
        gsl_vector_set_zero(state->comp[j]->mean);
        gsl_matrix_set_zero(state->comp[j]->cov);
      }
    }

    for (i = 0; i < model->n; i++) {
      for (j = 0; j < model->n; j++) {
        gsl_matrix_set(loga, i, j,
            log(gsl_matrix_get(model->a, i, j)));
      }
    }

    gsl_matrix_set_zero(scgamma);

    // Accumulate
    for (s = 0; s < nos; s++) {
      gsl_matrix* logalpha = gsl_matrix_alloc(data[s]->size,
          model->n);
      gsl_matrix* logbeta = gsl_matrix_alloc(data[s]->size,
          model->n);

      forward_proc_log(model, data[s], logalpha);
      backward_proc_log(model, data[s], logbeta);

      double logpo = hmm_log_likelihood(logalpha);
      slogpo += logpo;

      for (t = 0; t < data[s]->size; t++) {
        for (i = 0; i < model->n; i++) {
          // Calculate gamma
          gsl_vector_set(gamma, i, exp(
                gsl_matrix_get(logalpha, t, i)
                + gsl_matrix_get(logbeta, t, i) - logpo));
        }

        if (t == 0) {
          // Accumulate pi
          gsl_vector_add(nmodel->pi, gamma);
        }

        if (t < data[s]->size - 1) {
          // Caculate xi
          for (j = 0; j < model->n; j++) {
            double logb = log(gmm_pdf(model->states[j],
                  data[s]->data[t + 1]));
            for (i = 0; i < model->n; i++) {
              gsl_matrix_set(xi, i, j, exp(
                    gsl_matrix_get(logalpha, t, i)
                    + gsl_matrix_get(loga, i, j)
                    + gsl_matrix_get(logbeta, t + 1, j)
                    + logb - logpo));
            }
          }

          // Accumulate a
          gsl_matrix_add(nmodel->a, xi);
        }

        for (i = 0; i < model->n; i++) {
          gmm_t* state = model->states[i];
          gmm_t* nstate = nmodel->states[i];

          // Calculate cgamma of mixture components
          for (j = 0; j < model->k; j++) {
            gsl_vector_set(cgamma, j, 
                gsl_vector_get(state->weight, j)
                * gaussian_pdf(state->comp[j], data[s]->data[t]));
          }

          double sum = gsl_blas_dasum(cgamma);
          if (sum > 0) {
            // Normalize
            gsl_vector_scale(cgamma, 
                gsl_vector_get(gamma, i) / sum);

            // Accumulate mixture weight
            gsl_vector_add(nstate->weight, cgamma);

            // Accumulate mean and cov
            for (j = 0; j < model->k; j++) {
              gsl_vector_memcpy(mean, data[s]->data[t]);
              gsl_vector_scale(mean, gsl_vector_get(cgamma, j));
              gsl_vector_add(nstate->comp[j]->mean, mean);

              gsl_vector_memcpy(mean, data[s]->data[t]);
              gsl_vector_sub(mean, state->comp[j]->mean);
              gsl_matrix_set_zero(cov);
              gsl_blas_dger(gsl_vector_get(cgamma, j), mean, mean, cov);
              gsl_matrix_add(nstate->comp[j]->cov, cov);

              *gsl_matrix_ptr(scgamma, i, j) 
                += gsl_vector_get(cgamma, j);
            }
          }
        }
      }

      gsl_matrix_free(logalpha);
      gsl_matrix_free(logbeta);
    }

    // Normalize
    // Normalize pi, sum = 1
    gsl_vector_scale(nmodel->pi, 
        1.0 / gsl_blas_dasum(nmodel->pi));

    double scale;
    for (i = 0; i < model->n; i++) {
      // Normalize a, sum of each row = 1
      gsl_vector_view v = gsl_matrix_row(nmodel->a, i);
      scale = gsl_blas_dasum(&v.vector);
      if (scale > 0) {
        gsl_vector_scale(&v.vector, 1.0 / scale);

        gmm_t* state = nmodel->states[i];

        // Normalize weight, sum = 1
        gsl_vector_scale(state->weight, 
            1.0 / gsl_blas_dasum(state->weight));

        for (j = 0; j < model->k; j++) {
          scale = gsl_matrix_get(scgamma, i, j);
          if (scale > 0) {
            // Normalize (rescale) mean
            gsl_vector_scale(state->comp[j]->mean, 1.0 / scale);

            // Normalize (rescale) cov
            gsl_matrix_scale(state->comp[j]->cov, 1.0 / scale);
          }
          else {
            fprintf(stderr, "Warning: state %ld, component %ld "
                "not visited\n", i, j);
            // Copy from original model
            gaussian_memcpy(state->comp[j],
                model->states[i]->comp[j]);
          }
        }
      }
      else {
        fprintf(stderr, "Warning: state %ld, component %ld "
            "not visited\n", i, j);
        // Copy from original model
        gsl_vector_view vo = gsl_matrix_row(model->a, i);
        gsl_vector_memcpy(&v.vector, &vo.vector);
        gmm_memcpy(nmodel->states[i], model->states[i]);
      }
    }
    
    // Replace the original
    hmmgmm_memcpy(model, nmodel);

    // Here the log p is of the previous model
    // And the difference is calculated with 
    //   the previous previous model
    fprintf(stderr, "Iteration %d: log p = %g, "
        "difference = %g\n", iter, slogpo, slogpo - plogpo);
    iter++;
  } while (slogpo - plogpo > 0.01);

  hmmgmm_free(nmodel);
  gsl_vector_free(gamma);
  gsl_vector_free(cgamma);
  gsl_matrix_free(xi);
  gsl_matrix_free(loga);
  gsl_matrix_free(scgamma);
  gsl_vector_free(mean);
  gsl_matrix_free(cov);
}

