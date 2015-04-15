/* NotGHMM
 * Copyright (c) 2014 Weipeng He <heweipeng@gmail.com>
 * notghmm.c : Implementation
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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "notghmm.h"
#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>

#define BW_STOP_THRESHOLD 0.1
#define NUM_INIT_SAMPLES 20

#define MORE_THAN_ZERO 1e-20
#define MORE_THAN_ONE (1.00001)
#define SMALL_DIAGONAL_CORRECTION 1e-6

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

hmmgmm_t* hmmgmm_alloc(size_t n, size_t k, size_t dim, int cov_diag) {
  size_t i;
  int suc = 0;

  hmmgmm_t* r = malloc(sizeof(hmmgmm_t));
  if (r) {
    r->n = n;
    r->k = k;
    r->dim = dim;
    r->cov_diag = cov_diag;

    r->states = calloc(n, sizeof(gmm_t*));
    if (r->states) {
      for (i = 0; i < n; i++) {
        r->states[i] = gmm_alloc(dim, k, cov_diag);
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

int hmmgmm_valid(const hmmgmm_t* model) {
  size_t i;

  if (!(model && model->a && model->pi && model->states)) {
    return 0;
  }
  
  if (!discrete_valid(model->pi)) {
    return 0;
  }

  for (i = 0; i < model->n; i++) {
    if (!gmm_valid(model->states[i])) {
      return 0;
    }
  }

  for (i = 0; i < model->n; i++) {
    gsl_vector_view v = gsl_matrix_row(model->a, i);
    if (!discrete_valid(&v.vector)) {
      return 0;
    }
  }

  return 1;
}

void hmmgmm_memcpy(hmmgmm_t* dest, const hmmgmm_t* src) {
  assert(dest->n == src->n && dest->k == src->k 
      && dest->dim == src->dim);
  gsl_vector_memcpy(dest->pi, src->pi);
  gsl_matrix_memcpy(dest->a, src->a);

  size_t i;
  for (i = 0; i < src->n; i++) {
    gmm_memcpy(dest->states[i], src->states[i]);
  }
}

void hmmgmm_fprint(FILE* stream, const hmmgmm_t* model) {
  fprintf(stream, "HMM Parameters\n");
  fprintf(stream, "N = %zu\n", model->n);
  fprintf(stream, "K = %zu\n", model->k);
  fprintf(stream, "d = %zu\n", model->dim);
  fprintf(stream, "pi = ");
  vector_fprint(stream, model->pi);
  fprintf(stream, "a = \n");
  matrix_fprint(stream, model->a);
  size_t i, j;
  for (i = 0; i < model->n; i++) {
    gmm_t* state = model->states[i];
    fprintf(stream, "\nState %zu\n", i);
    fprintf(stream, "weight = ");
    vector_fprint(stream, state->weight);

    for (j = 0; j < model->k; j++) {
      fprintf(stream, "Component %zu\n", j);
      fprintf(stream, "mean = ");
      vector_fprint(stream, state->comp[j]->mean);
      if (model->cov_diag) {
        fprintf(stream, "cov (diag) = ");
        vector_fprint(stream, state->comp[j]->diag);
      }
      else {
        fprintf(stream, "cov = \n");
        matrix_fprint(stream, state->comp[j]->cov);
      }
    }
  }
}

void hmmgmm_fwrite(FILE* stream, const hmmgmm_t* model) {
  size_t n = model->n;
  size_t k = model->k;
  size_t dim = model->dim;
  fwrite(&n, sizeof(size_t), 1, stream);
  fwrite(&k, sizeof(size_t), 1, stream);
  fwrite(&dim, sizeof(size_t), 1, stream);
  fwrite(&model->cov_diag, sizeof(int), 1, stream);

  gsl_vector_fwrite(stream, model->pi);
  gsl_matrix_fwrite(stream, model->a);
  size_t i, j;
  for (i = 0; i < model->n; i++) {
    gmm_t* state = model->states[i];
    gsl_vector_fwrite(stream, state->weight);

    for (j = 0; j < model->k; j++) {
      gsl_vector_fwrite(stream, state->comp[j]->mean);
      if (model->cov_diag) {
        gsl_vector_fwrite(stream, state->comp[j]->diag);
      }
      else {
        gsl_matrix_fwrite(stream, state->comp[j]->cov);
      }
    }
  }
}

hmmgmm_t* hmmgmm_fread(FILE* stream) {
  size_t n, k, dim;
  int cov_diag;
  fread(&n, sizeof(size_t), 1, stream);
  fread(&k, sizeof(size_t), 1, stream);
  fread(&dim, sizeof(size_t), 1, stream);
  fread(&cov_diag, sizeof(int), 1, stream);

  hmmgmm_t* model = hmmgmm_alloc(n, k, dim, cov_diag);

  gsl_vector_fread(stream, model->pi);
  gsl_matrix_fread(stream, model->a);
  size_t i, j;
  for (i = 0; i < model->n; i++) {
    gmm_t* state = model->states[i];
    gsl_vector_fread(stream, state->weight);

    for (j = 0; j < model->k; j++) {
      gsl_vector_fread(stream, state->comp[j]->mean);
      if (model->cov_diag) {
        gsl_vector_fread(stream, state->comp[j]->diag);
      }
      else {
        gsl_matrix_fread(stream, state->comp[j]->cov);
      }
    }
  }

  return model;
}

seq_t* seq_gen(const hmmgmm_t* model, size_t size,
    const gsl_rng* rng) {
  seq_t* seq = seq_alloc(size, model->dim);
  assert(seq);

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

void forward_proc_log(const hmmgmm_t* model,
    const seq_t* seq, gsl_matrix* logalpha) {
  size_t i, j, t;

  gsl_matrix* loga = gsl_matrix_alloc(model->n, model->n);
  for (i = 0; i < model->n; i++) {
    for (j = 0; j < model->n; j++) {
      gsl_matrix_set(loga, i, j,
          DEBUG_LOG(gsl_matrix_get(model->a, i, j)));
    }
  }

  for (i = 0; i < model->n; i++) {
    gsl_matrix_set(logalpha, 0, i,
        DEBUG_LOG(gsl_vector_get(model->pi, i))
        + gmm_pdf_log(model->states[i], seq->data[0]));
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
          + gmm_pdf_log(model->states[i], seq->data[t]));
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

double hmm_log_likelihood_all(const hmmgmm_t* model, seq_t** data, size_t nos) {
  size_t i;
  double logl = 0.0;
  for (i = 0; i < nos; i++) {
    gsl_matrix* logalpha = gsl_matrix_alloc(data[i]->size, data[i]->dim);
    forward_proc_log(model, data[i], logalpha);
    logl += hmm_log_likelihood(logalpha);
    gsl_matrix_free(logalpha);
  }

  return logl;
}

void backward_proc_log(const hmmgmm_t* model,
    const seq_t* seq, gsl_matrix* logbeta) {
  size_t i, j, t;

  gsl_matrix* loga = gsl_matrix_alloc(model->n, model->n);
  for (i = 0; i < model->n; i++) {
    for (j = 0; j < model->n; j++) {
      gsl_matrix_set(loga, i, j,
          DEBUG_LOG(gsl_matrix_get(model->a, i, j)));
    }
  }

  for (i = 0; i < model->n; i++) {
    gsl_matrix_set(logbeta, seq->size - 1, i, 0.0);
  }

  gsl_vector* b = gsl_vector_alloc(model->n);
  gsl_vector* v = gsl_vector_alloc(model->n);
  for (t = seq->size - 1; t > 0 ; t--) {
    for (i = 0; i < model->n; i++) {
      gsl_vector_set(b, i, gmm_pdf_log(model->states[i],
              seq->data[t]));
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
          DEBUG_LOG(gsl_matrix_get(model->a, i, j)));
    }
  }

  gsl_vector* logp = gsl_vector_alloc(model->n);
  for (i = 0; i < model->n; i++) {
    gsl_vector_set(logp, i,
        DEBUG_LOG(gsl_vector_get(model->pi, i))
        + gmm_pdf_log(model->states[i], seq->data[0]));
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
      gsl_vector_set(logp, i, m 
          + gmm_pdf_log(model->states[i], seq->data[t]));
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

void random_init(hmmgmm_t* model, seq_t** data, size_t nos,
    gsl_rng* rng) {
  assert(nos > 0);

  size_t i, j, k;
  double p, sum;
  
  sum = 0.0;
  for (i = 0; i < model->n; i++) {
    p = 0.75 + 0.5 * gsl_rng_uniform(rng);
    gsl_vector_set(model->pi, i, p);
    sum += p;
  }
  gsl_vector_scale(model->pi, 1.0 / sum);

  for (j = 0; j < model->n; j++) {
    gsl_vector_view v = gsl_matrix_row(model->a, j);
    sum = 0.0;
    for (i = 0; i < model->n; i++) {
      p = 0.75 + 0.5 * gsl_rng_uniform(rng);
      gsl_vector_set(&v.vector, i, p);
      sum += p;
    }
    gsl_vector_scale(&v.vector, 1.0 / sum);
  }

  for (j = 0; j < model->n; j++) {
    gsl_vector* v = model->states[j]->weight;
    sum = 0.0;
    for (i = 0; i < model->k; i++) {
      p = 0.75 + 0.5 * gsl_rng_uniform(rng);
      gsl_vector_set(v, i, p);
      sum += p;
    }
    gsl_vector_scale(v, 1.0 / sum);
  }

  gsl_vector* samples[NUM_INIT_SAMPLES];
  gsl_vector* dx = gsl_vector_alloc(model->dim);
  
  for (j = 0; j < model->n; j++) {
    for (i = 0; i < model->k; i++) {
      size_t sid = (size_t) (nos * gsl_rng_uniform(rng));
      seq_t* seq = data[sid];
      gsl_ran_sample(rng, samples, NUM_INIT_SAMPLES,
          seq->data, seq->size, sizeof(gsl_vector*));
      
      gsl_vector* mean = model->states[j]->comp[i]->mean;
      gsl_vector* diag = model->states[j]->comp[i]->diag;
      gsl_matrix* cov = model->states[j]->comp[i]->cov;

      gsl_vector_set_zero(mean);

      if (model->cov_diag) {
        gsl_vector_set_zero(diag);
      }
      else {
        gsl_matrix_set_zero(cov);
      }

      for (k = 0; k < NUM_INIT_SAMPLES; k++) {
        gsl_vector_add(mean, samples[k]);
      }
      gsl_vector_scale(mean, 1.0 / NUM_INIT_SAMPLES);

      for (k = 0; k < NUM_INIT_SAMPLES; k++) {
        gsl_vector_memcpy(dx, samples[k]);
        gsl_vector_sub(dx, mean);
        if (model->cov_diag) {
          gsl_vector_mul(dx, dx);
          gsl_vector_scale(dx, 1.0 / (NUM_INIT_SAMPLES - 1));
          gsl_vector_add(diag, dx);
        }
        else {
          gsl_blas_dger(1.0 / (NUM_INIT_SAMPLES - 1),
              dx, dx, cov);
        }
      }
      
      // Diagonal correction
      for (k = 0; k < model->dim; k++) {
        if (model->cov_diag) {
          *gsl_vector_ptr(diag, k) += SMALL_DIAGONAL_CORRECTION;
        }
        else {
          *gsl_matrix_ptr(cov, k, k) += SMALL_DIAGONAL_CORRECTION;
        }
      }
    }
  }

  gsl_vector_free(dx);
}

void kmeans_init(hmmgmm_t* model, seq_t** data, size_t nos,
    gsl_rng* rng) {
  assert(nos > 0);

  size_t i, j, k;
  double p, sum;
  
  sum = 0.0;
  for (i = 0; i < model->n; i++) {
    p = 0.75 + 0.5 * gsl_rng_uniform(rng);
    gsl_vector_set(model->pi, i, p);
    sum += p;
  }
  gsl_vector_scale(model->pi, 1.0 / sum);

  for (j = 0; j < model->n; j++) {
    gsl_vector_view v = gsl_matrix_row(model->a, j);
    sum = 0.0;
    for (i = 0; i < model->n; i++) {
      p = 0.75 + 0.5 * gsl_rng_uniform(rng);
      gsl_vector_set(&v.vector, i, p);
      sum += p;
    }
    gsl_vector_scale(&v.vector, 1.0 / sum);
  }

  size_t total = 0;
  for (i = 0; i < nos; i++) {
    total += data[i]->size;
  }

  gsl_vector** all = malloc(total * sizeof(gsl_vector*));
  gsl_vector** sorted = malloc(total * sizeof(gsl_vector*));
  j = 0;
  for (i = 0; i < nos; i++) {
    memcpy(all + j, data[i]->data,
        data[i]->size * sizeof(gsl_vector*));
    j += data[i]->size;
  }
  assert(j == total);

  size_t* states = malloc(total * sizeof(size_t));
  size_t* comps = malloc(total * sizeof(size_t));

  kmeans_cluster(all, total, model->n, states, NULL);

  gsl_vector* dx = gsl_vector_alloc(model->dim);
  size_t e = 0;
  size_t s;
  for (i = 0; i < model->n; i++) {
    s = e;
    for (j = 0; j < total; j++) {
      if (states[j] == i) {
        sorted[e] = all[j];
        e++;
      }
    }
    
    assert(e - s > 0);

    kmeans_cluster(sorted + s, e - s, model->k, comps + s, NULL);
    for (j = 0; j < model->k; j++) {
      gsl_vector* mean = model->states[i]->comp[j]->mean;
      gsl_vector* diag = model->states[i]->comp[j]->diag;
      gsl_matrix* cov = model->states[i]->comp[j]->cov;

      gsl_vector_set_zero(mean);

      if (model->cov_diag) {
        gsl_vector_set_zero(diag);
      }
      else {
        gsl_matrix_set_zero(cov);
      }

      size_t c = 0;
      for (k = s; k < e; k++) {
        if (comps[k] == j) {
          gsl_vector_add(mean, sorted[k]);
          c++;
        }
      }
      assert(c > 0);
      gsl_vector_scale(mean, 1.0 / c);

      gsl_vector_set(model->states[i]->weight, j,
          (double) c / (double) (e - s));

      if (c > 1) {
        for (k = s; k < e; k++) {
          if (comps[k] == j) {
            gsl_vector_memcpy(dx, sorted[k]);
            gsl_vector_sub(dx, mean);
            if (model->cov_diag) {
              gsl_vector_mul(dx, dx);
              gsl_vector_scale(dx, 1.0 / (c - 1));
              gsl_vector_add(diag, dx);
            }
            else {
              gsl_blas_dger(1.0 / (c - 1), dx, dx, cov);
            }
          }
        }
      }

      // Diagonal correction
      for (k = 0; k < model->dim; k++) {
        if (model->cov_diag) {
          *gsl_vector_ptr(diag, k) += SMALL_DIAGONAL_CORRECTION;
        }
        else {
          *gsl_matrix_ptr(cov, k, k) += SMALL_DIAGONAL_CORRECTION;
        }
      }
    }
  }

  assert(e == total);

  free(all);
  free(sorted);
  free(states);
  free(comps);
  gsl_vector_free(dx);
}

void baum_welch(hmmgmm_t* model, seq_t** data, size_t nos) {
  hmmgmm_t* nmodel = hmmgmm_alloc(model->n, model->k,
      model->dim, model->cov_diag);
  gsl_matrix* loga = gsl_matrix_alloc(model->n, model->n);
  gsl_matrix* scgamma = gsl_matrix_alloc(model->n, model->k);

  size_t i, j, k, s, t;
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
        if (model->cov_diag) {
          gsl_vector_set_zero(state->comp[j]->diag);
        }
        else {
          gsl_matrix_set_zero(state->comp[j]->cov);
        }
      }
    }

    for (i = 0; i < model->n; i++) {
      for (j = 0; j < model->n; j++) {
        gsl_matrix_set(loga, i, j,
            DEBUG_LOG(gsl_matrix_get(model->a, i, j)));
      }
    }

    gsl_matrix_set_zero(scgamma);

#pragma omp parallel private(i, j, s, t) shared(loga) reduction(+:slogpo)
    {
      gsl_vector* gamma = gsl_vector_alloc(model->n);
      gsl_vector* cgamma = gsl_vector_alloc(model->k);
      gsl_matrix* xi = gsl_matrix_alloc(model->n, model->n);
      gsl_vector* mean = gsl_vector_alloc(model->dim);

      // Accumulate
#pragma omp for
      for (s = 0; s < nos; s++) { // for all sequences
        gsl_matrix* logalpha = gsl_matrix_alloc(data[s]->size,
            model->n);
        gsl_matrix* logbeta = gsl_matrix_alloc(data[s]->size,
            model->n);

        forward_proc_log(model, data[s], logalpha);
        backward_proc_log(model, data[s], logbeta);

        double logpo = hmm_log_likelihood(logalpha);
        assert(isfinite(logpo));
        slogpo += logpo;

        for (t = 0; t < data[s]->size; t++) { // for all time
          for (i = 0; i < model->n; i++) {
            // Calculate gamma
            double gi = DEBUG_EXP(-logpo
                + gsl_matrix_get(logalpha, t, i)
                + gsl_matrix_get(logbeta, t, i));
            assert(gi >= 0.0 && gi <= MORE_THAN_ONE);
            gsl_vector_set(gamma, i, gi);
          }
          // is it necessary to normalize gamma ? 
          // cause there is arithmetic error

          if (t == 0) {
            // Accumulate pi
#pragma omp critical
            {
              gsl_vector_add(nmodel->pi, gamma);
            }
          }

          if (t < data[s]->size - 1) {
            // Caculate xi
            for (j = 0; j < model->n; j++) {
              double logb = gmm_pdf_log(model->states[j],
                  data[s]->data[t + 1]);
              for (i = 0; i < model->n; i++) {
                gsl_matrix_set(xi, i, j, DEBUG_EXP(
                      gsl_matrix_get(logalpha, t, i)
                      + gsl_matrix_get(loga, i, j)
                      + gsl_matrix_get(logbeta, t + 1, j)
                      + logb - logpo));
              }
            }

            // Accumulate a
            // Here we don't divide by gamma 
            // because we will normalize (sum is 1) a later.
#pragma omp critical
            {
              gsl_matrix_add(nmodel->a, xi);
            }
          }

          for (i = 0; i < model->n; i++) { // go through all states
            gmm_t* state = model->states[i];
            gmm_t* nstate = nmodel->states[i];

            // Calculate cgamma of mixture components
            for (j = 0; j < model->k; j++) {
              gsl_vector_set(cgamma, j, 
                  gsl_vector_get(state->weight, j)
                  * DEBUG_EXP(gaussian_pdf_log(state->comp[j],
                      data[s]->data[t])));
            }

            double sum = gsl_blas_dasum(cgamma);
            assert(isfinite(sum));
            if (sum > MORE_THAN_ZERO && isfinite(1.0 / sum)) {
              // Normalize and multiply by gamma
              gsl_vector_scale(cgamma, 
                  gsl_vector_get(gamma, i) / sum);

              // Accumulate mixture weight
#pragma omp critical
              {
                gsl_vector_add(nstate->weight, cgamma);
              }

              // Accumulate mean and cov
              for (j = 0; j < model->k; j++) {
                gsl_vector_memcpy(mean, data[s]->data[t]);
                gsl_vector_scale(mean, gsl_vector_get(cgamma, j));
#pragma omp critical
                {
                  gsl_vector_add(nstate->comp[j]->mean, mean);
                }

                gsl_vector_memcpy(mean, data[s]->data[t]);
                gsl_vector_sub(mean, state->comp[j]->mean);
                if (model->cov_diag) {
                  gsl_vector_mul(mean, mean);
                  gsl_vector_scale(mean, gsl_vector_get(cgamma, j));
#pragma omp critical
                  {
                    gsl_vector_add(nstate->comp[j]->diag, mean);
                  }
                }
                else {
#pragma omp critical
                  {
                    gsl_blas_dger(gsl_vector_get(cgamma, j), mean,
                        mean, nstate->comp[j]->cov);
                  }
                }
#pragma omp critical
                {
                  *gsl_matrix_ptr(scgamma, i, j) 
                    += gsl_vector_get(cgamma, j);
                }
              }
            }
          } // end for all states
        } // end for all time

        gsl_matrix_free(logalpha);
        gsl_matrix_free(logbeta);
      } // end for all sequences

      gsl_vector_free(gamma);
      gsl_vector_free(cgamma);
      gsl_matrix_free(xi);
      gsl_vector_free(mean);
    }
    // end of parallel

    // Normalize
    // Normalize pi, sum = 1
    gsl_vector_scale(nmodel->pi, 
        1.0 / gsl_blas_dasum(nmodel->pi));

    double scale;
    for (i = 0; i < model->n; i++) {
      // Normalize a, sum of each row = 1
      gsl_vector_view v = gsl_matrix_row(nmodel->a, i);
      scale = gsl_blas_dasum(&v.vector);
      if (scale > MORE_THAN_ZERO && isfinite(1.0 / scale)) {
        gsl_vector_scale(&v.vector, 1.0 / scale);

        gmm_t* state = nmodel->states[i];

        // Normalize weight, sum = 1
        double wnorm = gsl_blas_dasum(state->weight);
        if (wnorm == 0 || !isfinite(wnorm)) {
          fprintf(stderr, "Warning: Abnormal norm of weight = %g\n", wnorm);
        }
        gsl_vector_scale(state->weight, 
            1.0 / wnorm);

        for (j = 0; j < model->k; j++) {
          scale = gsl_matrix_get(scgamma, i, j);
          if (scale > MORE_THAN_ZERO && isfinite(1.0 / scale)) {
            // Normalize (rescale) mean
            gsl_vector_scale(state->comp[j]->mean, 1.0 / scale);

            // Normalize (rescale) cov
            if (model->cov_diag) {
              gsl_vector_scale(state->comp[j]->diag, 1.0 / scale);
            }
            else {
              gsl_matrix_scale(state->comp[j]->cov, 1.0 / scale);
            }

            // Diagonal correction
            for (k = 0; k < model->dim; k++) {
              if (model->cov_diag) {
                *gsl_vector_ptr(state->comp[j]->diag, k)
                  += SMALL_DIAGONAL_CORRECTION;
              }
              else {
                *gsl_matrix_ptr(state->comp[j]->cov, k, k) 
                  += SMALL_DIAGONAL_CORRECTION;
              }
            }
          }
          else {
            fprintf(stderr, "Warning: state %zu, component %zu "
                "not visited\n", i, j);
            // Copy from original model
            gaussian_memcpy(state->comp[j],
                model->states[i]->comp[j]);
          }
        }
      }
      else {
        fprintf(stderr, "Warning: state %zu not visited\n", i);
        // Copy from original model
        gsl_vector_view vo = gsl_matrix_row(model->a, i);
        gsl_vector_memcpy(&v.vector, &vo.vector);
        gmm_memcpy(nmodel->states[i], model->states[i]);
      }
    }

    // Replace the original
    hmmgmm_memcpy(model, nmodel);

    // Normalize slogp
    slogpo /= (double) nos;

    // Here the log p is of the previous model
    // And the difference is calculated with 
    //   the previous previous model
    fprintf(stderr, "Iteration %d: log p = %g, "
        "difference = %g\n", iter, slogpo, slogpo - plogpo);
    iter++;

    // assert(slogpo >= plogpo);
    if (slogpo < plogpo) {
      fprintf(stderr, "Warning: reestimation doesn't increase likelihood\n");
    }
  } while (slogpo - plogpo > BW_STOP_THRESHOLD);

  hmmgmm_free(nmodel);
  gsl_matrix_free(loga);
  gsl_matrix_free(scgamma);
}

