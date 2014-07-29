// genseq_test.c
// Test the generate sequence function.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"

#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
  hmmgmm_t* model = hmmgmm_alloc(2, 1, 2);

  gsl_vector_set(model->pi, 0, 1.0);
  gsl_vector_set(model->pi, 1, 0.0);

  gsl_matrix_set(model->a, 0, 0, .2);
  gsl_matrix_set(model->a, 0, 1, .8);
  gsl_matrix_set(model->a, 1, 0, .8);
  gsl_matrix_set(model->a, 1, 1, .2);

  gsl_vector_set(model->states[0]->weight, 0, 1.0);
  gsl_vector_set(model->states[1]->weight, 0, 1.0);

  gmm_t* state = model->states[0];
  gsl_vector_set(state->comp[0]->mean, 0, 0.0);
  gsl_vector_set(state->comp[0]->mean, 1, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 0, 1.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 1, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 0, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 1, 1.0);

  state = model->states[1];
  gsl_vector_set(state->comp[0]->mean, 0, 1.0);
  gsl_vector_set(state->comp[0]->mean, 1, 1.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 0, 2.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 1, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 0, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 1, 4.0);

  hmmgmm_t* model2 = hmmgmm_alloc(2, 1, 2);

  gsl_vector_set(model2->pi, 0, 1.0);
  gsl_vector_set(model2->pi, 1, 0.0);

  gsl_matrix_set(model2->a, 0, 0, .9);
  gsl_matrix_set(model2->a, 0, 1, .1);
  gsl_matrix_set(model2->a, 1, 0, .3);
  gsl_matrix_set(model2->a, 1, 1, .7);

  gsl_vector_set(model2->states[0]->weight, 0, 1.0);
  gsl_vector_set(model2->states[1]->weight, 0, 1.0);

  state = model2->states[0];
  gsl_vector_set(state->comp[0]->mean, 0, 0.0);
  gsl_vector_set(state->comp[0]->mean, 1, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 0, 1.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 1, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 0, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 1, 1.0);

  state = model2->states[1];
  gsl_vector_set(state->comp[0]->mean, 0, 1.0);
  gsl_vector_set(state->comp[0]->mean, 1, 1.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 0, 2.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 1, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 0, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 1, 4.0);


  int size = 1000;
  seq_t* seq = seq_gen(model, size);

  gsl_matrix* alpha = gsl_matrix_alloc(size, model->n);
  forward_proc(model, seq, alpha);

  gsl_matrix* logalpha = gsl_matrix_alloc(size, model->n);
  forward_proc_log(model, seq, logalpha);

  gsl_matrix* logalpha2 = gsl_matrix_alloc(size, model2->n);
  forward_proc_log(model2, seq, logalpha2);

  int i;
  for (i = 0; i < size; i++) {
    printf("%g %g; %g %g; %g %g; %g %g\n",
        gsl_vector_get(seq->data[i], 0),
        gsl_vector_get(seq->data[i], 1),
        log(gsl_matrix_get(alpha, i, 0)),
        log(gsl_matrix_get(alpha, i, 1)),
        gsl_matrix_get(logalpha, i, 0),
        gsl_matrix_get(logalpha, i, 1),
        gsl_matrix_get(logalpha2, i, 0),
        gsl_matrix_get(logalpha2, i, 1));
  }

  gsl_matrix_free(alpha);
  gsl_matrix_free(logalpha);
  gsl_matrix_free(logalpha2);
  hmmgmm_free(model);
  hmmgmm_free(model2);

  return 0;
}

