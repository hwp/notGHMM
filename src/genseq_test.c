// genseq_test.c
// Test the generate sequence function.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"

#include <stdio.h>

int main(int argc, char** argv) {
  hmmgmm_t* model = hmmgmm_alloc(2, 1, 2);

  gsl_vector_set(model->pi, 0, .4);
  gsl_vector_set(model->pi, 1, .6);

  gsl_matrix_set(model->a, 0, 0, .2);
  gsl_matrix_set(model->a, 0, 1, .8);
  gsl_matrix_set(model->a, 1, 0, .3);
  gsl_matrix_set(model->a, 1, 1, .7);

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
  gsl_vector_set(state->comp[0]->mean, 0, 5.0);
  gsl_vector_set(state->comp[0]->mean, 1, 10.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 0, 2.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 1, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 0, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 1, 4.0);

  int size = 1000;
  seq_t* seq = seq_gen(model, size);
  int i;
  for (i = 0; i < size; i++) {
    printf("%g %g\n", gsl_vector_get(seq->data[i], 0),
        gsl_vector_get(seq->data[i], 1));
  }

  hmmgmm_free(model);

  return 0;
}

